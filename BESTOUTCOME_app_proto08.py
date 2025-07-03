

import sys
sys.modules['torch.classes'] = None  # fix torch Streamlit bug

import streamlit as st
import pdfplumber
import re
from pathlib import Path
import pandas as pd
import os
import pickle
import torch
import nltk
import shutil
import logging
import concurrent.futures

from haystack.schema import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder

# Silence PDF extraction warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

RAG_INDEX_DIR = "./faiss_index"
UPLOAD_DIR = "./uploaded_pdfs"
RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
HF_MODEL = "google/flan-t5-base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
load_nltk_resources()

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(HF_MODEL)

@st.cache_resource
def load_model():
    return AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL,
                                                 torch_dtype=torch.float16).to(device)
                                                #  device_map="auto")

@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANK_MODEL)

if not os.path.exists(RAG_INDEX_DIR):
    os.makedirs(RAG_INDEX_DIR)
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def extract_speakers(fpath: Path) -> pd.DataFrame:
    data = []
    current = None
    with pdfplumber.open(fpath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split('\n'):
                match = SPEAKER_PATTERN.match(line.strip())
                if match:
                    if current:
                        data.append(current)
                    current = {"speaker": match.group("speaker"),
                               "timestamp": match.group("timestamp"),
                               "content": "",
                               "source": fpath.name}
                elif current:
                    current["content"] += " " + line.strip()
                    
    if current:
        data.append(current)
    return pd.DataFrame(data)

def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
    doc_store = FAISSDocumentStore(embedding_dim=768,
                                   faiss_index_factory_str="Flat",
                                   sql_url="sqlite:///expert_rag.db")

    retriever = EmbeddingRetriever(document_store=doc_store,
                                   embedding_model=embedding_model,
                                   use_gpu=torch.cuda.is_available())

    documents = []
    for root, _, files in os.walk(doc_dir):
        for fname in files:
            if not fname.lower().endswith((".pdf", ".txt", ".html")):
                continue
            fpath = os.path.join(root, fname)
            try:
                df = extract_speakers(Path(fpath))
            except Exception as e:
                print(f"Failed to extract from {fpath}: {e}")
                continue
            rows = df.to_dict("records")
            i = 0
            while i < len(rows):
                chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
                meta = {"source": rows[i]["source"], "speaker": rows[i]["speaker"], "timestamp": rows[i]["timestamp"]}
                if i+1 < len(rows) and rows[i+1]['speaker'] == "Expert":
                    chunk_text += f"\n{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
                    meta['speaker'] += f", {rows[i+1]['speaker'] }"
                    meta['timestamp'] += f", {rows[i+1]['timestamp']}"
                    i += 1
                documents.append(Document(content=chunk_text, meta=meta))
                i += 1

    preprocessor = PreProcessor(clean_empty_lines=True,
                                clean_whitespace=True,
                                clean_header_footer=True,
                                split_length=192, # 150
                                split_overlap=30,
                                split_respect_sentence_boundary=True,
                                language="en")

    chunks = preprocessor.process(documents)
    doc_store.write_documents(chunks)
    doc_store.update_embeddings(retriever)
    doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
                   config_path=os.path.join(index_dir, "faiss_config.json"))
    with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    return doc_store, retriever, chunks

def expert_question(question, retriever, documents, tokenizer, model, reranker, top_k=8, max_input_tokens=768):
    retrieved_docs = retriever.retrieve(question, top_k=16)
    pairs = [[question, doc.content] for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:top_k]]

    context = "\n\n".join([doc.content for doc in top_docs])
    prompt = (f"You are an analyst summarizing expert calls about Uber. "
              f"Use only the context from the transcript below. "
              f"If the context includes details like revenue share, driver pay, complaints, or incentives, summarize those. "
              f"If the context does not contain an answer, say so.\n\n"
              f"Context:\n{context}\n\n"
              f"Question:\n{question}\n\n"
              f"Answer:")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)

    def safe_generate():
        try:
            if torch.cuda.is_available():
                print(f"[INFO] GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                print(f"[INFO] GPU memory reserved:  {torch.cuda.memory_reserved() / 1e6:.2f} MB")

            with torch.no_grad():
                return model.generate(**inputs,
                                      pad_token_id=tokenizer.pad_token_id,
                                      max_new_tokens=128,
                                      num_beams=3,
                                      temperature=0.3,
                                      top_p=0.8)
        except Exception as e:
            print(f"[safe_generate ERROR]: {e}")
            raise e

    try:
        output = safe_generate()
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future = executor.submit(safe_generate)
        #     output = future.result(timeout=45)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip(), top_docs, scores[:top_k]
    except Exception as e:
        return f"[ERROR during generation: {e}]", top_docs, scores[:top_k]

st.title("AI Expert Call Query Tool")

st.markdown("""**Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
               **Step 2:** After upload and indexing, you can ask questions about the content.""")

if 'rag_built' not in st.session_state:
    st.session_state.rag_built = False

uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"], key="upload")

if uploaded and not st.session_state.rag_built:
    for f in uploaded:
        with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
            out.write(f.read())
    with st.spinner("Building FAISS index from uploaded documents..."):
        doc_store, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
        tokenizer = load_tokenizer()
        model = load_model()
        reranker = load_reranker()
        st.session_state.retriever = retriever
        st.session_state.expert_docs = expert_docs
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.reranker = reranker
        st.session_state.rag_built = True
        st.success("Index built successfully.")

question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built, key="question_input")
top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built, key="slider_chunks")

if st.button("Get Answer", disabled=not st.session_state.rag_built, key="get_answer") and question.strip():
    with st.spinner("Retrieving answer..."):
        answer, retrieved_docs, scores = expert_question(question,
                                                         st.session_state.retriever,
                                                         st.session_state.expert_docs,
                                                         st.session_state.tokenizer,
                                                         st.session_state.model,
                                                         st.session_state.reranker,
                                                         top_k=top_k)

        st.markdown(f"**Answer:** {answer}")

        with st.expander("Show retrieved context"):
            for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
                meta = doc.meta
                st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}] — Score: {score:.4f}")
                st.write(doc.content)



