

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

# Streamlit UI
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



# "What did the expert say about Uber's margins in Q1?"
# "What did the expert say about Uber's driver incentives in Q1?"
# "Was there any mention of regulatory risks?"
# "How did the expert assess Uber’s competitive positioning?"
# "Do you think Uber will partner with Waymo?"





# import sys
# sys.modules['torch.classes'] = None  # Fix torch Streamlit bug

# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil
# import logging
# import concurrent.futures

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import CrossEncoder

# # Silence PDF extraction warnings
# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# def load_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "google/flan-t5-base"
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# if not os.path.exists(RAG_INDEX_DIR):
#     os.makedirs(RAG_INDEX_DIR)
# if not os.path.exists(UPLOAD_DIR):
#     os.makedirs(UPLOAD_DIR)

# def extract_speakers(fpath: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(fpath) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             for line in text.split('\n'):
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {
#                         "speaker": match.group("speaker"),
#                         "timestamp": match.group("timestamp"),
#                         "content": "",
#                         "source": fpath.name
#                     }
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat",
#         sql_url="sqlite:///expert_rag.db"
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )
#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"], "speaker": rows[i]["speaker"], "timestamp": rows[i]["timestamp"]}
#                 if i+1 < len(rows) and rows[i+1]['speaker'] == "Expert":
#                     chunk_text += f"\n{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker'] }"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(
#         clean_empty_lines=True,
#         clean_whitespace=True,
#         clean_header_footer=True,
#         split_length=150,
#         split_overlap=30,
#         split_respect_sentence_boundary=True,
#         language="en"
#     )
#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     return doc_store, retriever, chunks

# def expert_question(question, retriever, documents, tokenizer, model, reranker, top_k=8, max_input_tokens=768):
#     retrieved_docs = retriever.retrieve(question, top_k=16)
#     pairs = [[question, doc.content] for doc in retrieved_docs]
#     scores = reranker.predict(pairs)
#     scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
#     top_docs = [doc for doc, score in scored_docs[:top_k]]

#     context = "\n\n".join([doc.content for doc in top_docs])
#     prompt = (f"You are an analyst summarizing expert calls about Uber. "
#               f"Use only the context from the transcript below. "
#               f"If the context includes details like revenue share, driver pay, complaints, or incentives, summarize those. "
#               f"If the context does not contain an answer, say so.\n\n"
#               f"Context:\n{context}\n\n"
#               f"Question:\n{question}\n\n"
#               f"Answer:")

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)

#     def safe_generate():
#         with torch.no_grad():
#             return model.generate(
#                 **inputs,
#                 pad_token_id=tokenizer.pad_token_id,
#                 max_new_tokens=128,
#                 num_beams=2,
#                 temperature=0.3,
#                 top_p=0.95
#             )

#     try:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(safe_generate)
#             output = future.result(timeout=45)
#         return tokenizer.decode(output[0], skip_special_tokens=True).strip(), top_docs, scores[:top_k]
#     except Exception as e:
#         return f"[ERROR during generation: {e}]", top_docs, scores[:top_k]

# # Streamlit UI
# st.title("AI Expert Call Query Tool")

# st.markdown("""
# **Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
# **Step 2:** After upload and indexing, you can ask questions about the content.
# """)

# if 'rag_built' not in st.session_state:
#     st.session_state.rag_built = False

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"], key="upload")

# if uploaded and not st.session_state.rag_built:
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         doc_store, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, torch_dtype=torch.float16, device_map="auto")
#         reranker = CrossEncoder(RERANK_MODEL)
#         st.session_state.retriever = retriever
#         st.session_state.expert_docs = expert_docs
#         st.session_state.tokenizer = tokenizer
#         st.session_state.model = model
#         st.session_state.reranker = reranker
#         st.session_state.rag_built = True
#         st.success("Index built successfully.")

# question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built, key="question_input")
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built, key="slider_chunks")

# if st.button("Get Answer", disabled=not st.session_state.rag_built, key="get_answer") and question.strip():
#     with st.spinner("Retrieving answer..."):
#         answer, retrieved_docs, scores = expert_question(
#             question,
#             st.session_state.retriever,
#             st.session_state.expert_docs,
#             st.session_state.tokenizer,
#             st.session_state.model,
#             st.session_state.reranker,
#             top_k=top_k
#         )
#         st.markdown(f"**Answer:** {answer}")

#         with st.expander("Show retrieved context"):
#             for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
#                 meta = doc.meta
#                 st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}] — Score: {score:.4f}")
#                 st.write(doc.content)








# WORKS w/ half good half bad generation and breaks after generating 4 answers

# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil
# import logging
# import concurrent.futures

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import CrossEncoder

# # Silence PDF extraction warnings
# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# # Ensure NLTK is ready
# def load_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "google/flan-t5-base" # "MBZUAI/LaMini-Flan-T5-783M"  
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d, exist_ok=True)

# def extract_speakers(fpath: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(fpath) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             for line in text.split('\n'):
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {"speaker": match.group("speaker"),
#                                "timestamp": match.group("timestamp"),
#                                "content": "",
#                                "source": fpath.name}
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat",
#         sql_url="sqlite:///expert_rag.db"
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )
#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"], "speaker": rows[i]["speaker"], "timestamp": rows[i]["timestamp"]}
#                 if i+1 < len(rows) and rows[i+1]['speaker'] == "Expert":
#                     chunk_text += f"\n{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(
#         clean_empty_lines=True,
#         clean_whitespace=True,
#         clean_header_footer=True,
#         split_length=150,
#         split_overlap=30,
#         split_respect_sentence_boundary=True,
#         language="en"
#     )
#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     return doc_store, retriever, chunks

# def expert_question(question, retriever, documents, tokenizer, model, top_k=8, max_input_tokens=768):
#     retrieved_docs = retriever.retrieve(question, top_k=16)
#     reranker = CrossEncoder(RERANK_MODEL)
#     pairs = [[question, doc.content] for doc in retrieved_docs]
#     scores = reranker.predict(pairs)
#     scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
#     top_docs = [doc for doc, score in scored_docs[:top_k]]

#     context = "\n\n".join([doc.content for doc in top_docs])
#     prompt = (f"You are an analyst summarizing expert calls about Uber. "
#               f"Use only the context from the transcript below. "
#               f"If the context includes details like revenue share, driver pay, complaints, or incentives, summarize those. "
#               f"If the context does not contain an answer, say so.\n\n"
#               f"Context:\n{context}\n\n"
#               f"Question:\n{question}\n\n"
#               f"Answer:")

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)

#     def safe_generate():
#         with torch.no_grad():
#             return model.generate(
#                 **inputs,
#                 pad_token_id=tokenizer.pad_token_id,
#                 max_new_tokens=128,
#                 num_beams=2
#             )

#     try:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(safe_generate)
#             output = future.result(timeout=45)  # seconds
#         return tokenizer.decode(output[0], skip_special_tokens=True).strip(), top_docs, scores[:top_k]
#     except Exception as e:
#         return f"[ERROR during generation: {e}]", top_docs, scores[:top_k]

# # Streamlit UI
# st.title("AI Expert Call Query Tool")

# st.markdown("""
# **Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
# **Step 2:** After upload and indexing, you can ask questions about the content.
# """)

# if 'rag_built' not in st.session_state:
#     st.session_state.rag_built = False

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"], key="upload")

# if uploaded and not st.session_state.rag_built:
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         doc_store, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, torch_dtype=torch.float16, device_map="auto")
#         st.session_state.retriever = retriever
#         st.session_state.expert_docs = expert_docs
#         st.session_state.tokenizer = tokenizer
#         st.session_state.model = model
#         st.session_state.rag_built = True
#         st.success("Index built successfully.")

# question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built, key="question_input")
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built, key="slider_chunks")

# if st.button("Get Answer", disabled=not st.session_state.rag_built, key="get_answer") and question.strip():
#     with st.spinner("Retrieving answer..."):
#         answer, retrieved_docs, scores = expert_question(
#             question,
#             st.session_state.retriever,
#             st.session_state.expert_docs,
#             st.session_state.tokenizer,
#             st.session_state.model,
#             top_k=top_k
#         )
#         st.markdown(f"**Answer:** {answer}")

#         with st.expander("Show retrieved context"):
#             for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
#                 meta = doc.meta
#                 st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}] — Score: {score:.4f}")
#                 st.write(doc.content)









# STALLS and silently terms

# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil
# import logging

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import CrossEncoder

# # Silence PDF extraction warnings
# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# # Ensure NLTK is ready
# def load_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# CAUSAL_HF_MODEL = "Akhil-Theerthala/Kuvera-8B-v0.1.0"
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# # Clean directories
# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d, exist_ok=True)

# def extract_speakers(pdf_path: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             for line in text.split('\n'):
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {
#                         "speaker": match.group("speaker"),
#                         "timestamp": match.group("timestamp"),
#                         "content": "",
#                         "source": pdf_path.name
#                     }
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat",
#         sql_url="sqlite:///expert_rag.db"
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )
#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"], "speaker": rows[i]["speaker"], "timestamp": rows[i]["timestamp"]}
#                 if i+1 < len(rows) and rows[i+1]['speaker'] == "Expert":
#                     chunk_text += f"\n{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(
#         clean_empty_lines=True,
#         clean_whitespace=True,
#         clean_header_footer=True,
#         split_length=150,
#         split_overlap=30,
#         split_respect_sentence_boundary=True,
#         language="en"
#     )
#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     return doc_store, retriever, chunks

# def ask_expert_question(question, retriever, documents, tokenizer, model, top_k=8, max_input_tokens=1024):
#     retrieved_docs = retriever.retrieve(question, top_k=16)
#     reranker = CrossEncoder(RERANK_MODEL)
#     pairs = [[question, doc.content] for doc in retrieved_docs]
#     scores = reranker.predict(pairs)
#     scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
#     top_docs = [doc for doc, score in scored_docs[:top_k]]

#     context = "\n\n".join([doc.content for doc in top_docs])
#     prompt = (f"You are an expert analyst. Only use the context below to answer the question. "
#               f"If the context doesn’t contain an answer, say so.\n\n"
#               f"Context:\n{context}\n\n"
#               f"Question:\n{question}\n\n"
#               f"Answer:")

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             pad_token_id=tokenizer.eos_token_id,
#             max_new_tokens=256,
#             do_sample=False,
#             num_beams=3,
#             temperature=0.7
#         )
#     return tokenizer.decode(output[0], skip_special_tokens=True).strip(), top_docs, scores[:top_k]

# # Streamlit UI
# st.title("AI Expert Call Query Tool")

# st.markdown("""
# **Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
# **Step 2:** After upload and indexing, you can ask questions about the content.
# """)

# if 'rag_built' not in st.session_state:
#     st.session_state.rag_built = False

# # uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"])
# uploaded = st.file_uploader("Upload expert call PDFs",
#                             accept_multiple_files=True,
#                             type=["pdf", "txt", "html"],
#                             key="file_uploader_main")  # added a unique key here

# if uploaded and not st.session_state.rag_built:
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         doc_store, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(CAUSAL_HF_MODEL)
#         model = AutoModelForCausalLM.from_pretrained(CAUSAL_HF_MODEL, torch_dtype=torch.float16, device_map="auto")
#         st.session_state.retriever = retriever
#         st.session_state.expert_docs = expert_docs
#         st.session_state.tokenizer = tokenizer
#         st.session_state.model = model
#         st.session_state.rag_built = True
#         st.success("Index built successfully.")

# question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built)
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built)

# if st.button("Get Answer", disabled=not st.session_state.rag_built) and question.strip():
#     with st.spinner("Retrieving answer..."):
#         answer, retrieved_docs, scores = ask_expert_question(question,
#                                      st.session_state.retriever,
#                                      st.session_state.expert_docs,
#                                      st.session_state.tokenizer,
#                                      st.session_state.model,
#                                      top_k=top_k)
#         st.markdown(f"**Answer:** {answer}")

#         with st.expander("Show retrieved context"):
#             for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
#                 meta = doc.meta
#                 st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}] — Score: {score:.4f}")
#                 st.write(doc.content)

















# app_proto00.py

# import os
# import torch
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document

# # Load model
# HF_MODEL = "teknium/OpenHermes-2.5-Mistral"
# tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
# model = AutoModelForCausalLM.from_pretrained(
#     HF_MODEL,
#     device_map="auto",
#     torch_dtype=torch.float16
# )

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9,
#     repetition_penalty=1.1
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# # Prompt template
# custom_prompt_template = """
# You are an expert call analyst. Use the following transcript content to answer the question directly.

# Context:
# {context}

# Question: {question}
# Answer:
# """

# prompt = PromptTemplate(
#     template=custom_prompt_template,
#     input_variables=["context", "question"]
# )

# # Embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# # Streamlit UI
# st.set_page_config(page_title="Expert Call Q&A Tool")
# st.title("Expert Call Q&A Tool")

# uploaded_files = st.file_uploader("Upload expert call PDFs", type=["pdf", "txt", "html", "htm"], accept_multiple_files=True)

# if uploaded_files:
#     save_dir = "uploaded_docs"
#     os.makedirs(save_dir, exist_ok=True)
#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(save_dir, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())
#     st.success("All files saved. Please re-run the embedding script and refresh.")

# # Load FAISS index
# if os.path.exists("faiss_index"):
#     db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     retriever = db.as_retriever(search_kwargs={"k": 10})

#     st.subheader("Ask a question:")
#     question = st.text_input("What would you like to know?")
#     num_chunks = st.slider("Number of transcript chunks to use", min_value=2, max_value=20, value=10)

#     if question:
#         retrieved_docs = retriever.get_relevant_documents(question)
#         top_docs = retrieved_docs[:num_chunks]
#         context = "\n\n".join([doc.page_content for doc in top_docs])

#         # Build QA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever,
#             chain_type_kwargs={"prompt": prompt}
#         )

#         # Run query
#         answer = qa_chain.run(question)

#         st.markdown(f"**Answer:** {answer}")

#         with st.expander("Show retrieved context"):
#             for i, doc in enumerate(top_docs):
#                 st.markdown(f"**Chunk {i+1}:** {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}\n")
# else:
#     st.warning("Please make sure the FAISS index exists and contains the embedded transcripts.")








# //////////////////////////////////////////////////////////////////////////////////////////////



# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil
# import logging

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import CrossEncoder

# # Silence PDF extraction warnings
# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# # Ensure NLTK is ready
# def load_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "MBZUAI/LaMini-Flan-T5-783M" # "teknium/OpenHermes-2.5-Mistral"
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# # Clean directories
# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d, exist_ok=True)

# def extract_speakers(pdf_path: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             for line in text.split('\n'):
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {
#                         "speaker": match.group("speaker"),
#                         "timestamp": match.group("timestamp"),
#                         "content": "",
#                         "source": pdf_path.name
#                     }
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat",
#         sql_url="sqlite:///expert_rag.db"
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )
#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"], "speaker": rows[i]["speaker"], "timestamp": rows[i]["timestamp"]}
#                 if i+1 < len(rows) and rows[i+1]['speaker'] == "Expert":
#                     chunk_text += f"\n{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(
#         clean_empty_lines=True,
#         clean_whitespace=True,
#         clean_header_footer=True,
#         split_length=150,
#         split_overlap=30,
#         split_respect_sentence_boundary=True,
#         language="en"
#     )
#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     return doc_store, retriever, chunks

# def ask_expert_question(question, retriever, documents, tokenizer, model, top_k=8, max_input_tokens=1024):
#     retrieved_docs = retriever.retrieve(question, top_k=16)
#     reranker = CrossEncoder(RERANK_MODEL)
#     pairs = [[question, doc.content] for doc in retrieved_docs]
#     scores = reranker.predict(pairs)
#     scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
#     top_docs = [doc for doc, score in scored_docs[:top_k]]

#     context = " \n\n".join([doc.content for doc in top_docs])
#     prompt = (f"You are an analyst summarizing expert calls about Uber. "
#               f"Use only the context from the transcript below. "
#               f"If the context includes details like revenue share, driver pay, complaints, or incentives, summarize those. "
#               f"If the context does not contain an answer, say so.\n\n"
#               f"Context:\n{context}\n\n"
#               f"Question:\n{question}\n\n"
#               f"Answer:")

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
#     with torch.no_grad():
#         output = model.generate(**inputs,
#                                 pad_token_id=tokenizer.pad_token_id,
#                                 max_new_tokens=256,
#                                 num_beams=2,
#                                 temperature=0.7)
#     return tokenizer.decode(output[0], skip_special_tokens=True).strip(), top_docs, scores[:top_k]

# # Streamlit UI
# st.title("AI Expert Call Query Tool")

# st.markdown("""
# **Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
# **Step 2:** After upload and indexing, you can ask questions about the content.
# """)

# if 'rag_built' not in st.session_state:
#     st.session_state.rag_built = False

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"])

# if uploaded and not st.session_state.rag_built:
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         doc_store, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, torch_dtype=torch.float16, device_map="auto")
#         st.session_state.retriever = retriever
#         st.session_state.expert_docs = expert_docs
#         st.session_state.tokenizer = tokenizer
#         st.session_state.model = model
#         st.session_state.rag_built = True
#         st.success("Index built successfully.")

# question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built)
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built)

# if st.button("Get Answer", disabled=not st.session_state.rag_built) and question.strip():
#     with st.spinner("Retrieving answer..."):
#         answer, retrieved_docs, scores = ask_expert_question(question,
#                                      st.session_state.retriever,
#                                      st.session_state.expert_docs,
#                                      st.session_state.tokenizer,
#                                      st.session_state.model,
#                                      top_k=top_k)
#         st.markdown(f"**Answer:** {answer}")

#         with st.expander("Show retrieved context"):
#             for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
#                 meta = doc.meta
#                 st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}] — Score: {score:.4f}")
#                 st.write(doc.content)



# //////////////////////////////////////////////////////////////////////////////////////////////








# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil
# import logging

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import CrossEncoder

# # Silence PDF extraction warnings
# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# # Ensure NLTK is ready
# def load_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "MBZUAI/LaMini-Flan-T5-783M"
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# # Clean directories
# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d, exist_ok=True)

# def extract_speakers(pdf_path: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             for line in text.split('\n'):
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {
#                         "speaker": match.group("speaker"),
#                         "timestamp": match.group("timestamp"),
#                         "content": "",
#                         "source": pdf_path.name
#                     }
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat",
#         sql_url="sqlite:///expert_rag.db"
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )
#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"], "speaker": rows[i]["speaker"], "timestamp": rows[i]["timestamp"]}
#                 if i+1 < len(rows) and rows[i+1]['speaker'] == "Expert":
#                     chunk_text += f"\n{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(
#         clean_empty_lines=True,
#         clean_whitespace=True,
#         clean_header_footer=True,
#         split_length=150,
#         split_overlap=30,
#         split_respect_sentence_boundary=True,
#         language="en"
#     )
#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     return doc_store, retriever, chunks

# def expert_question(question, retriever, documents, tokenizer, model, reranker, top_k=8, max_input_tokens=512): # 1024
#     retrieved_docs = retriever.retrieve(question, top_k=top_k * 2)
#     pairs = [[question, doc.content] for doc in retrieved_docs]
#     scores = reranker.predict(pairs)
#     reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)[:top_k]

#     context = "\n\n".join([doc.content for doc, _ in reranked])
#     prompt = (f"You are an analyst summarizing expert calls about Uber. "
#               f"Use only the context from the transcript below. If the context does not contain an answer, say so.\n\n"
#               f"Context:\n{context}\n\n"
#               f"Question:\n{question}\n\n"
#               f"Answer:")
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
#     with torch.no_grad():
#         output = model.generate(**inputs,
#                                 pad_token_id=tokenizer.pad_token_id,
#                                 max_new_tokens=256,
#                                 num_beams=2,
#                                 temperature=0.7)
#     return tokenizer.decode(output[0], skip_special_tokens=True).strip(), reranked

# # Streamlit UI
# st.title("AI Expert Call Query Tool")

# st.markdown("""
# **Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
# **Step 2:** After upload and indexing, you can ask questions about the content.
# """)

# if 'rag_built' not in st.session_state:
#     st.session_state.rag_built = False

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"])

# if uploaded and not st.session_state.rag_built:
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         doc_store, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, torch_dtype=torch.float16, device_map="auto")
#         reranker = CrossEncoder(RERANK_MODEL)
#         st.session_state.retriever = retriever
#         st.session_state.expert_docs = expert_docs
#         st.session_state.tokenizer = tokenizer
#         st.session_state.model = model
#         st.session_state.reranker = reranker
#         st.session_state.rag_built = True
#         st.success("Index built successfully.")

# question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built)
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built)

# if st.button("Get Answer", disabled=not st.session_state.rag_built) and question.strip():
#     with st.spinner("Retrieving answer..."):
#         answer, reranked_docs = expert_question(question,
#                                      st.session_state.retriever,
#                                      st.session_state.expert_docs,
#                                      st.session_state.tokenizer,
#                                      st.session_state.model,
#                                      st.session_state.reranker,
#                                      top_k=top_k)
#         st.markdown(f"**Answer:** {answer}")

#         with st.expander("Show retrieved context with scores"):
#             for i, (doc, score) in enumerate(reranked_docs):
#                 meta = doc.meta
#                 st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}] — Score: {score:.4f}")
#                 st.write(doc.content)


# "What did the expert say about Uber's margins in Q1?"
# "What did the expert say about Uber's driver incentives in Q1?"
# "Was there any mention of regulatory risks?"
# "How did the expert assess Uber’s competitive positioning?
# "Do you think Uber will partner with Waymo?"















# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil
# import logging

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Silence PDF extraction warnings
# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# # Ensure NLTK is ready
# def load_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "MBZUAI/LaMini-Flan-T5-783M"

# # Clean directories
# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d, exist_ok=True)

# def extract_speakers(pdf_path: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             for line in text.split('\n'):
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {
#                         "speaker": match.group("speaker"),
#                         "timestamp": match.group("timestamp"),
#                         "content": "",
#                         "source": pdf_path.name
#                     }
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat",
#         sql_url="sqlite:///expert_rag.db"
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )
#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"], "speaker": rows[i]["speaker"], "timestamp": rows[i]["timestamp"]}
#                 if i+1 < len(rows) and rows[i+1]['speaker'] == "Expert":
#                     chunk_text += f"\n{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(
#         clean_empty_lines=True,
#         clean_whitespace=True,
#         clean_header_footer=True,
#         split_length=150,
#         split_overlap=30,
#         split_respect_sentence_boundary=True,
#         language="en"
#     )
#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     return doc_store, retriever, chunks

# def ask_expert_question(question, retriever, documents, tokenizer, model, top_k=8, max_input_tokens=1024):
#     # Filter chunks based on topic keywords
#     topic_keywords = question.lower().split()
#     filtered_docs = [doc for doc in documents if any(keyword in doc.content.lower() for keyword in topic_keywords)]

#     if not filtered_docs:
#         # fallback if filtering excludes all
#         filtered_docs = documents

#     # Rank using retriever
#     retrieved_docs = retriever.retrieve(question, top_k=top_k, documents=filtered_docs)
#     retrieved_docs = retriever.retrieve(question, top_k=top_k)
#     context = " \n\n".join([doc.content for doc in retrieved_docs])
#     prompt = (f"You are an analyst summarizing expert calls about Uber. "
#               f"Use only the context from the transcript below. If the context does not contain an answer, say so.\n\n"
#               f"Context:\n{context}\n\n"
#               f"Question:\n{question}\n\n"
#               f"Answer:")
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
#     with torch.no_grad():
#         output = model.generate(**inputs,
#                                 pad_token_id=tokenizer.pad_token_id,
#                                 max_new_tokens=256,
#                                 num_beams=2,
#                                 temperature=0.7,
#                                 top_k=top_k)
#     return tokenizer.decode(output[0], skip_special_tokens=True).strip(), retrieved_docs

# # Streamlit UI
# st.title("AI Expert Call Query Tool")

# st.markdown("""
# **Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
# **Step 2:** After upload and indexing, you can ask questions about the content.
# """)

# if 'rag_built' not in st.session_state:
#     st.session_state.rag_built = False

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"])

# if uploaded and not st.session_state.rag_built:
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         doc_store, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, torch_dtype=torch.float16, device_map="auto")
#         st.session_state.retriever = retriever
#         st.session_state.expert_docs = expert_docs
#         st.session_state.tokenizer = tokenizer
#         st.session_state.model = model
#         st.session_state.rag_built = True
#         st.success("Index built successfully.")

# question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built)
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built)

# if st.button("Get Answer", disabled=not st.session_state.rag_built) and question.strip():
#     with st.spinner("Retrieving answer..."):
#         answer, retrieved_docs = ask_expert_question(question,
#                                      st.session_state.retriever,
#                                      st.session_state.expert_docs,
#                                      st.session_state.tokenizer,
#                                      st.session_state.model,
#                                      top_k=top_k)
#         st.markdown(f"**Answer:** {answer}")

#         with st.expander("Show retrieved context"):
#             for i, doc in enumerate(retrieved_docs):
#                 meta = doc.meta
#                 st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}]")
#                 st.write(doc.content)





# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil
# import logging

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Silence PDF extraction warnings
# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# # Ensure NLTK is ready
# def load_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "MBZUAI/LaMini-Flan-T5-783M"

# # Clean directories
# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d, exist_ok=True)

# def extract_speakers(pdf_path: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             for line in text.split('\n'):
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {"speaker": match.group("speaker"),
#                                "timestamp": match.group("timestamp"),
#                                "content": "",
#                                "source": pdf_path.name}

#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     doc_store = FAISSDocumentStore(embedding_dim=768,
#                                    faiss_index_factory_str="Flat",
#                                    sql_url="sqlite:///expert_rag.db")

#     retriever = EmbeddingRetriever(document_store=doc_store,
#                                    embedding_model=embedding_model,
#                                    use_gpu=torch.cuda.is_available())

#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"], "speaker": rows[i]["speaker"], "timestamp": rows[i]["timestamp"]}
#                 if i+1 < len(rows) and rows[i+1]['speaker'] == "Expert":
#                     chunk_text += f"\n{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(clean_empty_lines=True,
#                                 clean_whitespace=True,
#                                 clean_header_footer=True,
#                                 split_length=150,
#                                 split_overlap=30,
#                                 split_respect_sentence_boundary=True,
#                                 language="en")

#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     return doc_store, retriever, chunks

# def expert_question(question, retriever, documents, tokenizer, model, top_k=8, max_input_tokens=1024):
#     retrieved_docs = retriever.retrieve(question, top_k=top_k)
#     context = " ".join([doc.content for doc in retrieved_docs])
#     prompt = (f"You are an analyst summarizing expert calls about Uber. "
#               f"Use only the context from the transcript below. If the context does not contain an answer, say so.\n\n"
#               f"Context:\n{context}\n\n"
#               f"Question:\n{question}\n\n"
#               f"Answer:")
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
#     with torch.no_grad():
#         output = model.generate(**inputs,
#                                 pad_token_id=tokenizer.pad_token_id,
#                                 max_new_tokens=256,
#                                 num_beams=2,
#                                 temperature=0.7,
#                                 top_k=top_k)
#     return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# # Streamlit UI
# st.title("AI Expert Call Query Tool")

# st.markdown("""
# **Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
# **Step 2:** After upload and indexing, you can ask questions about the content.
# """)

# if 'rag_built' not in st.session_state:
#     st.session_state.rag_built = False

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"])

# if uploaded and not st.session_state.rag_built:
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         doc_store, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, torch_dtype=torch.float16, device_map="auto")
#         st.session_state.retriever = retriever
#         st.session_state.expert_docs = expert_docs
#         st.session_state.tokenizer = tokenizer
#         st.session_state.model = model
#         st.session_state.rag_built = True
#         st.success("Index built successfully.")

# question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built)
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built)

# if st.button("Get Answer", disabled=not st.session_state.rag_built) and question.strip():
#     with st.spinner("Retrieving answer..."):
#         answer = expert_question(question,
#                                  st.session_state.retriever,
#                                  st.session_state.expert_docs,
#                                  st.session_state.tokenizer,
#                                  st.session_state.model,
#                                  top_k=top_k)
#         st.markdown(f"**Answer:** {answer}")






# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil
# import glob
# import logging

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# # ////////////////////////////////////////////////////////////////////
# def load_nltk_resources():
#     needed = ['punkt']
#     for res in needed:
#         try:
#             nltk.data.find(f'tokenizers/{res}')
#         except LookupError:
#             nltk.download(res)
# load_nltk_resources()

# # ////////////////////////////////////////////////////////////////////

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "MBZUAI/LaMini-Flan-T5-783M"

# # ////////////////////////////////////////////////////////////////////

# def clean_dir(d):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d, exist_ok=True)

# # Clean on startup
# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     clean_dir(d)

# # ////////////////////////////////////////////////////////////////////

# def print_and_write_index_dir(index_dir):
#     print(f"\n--- Contents of {index_dir}: ---")
#     if not os.path.exists(index_dir):
#         print("Directory does not exist!")
#         return
#     files = os.listdir(index_dir)
#     with open(os.path.join(index_dir, "dir_listing.txt"), "w") as f:
#         for name in files:
#             print(name)
#             f.write(f"{name}\n")
#     print("--- End directory listing ---\n")

# # ////////////////////////////////////////////////////////////////////

# def extract_speakers(pdf_path: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             lines = text.split('\n')
#             for line in lines:
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {"speaker": match.group("speaker"),
#                                "timestamp": match.group("timestamp"),
#                                "content": "",
#                                "source": pdf_path.name}
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# # ////////////////////////////////////////////////////////////////////

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     clean_dir(index_dir)
#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat"
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )

#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#                 if df.empty:
#                     print(f"No speaker data extracted from {fpath}")
#                     continue
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"],
#                         "speaker": rows[i]["speaker"],
#                         "timestamp": rows[i]["timestamp"]}
#                 if i + 1 < len(rows) and rows[i + 1]['speaker'] == "Expert":
#                     chunk_text += "\n" + f"{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     if not documents:
#         raise ValueError("No documents were extracted from the uploaded files")

#     print(f"Extracted {len(documents)} document chunks")
#     preprocessor = PreProcessor(
#         clean_empty_lines=True,
#         clean_whitespace=True,
#         clean_header_footer=True,
#         split_length=150,
#         split_overlap=30,
#         split_respect_sentence_boundary=True,
#         language="en"
#     )
#     chunks = preprocessor.process(documents)
#     print(f"Preprocessed into {len(chunks)} chunks")
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     faiss_index_path = os.path.join(index_dir, "faiss_index.faiss")
#     faiss_config_path = os.path.join(index_dir, "faiss_config.json")
#     doc_store.save(index_path=faiss_index_path, config_path=faiss_config_path)
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     print("FAISS index built and saved successfully.")
#     print_and_write_index_dir(index_dir)
    
#     return doc_store, retriever, chunks

# # ////////////////////////////////////////////////////////////////////

# def load_existing_rag_store(index_dir: str, embedding_model: str):
#     faiss_index_path = os.path.join(index_dir, "faiss_index.faiss")
#     faiss_config_path = os.path.join(index_dir, "faiss_config.json")
#     docs_path = os.path.join(index_dir, "docs.pkl")
#     if not all(os.path.exists(p) for p in [faiss_index_path, faiss_config_path, docs_path]):
#         return None, None, []
#     try:
#         doc_store = FAISSDocumentStore.load(
#             index_path=faiss_index_path,
#             config_path=faiss_config_path
#         )
#         retriever = EmbeddingRetriever(
#             document_store=doc_store,
#             embedding_model=embedding_model,
#             use_gpu=torch.cuda.is_available()
#         )
#         with open(docs_path, "rb") as f:
#             chunks = pickle.load(f)
#         print(f"Loaded existing FAISS store with {len(chunks)} documents")
#         return doc_store, retriever, chunks
#     except Exception as e:
#         print(f"Failed to load existing store: {e}")
#         return None, None, []

# # ////////////////////////////////////////////////////////////////////


# # --- Streamlit UI ---
# st.title("AI Expert Call Query Tool")

# st.markdown("""
# **Step 1:** Upload one or more expert call transcripts (PDF, TXT, or HTML).  
# **Step 2:** After upload and indexing, you can ask questions about the content.
# """)

# if 'rag_built' not in st.session_state:
#     st.session_state.rag_built = False
#     st.session_state.retriever = None
#     st.session_state.expert_docs = []
#     st.session_state.tokenizer = None
#     st.session_state.model = None

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"])

# if uploaded and not st.session_state.rag_built:
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         try:
#             _, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#             tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#             model = AutoModelForSeq2SeqLM.from_pretrained(
#                 HF_MODEL,
#                 torch_dtype=torch.float16,
#                 device_map="auto"
#             ).to("cuda" if torch.cuda.is_available() else "cpu")
#             st.session_state.retriever = retriever
#             st.session_state.expert_docs = expert_docs
#             st.session_state.tokenizer = tokenizer
#             st.session_state.model = model
#             st.session_state.rag_built = True
#             st.success(f"Successfully built index with {len(expert_docs)} document chunks!")
#         except Exception as e:
#             st.error(f"Error building RAG store: {str(e)}")
#             st.stop()

# elif not uploaded:
#     _, retriever, expert_docs = load_existing_rag_store(RAG_INDEX_DIR, RAG_MODEL)
#     if retriever:
#         st.session_state.retriever = retriever
#         st.session_state.expert_docs = expert_docs
#         st.session_state.rag_built = True
#         st.info(f"Loaded existing index with {len(expert_docs)} document chunks")

# question = st.text_input("Ask a question:", disabled=not st.session_state.rag_built)
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6, disabled=not st.session_state.rag_built)

# if st.button("Get Answer", disabled=not st.session_state.rag_built) and question.strip():
#     if st.session_state.retriever:
#         with st.spinner("Retrieving answer..."):
#             try:
#                 retrieved_docs = st.session_state.retriever.retrieve(question, top_k=top_k)
#                 context = "\n".join([doc.content for doc in retrieved_docs])
#                 prompt = ("You are a financial analyst. Use only the transcript context below to answer the question concisely. "
#                           "If the context does not contain the answer, say 'Insufficient information in the expert calls.'\n\n"
#                           f"Context:\n{context}\n\n"
#                           f"Question: {question}\n\n"
#                           f"Answer (1-2 sentences, cite speaker/timestamp):")
#                 if st.session_state.tokenizer and st.session_state.model:
#                     inputs = st.session_state.tokenizer(
#                         prompt,
#                         return_tensors="pt",
#                         truncation=True, 
#                         max_length=768
#                     ).to(st.session_state.model.device)
#                     with torch.no_grad():
#                         output = st.session_state.model.generate(
#                             **inputs,
#                             pad_token_id=st.session_state.tokenizer.pad_token_id,
#                             max_new_tokens=256,
#                             num_beams=3,
#                             temperature=0.4,
#                         )
#                     answer = st.session_state.tokenizer.decode(output[0], skip_special_tokens=True).strip()
#                     st.markdown(f"**Answer:** {answer}")
#                 else:
#                     st.markdown(f"**Context Retrieved:** {len(retrieved_docs)} chunks found")
#                 with st.expander("Show retrieved context"):
#                     for i, doc in enumerate(retrieved_docs):
#                         meta = doc.meta
#                         st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}]")
#                         st.write(doc.content)
#             except Exception as e:
#                 st.error(f"Error during retrieval: {str(e)}")
#     else:
#         st.warning("Please upload PDF files first to build the search index.")

# if st.checkbox("Show Debug Info"):
#     st.write(f"RAG Built: {st.session_state.rag_built}")
#     st.write(f"Retriever Available: {st.session_state.retriever is not None}")
#     st.write(f"Number of Documents: {len(st.session_state.expert_docs)}")
#     if os.path.exists(RAG_INDEX_DIR):
#         st.write(f"Index Directory Contents: {os.listdir(RAG_INDEX_DIR)}")





# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import logging

# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# def load_nltk_resources():
#     needed = ['punkt']
#     for res in needed:
#         try:
#             nltk.data.find(f'tokenizers/{res}')
#         except LookupError:
#             nltk.download(res)
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "MBZUAI/LaMini-Flan-T5-783M"

# # --- ENSURE RAG/UPLOAD DIRS EXIST WITHOUT DELETING ---
# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     os.makedirs(d, exist_ok=True)
#     print(f"Ensured directory exists: {d}")

# def extract_speakers(pdf_path: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             lines = text.split('\n')
#             for line in lines:
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {"speaker": match.group("speaker"),
#                                "timestamp": match.group("timestamp"),
#                                "content": "",
#                                "source": pdf_path.name}
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat"
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )

#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"],
#                         "speaker": rows[i]["speaker"],
#                         "timestamp": rows[i]["timestamp"]}
#                 if i + 1 < len(rows) and rows[i + 1]['speaker'] == "Expert":
#                     chunk_text += "\n" + f"{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(
#         clean_empty_lines=True,
#         clean_whitespace=True,
#         clean_header_footer=True,
#         split_length=150,
#         split_overlap=30,
#         split_respect_sentence_boundary=True,
#         language="en"
#     )
#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     print("FAISS index built and saved.")
#     return doc_store, retriever, chunks

# # --- Streamlit UI ---
# st.title("AI Expert Call Query Tool")

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"])

# if uploaded:
#     # Save all uploaded files to UPLOAD_DIR
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         _, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)

#         # --- MODEL LOAD PATCH ---
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
#         model = model.to(device)
# else:
#     retriever = None
#     expert_docs = []
#     tokenizer = None
#     model = None

# question = st.text_input("Ask a question:")
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6)

# if st.button("Get Answer") and question.strip() and retriever:
#     with st.spinner("Retrieving answer..."):
#         retrieved_docs = retriever.retrieve(question, top_k=top_k)
#         context = "\n".join([doc.content for doc in retrieved_docs])
#         prompt = (
#             "You are a financial analyst. Use only the transcript context below to answer the question concisely. "
#             "If the context does not contain the answer, say 'Insufficient information in the expert calls.'\n\n"
#             f"Context:\n{context}\n\n"
#             f"Question: {question}\n\n"
#             f"Answer (1-2 sentences, cite speaker/timestamp):"
#         )
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
#         with torch.no_grad():
#             output = model.generate(
#                 **inputs,
#                 pad_token_id=tokenizer.pad_token_id,
#                 max_new_tokens=256,
#                 num_beams=3,
#                 temperature=0.4,
#             )
#         answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
#         st.markdown(f"**Answer:** {answer}")
#         with st.expander("Show retrieved context"):
#             for i, doc in enumerate(retrieved_docs):
#                 meta = doc.meta
#                 st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}]")
#                 st.write(doc.content)




# import streamlit as st
# import pdfplumber
# import re
# from pathlib import Path
# import pandas as pd
# import os
# import pickle
# import torch
# import nltk
# import shutil

# from haystack.schema import Document
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import EmbeddingRetriever, PreProcessor
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import logging

# logging.getLogger("pdfminer").setLevel(logging.ERROR)
# logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# def load_nltk_resources():
#     needed = ['punkt']
#     for res in needed:
#         try:
#             nltk.data.find(f'tokenizers/{res}')
#         except LookupError:
#             nltk.download(res)
# load_nltk_resources()

# SPEAKER_PATTERN = re.compile(r"(?P<speaker>Client|Expert)\s+(?P<timestamp>\d{2}:\d{2}:\d{2})")

# RAG_INDEX_DIR = "./faiss_index"
# UPLOAD_DIR = "./uploaded_pdfs"
# RAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL = "MBZUAI/LaMini-Flan-T5-783M"

# # --- AUTOMATED CLEANUP OF RAG/UPLOAD DIRS ON EVERY START ---
# for d in (RAG_INDEX_DIR, UPLOAD_DIR):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#         print(f"Deleted old directory: {d}")
#     os.makedirs(d, exist_ok=True)
#     print(f"Created directory: {d}")

# def extract_speakers(pdf_path: Path) -> pd.DataFrame:
#     data = []
#     current = None
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             lines = text.split('\n')
#             for line in lines:
#                 match = SPEAKER_PATTERN.match(line.strip())
#                 if match:
#                     if current:
#                         data.append(current)
#                     current = {"speaker": match.group("speaker"),
#                                "timestamp": match.group("timestamp"),
#                                "content": "",
#                                "source": pdf_path.name}
#                 elif current:
#                     current["content"] += " " + line.strip()
#     if current:
#         data.append(current)
#     return pd.DataFrame(data)

# def build_rag_store(doc_dir: str, index_dir: str, embedding_model: str):
#     # index_dir is freshly created at app startup

#     doc_store = FAISSDocumentStore(
#         embedding_dim=768,
#         faiss_index_factory_str="Flat"
#         # sql_url is intentionally omitted to avoid SQL/FAISS mismatches
#     )
#     retriever = EmbeddingRetriever(
#         document_store=doc_store,
#         embedding_model=embedding_model,
#         use_gpu=torch.cuda.is_available()
#     )

#     documents = []
#     for root, _, files in os.walk(doc_dir):
#         for fname in files:
#             if not fname.lower().endswith((".pdf", ".txt", ".html")):
#                 continue
#             fpath = os.path.join(root, fname)
#             try:
#                 df = extract_speakers(Path(fpath))
#             except Exception as e:
#                 print(f"Failed to extract from {fpath}: {e}")
#                 continue
#             rows = df.to_dict("records")
#             i = 0
#             while i < len(rows):
#                 chunk_text = f"{rows[i]['speaker']} [{rows[i]['timestamp']}]: {rows[i]['content'].strip()}"
#                 meta = {"source": rows[i]["source"],
#                         "speaker": rows[i]["speaker"],
#                         "timestamp": rows[i]["timestamp"]}
#                 if i + 1 < len(rows) and rows[i + 1]['speaker'] == "Expert":
#                     chunk_text += "\n" + f"{rows[i+1]['speaker']} [{rows[i+1]['timestamp']}]: {rows[i+1]['content'].strip()}"
#                     meta['speaker'] += f", {rows[i+1]['speaker']}"
#                     meta['timestamp'] += f", {rows[i+1]['timestamp']}"
#                     i += 1
#                 documents.append(Document(content=chunk_text, meta=meta))
#                 i += 1

#     preprocessor = PreProcessor(clean_empty_lines=True,
#                                 clean_whitespace=True,
#                                 clean_header_footer=True,
#                                 split_length=150,
#                                 split_overlap=30,
#                                 split_respect_sentence_boundary=True,
#                                 language="en")
#     chunks = preprocessor.process(documents)
#     doc_store.write_documents(chunks)
#     doc_store.update_embeddings(retriever)
#     doc_store.save(index_path=os.path.join(index_dir, "faiss_index.faiss"),
#                    config_path=os.path.join(index_dir, "faiss_config.json"))
#     with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     print("FAISS index built and saved.")
#     return doc_store, retriever, chunks

# # --- Streamlit UI ---
# st.title("AI Expert Call Query Tool")

# uploaded = st.file_uploader("Upload expert call PDFs", accept_multiple_files=True, type=["pdf", "txt", "html"])

# if uploaded:
#     # Save all uploaded files to UPLOAD_DIR
#     for f in uploaded:
#         with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
#             out.write(f.read())
#     with st.spinner("Building FAISS index from uploaded documents..."):
#         _, retriever, expert_docs = build_rag_store(UPLOAD_DIR, RAG_INDEX_DIR, RAG_MODEL)
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL,
#                                                       torch_dtype=torch.float16,
#                                                       device_map="auto").to("cuda" if torch.cuda.is_available() else "cpu")
# else:
#     retriever = None
#     expert_docs = []
#     tokenizer = None
#     model = None

# question = st.text_input("Ask a question:")
# top_k = st.slider("Number of transcript chunks to use", 2, 12, 6)

# if st.button("Get Answer") and question.strip() and retriever:
#     with st.spinner("Retrieving answer..."):
#         retrieved_docs = retriever.retrieve(question, top_k=top_k)
#         context = "\n".join([doc.content for doc in retrieved_docs])
#         prompt = ("You are a financial analyst. Use only the transcript context below to answer the question concisely. "
#                   "If the context does not contain the answer, say 'Insufficient information in the expert calls.'\n\n"
#                   f"Context:\n{context}\n\n"
#                   f"Question: {question}\n\n"
#                   f"Answer (1-2 sentences, cite speaker/timestamp):")
#         inputs = tokenizer(prompt, 
#                            return_tensors="pt", 
#                            truncation=True, max_length=768).to(model.device)
#         with torch.no_grad():
#             output = model.generate(**inputs,
#                                     pad_token_id=tokenizer.pad_token_id,
#                                     max_new_tokens=256,
#                                     num_beams=3,
#                                     temperature=0.4,)
#         answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
#         st.markdown(f"**Answer:** {answer}")
#         with st.expander("Show retrieved context"):
#             for i, doc in enumerate(retrieved_docs):
#                 meta = doc.meta
#                 st.write(f"**Chunk {i+1}:** {meta.get('speaker', '')} [{meta.get('timestamp', '')}]")
#                 st.write(doc.content)
