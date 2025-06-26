# FROM python:3.10-slim
FROM python:3.10

RUN apt-get update && apt-get install -y build-essential poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app_proto.py ./
COPY requirements.txt ./

RUN pip install --upgrade pip && pip install --upgrade streamlit && pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

EXPOSE 8501

CMD ["streamlit", "run", "app_proto.py", "--server.port=8501", "--server.address=0.0.0.0"]
