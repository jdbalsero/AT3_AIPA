FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt')"

COPY src/ src/
COPY .env .

RUN mkdir -p chroma_persistent_storage

# Install Spacy model 
RUN python -m spacy download en_core_web_md

EXPOSE 8501
CMD ["streamlit", "run", "src/app/app.py", "--server.fileWatcherType=none", "--server.port=8501", "--server.address=0.0.0.0"]