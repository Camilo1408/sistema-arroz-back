# Dockerfile — versión para Hugging Face Spaces
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces usa el puerto 7860
EXPOSE 7860

CMD ["sh", "-c", "python download_models.py && uvicorn main:app --host 0.0.0.0 --port 7860"]