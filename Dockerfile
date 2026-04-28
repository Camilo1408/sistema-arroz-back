# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instala gcc y otras herramientas necesarias para compilar
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# Instala dependencias Python (aquí SÍ tiene espacio para compilar)
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código
COPY . .

# Expone el puerto 8000
EXPOSE 8000

# Comando de inicio
CMD ["sh", "-c", "python download_models.py && uvicorn main:app --host 0.0.0.0 --port 8000"]