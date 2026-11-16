FROM python:3.10-slim

WORKDIR /app

# Installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia i moduli sorgente e l'applicazione
COPY ./src ./src
COPY ./app ./app

# Imposta PYTHONPATH per consentire l'import dei pacchetti interni
ENV PYTHONPATH=/app

# Espone la porta del servizio
EXPOSE 8000

# Avvia FastAPI tramite Uvicorn puntando ad app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]