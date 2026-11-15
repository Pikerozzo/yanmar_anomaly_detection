import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Union
from src.pipeline import AnomalyPipeline
import pandas as pd
from data.data_utils import extract_features

app = FastAPI(
    title="Time Series Anomaly Detection API",
    version="1.0.0",
    description="Servizio FastAPI per training e predizione di anomalie."
)

# Inizializza la pipeline (il modello viene cercato in models/model.pkl)
pipeline = AnomalyPipeline(model_path="models/model.pkl")

@app.post("/train", status_code=status.HTTP_200_OK)
async def train_model(file: UploadFile = File(...)):
    """
    Endpoint per il training del modello tramite caricamento di un file CSV.
    """
    try:
        # Assicura che la cartella data esista
        os.makedirs("data", exist_ok=True)
        
        # Salva il file caricato nella cartella data/ del container (e quindi sul volume host)
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        data = pd.read_csv(file_path, sep="\t",names=[f'ch{i}' for i in range(4)])
        data_shape = data.shape
        data = extract_features(data)
        
        channels, seq_len = data.shape
        # Esegue il training usando la pipeline esistente
        pipeline.train(data=data)
        
        return {
            "status": "success",
            "message": f"File '{file.filename}' caricato e modello addestrato con successo.",
            "data_shape": data_shape
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno durante il training: {str(e)}")

# (Lascia invariato l'endpoint /predict)

@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_anomalies(file: UploadFile = File(...)):
    if not pipeline.trained():
        raise RuntimeError("Il modello non è stato ancora addestrato. Eseguire prima una richiesta POST su /train.")

    try:
        data = pd.read_csv(file.filename, sep="\t",names=[f'ch{i}' for i in range(4)])
        data = extract_features(data)
        results = pipeline.predict(data)
        return {
            "status": "success",
            "predictions": results
        }
        return 'predict successo'
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno durante la predizione: {str(e)}")