import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Union
from src.pipeline import AnomalyPipeline
import pandas as pd
from data.data_utils import extract_features, load_all_files_parallel, load_and_extract_features

app = FastAPI(
    title="Time Series Anomaly Detection API",
    version="1.0.0",
    description="Servizio FastAPI per training e predizione di anomalie."
)

pipeline = AnomalyPipeline()

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <body style="text-align: center">
        <h1>MODEL TRAINING</h1>
        <form action="/train" method="POST" enctype="multipart/form-data">
            <input type="file" multiple name="files"/>
            <input type="submit" value="Send"/>
        </form>

        <br/>
        <hr/>
        <br/>

        <h1>MODEL PREDICTION</h1>
        <form action="/predict" method="POST" enctype="multipart/form-data">
            <input type="file" multiple name="files"/>
            <input type="submit" value="Send"/>
        </form>
    </body>
    """

@app.post("/train", status_code=status.HTTP_200_OK)
async def train_model(files: List[UploadFile]):
    """
    Endpoint per il training del modello tramite caricamento di un file CSV.
    """
    try:
        data, data_feats = await load_and_extract_features(files)
        data_shape = data.shape
        success = pipeline.train(data=data_feats)

        return {
            "status": success,
            "message": f"Modello addestrato con successo." if success else "Errore nell'addestramento del modello",
            "data_shape": data_shape
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno durante il training: {str(e)}")


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_anomalies(files: List[UploadFile]):
    try:
        data, data_feats = await load_and_extract_features(files)
        results = pipeline.predict(data_feats)
        return results
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno durante la predizione: {str(e)}")