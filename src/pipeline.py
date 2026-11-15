import os
import joblib
import numpy as np
import pandas as pd
from models import FeatEngOCSVM
from sklearn.preprocessing import RobustScaler

class AnomalyPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None

    def trained(self):
        return self.model and self.scaler

    def train(self, data: dict[str,pd.DataFrame]):
        if data is None:
            raise ValueError(f"Dataset non fornito")

        channels = len(data)
        channel_models = {}
        channel_scaler = {}
        for c in range(channels):
            ch_feats = data[c]

            # Scale features for this specific channel
            scaler = RobustScaler()
            scaler.fit(ch_feats)
            scaled_features = scaler.transform(ch_feats)

            
            model = FeatEngOCSVM()
            model.fit(scaled_features)

            channel_models[c] = model
            channel_scaler[c] = scaler

        self.model = channel_models
        self.scaler = channel_scaler
        
        joblib.dump(self.model, "ocsvm_models.joblib")
        joblib.dump(self.scaler, "ocsvm_scalers.joblib")



    def predict(self, data: dict[str,pd.DataFrame]):
        if not self.model or not self.scaler:
            try:
                self.model = joblib.load("ocsvm_models.joblib")
                self.scaler = joblib.load("ocsvm_scalers.joblib")
            except:
                raise RuntimeError("ERRORE: non è stato trovato nessun modello addestrato.")

        anom_scores = []
        channel_preds = []
        channels = len(data)
        for ch in channels:
            scaled_features = self.scaler.transform(data[ch])
            anom_scores.append(-self.model[ch].decision_function(scaled_features))
            channel_preds.append(self.model[ch].predict(scaled_features)==-1)

        anom_scores = np.stack(anom_scores)
        channel_preds = np.stack(channel_preds)
        
        results = {
            "anomaly_score": anom_scores,
            "predictions": channel_preds
            }
        return results