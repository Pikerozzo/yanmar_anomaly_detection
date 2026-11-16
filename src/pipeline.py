import os
import joblib
import numpy as np
import pandas as pd
from models.FeatEngOCSVM import FeatEngOCSVM
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split

os.makedirs("models/configs", exist_ok=True)


class AnomalyPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.thresholds = None

    def __compute_threshold(self, scores):
        return np.mean(scores) + 3*np.std(scores)

    def train(self, data):
        try:
            if data is None:
                raise ValueError(f"Dataset non fornito")

            channel_models = {}
            channel_scaler = {}
            channel_thresholds = {}
            for c in range(len(data)):
                ch_feats = pd.DataFrame(data[f'ch{c}'])
                ch_train, ch_valid = train_test_split(ch_feats, random_state=42, train_size=0.8)

                # scale features for this specific channel
                scaler = RobustScaler()
                scaler.fit(ch_train)
                scaled_train_features = scaler.transform(ch_train)
                scaled_valid_features = scaler.transform(ch_valid)

                model = FeatEngOCSVM()
                model.fit(scaled_train_features)

                valid_scores = -model.decision_function(scaled_valid_features)
                channel_thresholds[c] = self.__compute_threshold(valid_scores)

                channel_models[c] = model
                channel_scaler[c] = scaler

            self.model = channel_models
            self.scaler = channel_scaler
            self.thresholds = channel_thresholds
            
            joblib.dump(self.model, "models/configs/ocsvm_models.joblib")
            joblib.dump(self.scaler, "models/configs/ocsvm_scalers.joblib")
            joblib.dump(self.thresholds, "models/configs/thresholds.joblib")
        except:
            return False
        return True



    def predict(self, data: dict[str,pd.DataFrame]):
        if not self.model or not self.scaler:
            print("LOADING MODELS...")
            try:
                self.model = joblib.load("models/configs/ocsvm_models.joblib")
                self.scaler = joblib.load("models/configs/ocsvm_scalers.joblib")
                self.thresholds = joblib.load("models/configs/thresholds.joblib")
            except Exception as e:
                raise RuntimeError("ERRORE: non è stato trovato nessun modello addestrato.", e)
        try:
            anom_scores = []
            channel_preds = []
            for ch in range(len(data)):
                ch_feats = pd.DataFrame(data[f'ch{ch}'])
                scaled_features = self.scaler[ch].transform(ch_feats)
                ch_anom_scores = -self.model[ch].decision_function(scaled_features)
                
                anom_scores.append(ch_anom_scores)
                channel_preds.append(ch_anom_scores > self.thresholds[ch])
                
            anom_scores = np.stack(anom_scores)
            channel_preds = np.stack(channel_preds)
            
            return {
                "anomaly_score": anom_scores.T.tolist(),
                "predictions": channel_preds.T.tolist()
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            return 