# api_scoring.py - version avec print des features
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
from typing import List

# ===== Réglages MLflow =====
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "Projet_Scoring_Best_Model"
MODEL_STAGE = None  # None => on prend latest

app = FastAPI(
    title="API de Scoring (MLflow Registry)",
    description="API qui interroge le modèle depuis MLflow Model Registry",
    version="1.0.0",
)

class InputData(BaseModel):
    data: dict  # {feature: valeur}

def _get_features_from_signature(model_uri: str):
    import mlflow.models
    info = mlflow.models.get_model_info(model_uri)
    if info is not None and info.signature is not None:
        return [col.name for col in info.signature.inputs.inputs]
    return None

def _save_swagger_example(features, path="input_example_swagger.json"):
    example = {"data": {feat: 0 for feat in features}}
    with open(path, "w") as f:
        json.dump(example, f, indent=2)

def _load_model_and_features():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MODEL_STAGE:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    else:
        model_uri = f"models:/{MODEL_NAME}/latest"

    model = mlflow.sklearn.load_model(model_uri)
    feats = _get_features_from_signature(model_uri)

    if feats is None and hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)

    if not feats:
        raise RuntimeError("Impossible de déterminer la liste des features du modèle")

    # === AJOUT : imprimer les features au format JSON ===
    example = {"data": {feat: 0 for feat in feats}}
    print("\n=== Liste des features détectées ===")
    print(json.dumps(example, indent=2))

    _save_swagger_example(feats)
    return model, feats

MODEL, FEATURES = _load_model_and_features()

@app.get("/")
def home():
    return {"message": "API scoring – modèle appelé depuis MLflow Registry (FastAPI)"}

@app.get("/features")
def get_features():
    return FEATURES

@app.post("/predict")
def predict(input: InputData):
    input_data = input.data
    # Vérifie qu'on a toutes les features attendues
    missing = [f for f in FEATURES if f not in input_data]
    if missing:
        return {
            "error": "Features manquantes",
            "missing": missing,
            "expected": FEATURES,
        }

    # Ordonne et prépare l'array
    X = np.array([input_data[f] for f in FEATURES])
    proba = float(MODEL.predict_proba([X])[0][1])  # Classe positive = colonne 1
    pred = int(MODEL.predict([X])[0])

    return {
        "prediction": pred,
        "probability_bad_payer": proba,
    }
