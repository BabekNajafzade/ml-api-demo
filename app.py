import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse


MODEL_PATH = "model.pkl"

app = FastAPI(title="Simple ML API", version="1.0")

model = None
feature_columns = None
label_map = None


class PredictionRequest(BaseModel):
    age: int
    salary: float


@app.on_event("startup")
def load_model():
    global model, feature_columns, label_map

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found")

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    label_map = artifact["label_map"]


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict")
def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([req.model_dump()])
    print('kodd: ', req)
    try:
        X = df[feature_columns]
        prob = model.predict_proba(X)[0][1]
        pred = 1 if prob >= 0.5 else 0

        return {
            "prediction": pred,
            "label": label_map[pred],
            "probability": round(float(prob), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
