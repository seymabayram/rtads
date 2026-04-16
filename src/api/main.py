from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.database import save_reading, get_stats, get_recent
from config.settings import ISO_FOREST_PATH, LOF_PATH, OCSVM_PATH

app = FastAPI(title="RTADS - Real Time Anomaly Detection System")

# Load Pre-trained Models
print("📦 Loading models...")
iso_forest = joblib.load(ISO_FOREST_PATH)
lof = joblib.load(LOF_PATH)
svm = joblib.load(OCSVM_PATH)
print("✅ Models loaded successfully")

class SensorData(BaseModel):
    temperature: float
    speed: float
    pressure: float

@app.get("/")
def root():
    return {
        "status": "online",
        "project": "RTADS",
        "description": "Real-Time Anomaly Detection System"
    }

@app.get("/stats")
def stats():
    return get_stats()

@app.get("/history")
def history():
    rows = get_recent(limit=20)
    return [
        {
            "timestamp":   r[0],
            "temperature": r[1],
            "speed":       r[2],
            "pressure":    r[3],
            "status":      r[4],
            "confidence":  r[5],
            "votes":       r[6]
        }
        for r in rows
    ]

@app.post("/predict")
def predict(data: SensorData):
    reading = np.array([[data.temperature, data.speed, data.pressure]])

    # Predictions (1 for Normal, -1 for Anomaly)
    iso_res = iso_forest.predict(reading)[0]
    lof_res = lof.predict(reading)[0]
    svm_res = svm.predict(reading)[0]

    # Decision functions (Anomaly Scores)
    iso_score = round(float(iso_forest.decision_function(reading)[0]), 4)
    lof_score = round(float(lof.decision_function(reading)[0]), 4)
    svm_score = round(float(svm.decision_function(reading)[0]), 4)

    # Ensemble Voting
    # Convert results: -1 -> Anomaly (1), 1 -> Normal (0)
    results = [iso_res, lof_res, svm_res]
    anomaly_votes = results.count(-1)

    if anomaly_votes >= 2:
        status = "ANOMALY"
        confidence = round(anomaly_votes / 3 * 100)
    else:
        status = "NORMAL"
        confidence = round((3 - anomaly_votes) / 3 * 100)

    # Save to Database
    save_reading({
        "temperature": data.temperature,
        "speed":       data.speed,
        "pressure":    data.pressure,
        "status":      status,
        "confidence":  confidence,
        "votes":       anomaly_votes,
        "iso":         "ANOMALY" if iso_res == -1 else "NORMAL",
        "lof":         "ANOMALY" if lof_res == -1 else "NORMAL",
        "svm":         "ANOMALY" if svm_res == -1 else "NORMAL"
    })

    return {
        "status":     status,
        "confidence": confidence,
        "votes":      anomaly_votes,
        "individual_models": {
            "isolation_forest": {"result": "ANOMALY" if iso_res == -1 else "NORMAL", "score": iso_score},
            "lof":              {"result": "ANOMALY" if lof_res == -1 else "NORMAL", "score": lof_score},
            "oc_svm":           {"result": "ANOMALY" if svm_res == -1 else "NORMAL", "score": svm_score}
        }
    }
