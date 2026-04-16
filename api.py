from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from data_generator import generate_normal_data
from database import save_reading, get_stats, get_recent

app = FastAPI(title="RTADS - Real Time Anomaly Detection System")

iso_forest = joblib.load("model.pkl")

data = generate_normal_data(n=2000)

lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(data)

svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
svm.fit(data)

print("Models ready: Isolation Forest, LOF, One-Class SVM")
print("Database connected: rtads.db")

class SensorData(BaseModel):
    temperature: float
    speed: float
    pressure: float

@app.get("/")
def root():
    return {"message": "RTADS API is running"}

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

    iso_result = iso_forest.predict(reading)[0]
    lof_result = lof.predict(reading)[0]
    svm_result = svm.predict(reading)[0]

    iso_score = round(float(iso_forest.decision_function(reading)[0]), 4)
    lof_score = round(float(lof.decision_function(reading)[0]), 4)
    svm_score = round(float(svm.decision_function(reading)[0]), 4)

    votes = [iso_result, lof_result, svm_result].count(-1)

    if votes >= 2:
        status = "ANOMALY"
        confidence = round(votes / 3 * 100)
    else:
        status = "NORMAL"
        confidence = round((3 - votes) / 3 * 100)

    iso_str = "ANOMALY" if iso_result == -1 else "NORMAL"
    lof_str = "ANOMALY" if lof_result == -1 else "NORMAL"
    svm_str = "ANOMALY" if svm_result == -1 else "NORMAL"

    save_reading({
        "temperature": data.temperature,
        "speed":       data.speed,
        "pressure":    data.pressure,
        "status":      status,
        "confidence":  confidence,
        "votes":       votes,
        "iso":         iso_str,
        "lof":         lof_str,
        "svm":         svm_str
    })

    return {
        "status":     status,
        "confidence": confidence,
        "votes":      votes,
        "models": {
            "isolation_forest": {"result": iso_str, "score": iso_score},
            "lof":              {"result": lof_str, "score": lof_score},
            "svm":              {"result": svm_str, "score": svm_score}
        },
        "input": {
            "temperature": data.temperature,
            "speed":       data.speed,
            "pressure":    data.pressure
        }
    }