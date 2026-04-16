import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.data_generator import generate_normal_data, generate_single_reading
from config.settings import (
    ISO_FOREST_PATH, LOF_PATH, OCSVM_PATH,
    ISOLATION_FOREST_PARAMS, LOF_PARAMS, OCSVM_PARAMS,
    MODELS_DIR
)

def train_models():
    print("🚀 Starting Unified Training Process...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # 1. Generate Training Data (Normal Only for Unsupervised Learning)
    print("📊 Generating training data (n=2000)...")
    data = generate_normal_data(n=2000)

    # 2. Train Isolation Forest
    print("🌲 Training Isolation Forest...")
    iso_forest = IsolationForest(**ISOLATION_FOREST_PARAMS)
    iso_forest.fit(data)
    joblib.dump(iso_forest, ISO_FOREST_PATH)
    print(f"✅ Saved to {ISO_FOREST_PATH}")

    # 3. Train Local Outlier Factor
    print("📍 Training Local Outlier Factor (Novelty Mode)...")
    lof = LocalOutlierFactor(**LOF_PARAMS)
    lof.fit(data)
    joblib.dump(lof, LOF_PATH)
    print(f"✅ Saved to {LOF_PATH}")

    # 4. Train One-Class SVM
    print("🛡️ Training One-Class SVM...")
    svm = OneClassSVM(**OCSVM_PARAMS)
    svm.fit(data)
    joblib.dump(svm, OCSVM_PATH)
    print(f"✅ Saved to {OCSVM_PATH}")

    # 5. Simple Evaluation
    print("\n📈 Evaluating Models (Basic Test)...")
    test_normal = generate_single_reading(anomaly=False)
    test_anomaly = generate_single_reading(anomaly=True)

    for name, model in [("Isolation Forest", iso_forest), ("LOF", lof), ("One-Class SVM", svm)]:
        n_res = "Normal" if model.predict(test_normal)[0] == 1 else "Anomaly"
        a_res = "Anomaly" if model.predict(test_anomaly)[0] == -1 else "Normal"
        print(f" - {name}: Detected Normal as {n_res}, Detected Anomaly as {a_res}")

    print("\n✨ All models trained and saved successfully!")

if __name__ == "__main__":
    train_models()
