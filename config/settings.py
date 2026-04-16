import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database Configuration
DATABASE_PATH = os.path.join(BASE_DIR, "rtads.db")

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
ISO_FOREST_PATH = os.path.join(MODELS_DIR, "iso_forest.pkl")
LOF_PATH = os.path.join(MODELS_DIR, "lof.pkl")
OCSVM_PATH = os.path.join(MODELS_DIR, "ocsvm.pkl")

# Sensor Simulation Parameters
# (loc = Mean, scale = Standard Deviation)
SENSOR_CONFIG = {
    "temperature": {"loc": 70.0, "scale": 5.0, "anomaly_loc": 120.0},
    "speed":       {"loc": 100.0, "scale": 10.0, "anomaly_loc": 200.0},
    "pressure":    {"loc": 30.0, "scale": 3.0, "anomaly_loc": 80.0}
}

# AI Model Hyperparameters
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.05,
    "random_state": 42
}

LOF_PARAMS = {
    "n_neighbors": 20,
    "novelty": True,
    "contamination": 0.05
}

OCSVM_PARAMS = {
    "kernel": "rbf",
    "gamma": "auto",
    "nu": 0.05
}
