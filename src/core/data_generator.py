import numpy as np
import sys
import os

# Add the project root to sys.path to allow importing from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import SENSOR_CONFIG

def generate_normal_data(n=2000):
    """Generates synthetic normal sensor readings based on config."""
    np.random.seed(42)
    
    temperature = np.random.normal(
        loc=SENSOR_CONFIG["temperature"]["loc"], 
        scale=SENSOR_CONFIG["temperature"]["scale"], 
        size=n
    )
    speed = np.random.normal(
        loc=SENSOR_CONFIG["speed"]["loc"], 
        scale=SENSOR_CONFIG["speed"]["scale"], 
        size=n
    )
    pressure = np.random.normal(
        loc=SENSOR_CONFIG["pressure"]["loc"], 
        scale=SENSOR_CONFIG["pressure"]["scale"], 
        size=n
    )
    
    return np.column_stack([temperature, speed, pressure])

def generate_single_reading(anomaly=False):
    """Generates a single sensor reading (Normal or Anomaly)."""
    if anomaly:
        temperature = np.random.normal(loc=SENSOR_CONFIG["temperature"]["anomaly_loc"], scale=5)
        speed       = np.random.normal(loc=SENSOR_CONFIG["speed"]["anomaly_loc"], scale=10)
        pressure    = np.random.normal(loc=SENSOR_CONFIG["pressure"]["anomaly_loc"], scale=3)
    else:
        temperature = np.random.normal(loc=SENSOR_CONFIG["temperature"]["loc"], scale=5)
        speed       = np.random.normal(loc=SENSOR_CONFIG["speed"]["loc"], scale=10)
        pressure    = np.random.normal(loc=SENSOR_CONFIG["pressure"]["loc"], scale=3)
        
    return np.array([[temperature, speed, pressure]])
