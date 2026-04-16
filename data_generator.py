import numpy as np

def generate_normal_data(n=2000):
    np.random.seed(42)
    temperature = np.random.normal(loc=70, scale=5, size=n)
    speed       = np.random.normal(loc=100, scale=10, size=n)
    pressure    = np.random.normal(loc=30, scale=3, size=n)
    return np.column_stack([temperature, speed, pressure])

def generate_single_reading(anomaly=False):
    if anomaly:
        temperature = np.random.normal(loc=120, scale=5)
        speed       = np.random.normal(loc=200, scale=10)
        pressure    = np.random.normal(loc=80, scale=3)
    else:
        temperature = np.random.normal(loc=70, scale=5)
        speed       = np.random.normal(loc=100, scale=10)
        pressure    = np.random.normal(loc=30, scale=3)
    return np.array([[temperature, speed, pressure]])