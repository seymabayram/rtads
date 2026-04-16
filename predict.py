import joblib
from data_generator import generate_single_reading

model = joblib.load("model.pkl")

for i in range(10):
    anomaly = (i >= 8)
    reading = generate_single_reading(anomaly=anomaly)
    result  = model.predict(reading)

    if result[0] == -1:
        print(f"Okuma {i+1}: ⚠️  ANOMALY  → {reading[0]}")
    else:
        print(f"Okuma {i+1}: ✅ NORMAL  → {reading[0]}")