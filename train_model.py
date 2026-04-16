from sklearn.ensemble import IsolationForest
import joblib
from data_generator import generate_normal_data

data = generate_normal_data(n=2000)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model.fit(data)

joblib.dump(model, "model.pkl")
print(" Model training sucesfully completed and saved as model.pkl")