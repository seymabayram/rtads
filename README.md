# RTADS - Real Time Anomaly Detection System

A real-time anomaly detection system that simulates sensor data and uses machine learning to detect abnormal behavior.

## Features

- 3 AI models running simultaneously (Isolation Forest, LOF, One-Class SVM)
- Voting system with confidence score
- SQLite database for storing all readings
- Live dashboard with real-time charts
- REST API built with FastAPI

## Tech Stack

- Python 3.9
- scikit-learn
- FastAPI
- Streamlit
- Plotly
- SQLite

## Project Structure

rtads/
├── data_generator.py   # Sensor data simulation
├── train_model.py      # Model training
├── predict.py          # CLI prediction
├── api.py              # FastAPI REST API
├── database.py         # SQLite integration
├── dashboard.py        # Streamlit dashboard
├── model.pkl           # Trained model
└── requirements.txt

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 train_model.py
```

## Run

Terminal 1 - API:
```bash
uvicorn api:app --reload
```

Terminal 2 - Dashboard:
```bash
streamlit run dashboard.py
```

Open http://localhost:8501 in your browser.

## API Endpoints

- `GET /` - API status
- `POST /predict` - Predict normal/anomaly
- `GET /stats` - Overall statistics
- `GET /history` - Recent readings