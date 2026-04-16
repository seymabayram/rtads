import streamlit as st
import requests
import random
import plotly.graph_objects as go
import plotly.express as px
import time

st.set_page_config(page_title="RTADS Dashboard", layout="wide")
st.title("RTADS - Real Time Anomaly Detection System")

if "history" not in st.session_state:
    st.session_state.history = []
if "anomaly_count" not in st.session_state:
    st.session_state.anomaly_count = 0
if "normal_count" not in st.session_state:
    st.session_state.normal_count = 0

def generate_reading():
    if random.random() < 0.15:
        return {
            "temperature": round(random.uniform(110, 130), 2),
            "speed":       round(random.uniform(180, 220), 2),
            "pressure":    round(random.uniform(70, 90), 2)
        }
    return {
        "temperature": round(random.uniform(60, 80), 2),
        "speed":       round(random.uniform(85, 115), 2),
        "pressure":    round(random.uniform(25, 35), 2)
    }

data = generate_reading()

try:
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    result     = response.json()
    status     = result["status"]
    confidence = result["confidence"]
    votes      = result["votes"]
    models     = result["models"]

    stats_response = requests.get("http://127.0.0.1:8000/stats")
    stats          = stats_response.json()

    if status == "ANOMALY":
        st.session_state.anomaly_count += 1
    else:
        st.session_state.normal_count += 1

    st.session_state.history.append({
        "index":       st.session_state.normal_count + st.session_state.anomaly_count,
        "temperature": data["temperature"],
        "speed":       data["speed"],
        "pressure":    data["pressure"],
        "status":      status,
        "confidence":  confidence,
        "votes":       votes,
        "iso":         models["isolation_forest"]["result"],
        "lof":         models["lof"]["result"],
        "svm":         models["svm"]["result"],
    })

    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]

except Exception as e:
    st.error(f"API connection failed: {e}")
    st.stop()

# Alarm banner
if status == "ANOMALY":
    st.error(f"ANOMALY DETECTED | Confidence: {confidence}% | {votes}/3 models agree")
else:
    st.success(f"System Normal | Confidence: {confidence}%")

# Metrik kartlari - session + database
st.subheader("Live Stats")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Readings", stats["total"])
col2.metric("Normal", stats["normal"])
col3.metric("Anomaly", stats["anomalies"])
col4.metric("Anomaly Rate", f"{stats['anomaly_rate']}%")
col5.metric("Confidence", f"{confidence}%")

# Model oylamasi
st.subheader("Model Votes")
m1, m2, m3 = st.columns(3)

def model_badge(col, name, res):
    if res == "ANOMALY":
        col.error(f"{name}: ANOMALY")
    else:
        col.success(f"{name}: NORMAL")

model_badge(m1, "Isolation Forest", models["isolation_forest"]["result"])
model_badge(m2, "LOF", models["lof"]["result"])
model_badge(m3, "One-Class SVM", models["svm"]["result"])

# Grafikler
col_left, col_right = st.columns([2, 1])

with col_left:
    if st.session_state.history:
        indices   = [h["index"]       for h in st.session_state.history]
        temps     = [h["temperature"] for h in st.session_state.history]
        speeds    = [h["speed"]       for h in st.session_state.history]
        pressures = [h["pressure"]    for h in st.session_state.history]
        confs     = [h["confidence"]  for h in st.session_state.history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indices, y=temps,
                                 name="Temperature", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=indices, y=speeds,
                                 name="Speed", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=indices, y=pressures,
                                 name="Pressure", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=indices, y=confs,
                                 name="Confidence %",
                                 line=dict(color="purple", dash="dot")))

        anomaly_indices = [h["index"] for h in st.session_state.history
                           if h["status"] == "ANOMALY"]
        anomaly_temps   = [h["temperature"] for h in st.session_state.history
                           if h["status"] == "ANOMALY"]

        fig.add_trace(go.Scatter(
            x=anomaly_indices, y=anomaly_temps,
            mode="markers",
            name="Anomaly Point",
            marker=dict(color="red", size=10, symbol="x")
        ))

        fig.update_layout(
            title="Live Sensor Data",
            xaxis_title="Reading",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    pie_fig = px.pie(
        values=[stats["normal"], stats["anomalies"]],
        names=["Normal", "Anomaly"],
        color_discrete_sequence=["#00cc96", "#ef553b"],
        title="Overall Distribution"
    )
    pie_fig.update_layout(height=400)
    st.plotly_chart(pie_fig, use_container_width=True)

# Son okumalar tablosu
st.subheader("Recent Readings")
for h in reversed(st.session_state.history[-10:]):
    icon = "ANOMALY" if h["status"] == "ANOMALY" else "NORMAL"
    st.write(
        f"[{icon}] #{h['index']} | "
        f"Temp: {h['temperature']} | "
        f"Speed: {h['speed']} | "
        f"Pressure: {h['pressure']} | "
        f"Confidence: {h['confidence']}% | "
        f"IF: {h['iso']} | LOF: {h['lof']} | SVM: {h['svm']}"
    )

time.sleep(2)
st.rerun()