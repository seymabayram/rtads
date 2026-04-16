import streamlit as st
import requests
import random
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Page Configuration
st.set_page_config(
    page_title="RTADS Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ RTADS - Anomaly Detection System")
st.markdown("---")

# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "reading_count" not in st.session_state:
    st.session_state.reading_count = 0

# Configuration
API_URL = "http://127.0.0.1:8000"

def get_sensor_data():
    """Generates sensor data with occasional anomalies."""
    if random.random() < 0.15:
        return {
            "temperature": round(random.uniform(110, 140), 2),
            "speed":       round(random.uniform(180, 250), 2),
            "pressure":    round(random.uniform(70, 100), 2)
        }
    return {
        "temperature": round(random.uniform(55, 85), 2),
        "speed":       round(random.uniform(80, 120), 2),
        "pressure":    round(random.uniform(20, 40), 2)
    }

# Application Logic
sensor_input = get_sensor_data()

try:
    # 1. API Call for Prediction
    pred_response = requests.post(f"{API_URL}/predict", json=sensor_input)
    result = pred_response.json()
    
    # 2. API Call for Overall Stats
    stats_response = requests.get(f"{API_URL}/stats")
    total_stats = stats_response.json()

    # Process Results
    status     = result["status"]
    confidence = result["confidence"]
    votes      = result["votes"]
    ind_models = result["individual_models"]

    st.session_state.reading_count += 1
    
    # Add to session history
    st.session_state.history.append({
        "id":          st.session_state.reading_count,
        "temperature": sensor_input["temperature"],
        "speed":       sensor_input["speed"],
        "pressure":    sensor_input["pressure"],
        "status":      status,
        "confidence":  confidence,
        "votes":       votes,
        "iso":         ind_models["isolation_forest"]["result"],
        "lof":         ind_models["lof"]["result"],
        "svm":         ind_models["oc_svm"]["result"]
    })

    # Keep only last 50 readings in session
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]

    # Visual feedback based on status
    if status == "ANOMALY":
        st.error(f"🚨 **ANOMALY DETECTED!** Confidence: **{confidence}%** | Voting: **{votes}/3**")
    else:
        st.success(f"✅ **SYSTEM NORMAL** | Confidence: **{confidence}%**")

    # Metrics Section
    st.subheader("📊 System Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Processed", total_stats["total"])
    m2.metric("Anomalies Found", total_stats["anomalies"])
    m3.metric("Normal Readings", total_stats["normal"])
    m4.metric("Anomaly Rate", f"{total_stats['anomaly_rate']}%")

    # Model Insights
    with st.expander("🔍 Model Individual Insights"):
        c1, c2, c3 = st.columns(3)
        def show_model(col, name, data):
            color = "🔴" if data["result"] == "ANOMALY" else "🟢"
            col.write(f"{color} **{name}**")
            col.caption(f"Score: {data['score']}")
        
        show_model(c1, "Isolation Forest", ind_models["isolation_forest"])
        show_model(c2, "LOF", ind_models["lof"])
        show_model(c3, "One-Class SVM", ind_models["oc_svm"])

    # Plots Section
    st.subheader("📈 Real-Time Data Visualization")
    plot_col, dist_col = st.columns([2, 1])

    with plot_col:
        hist = st.session_state.history
        ids = [x["id"] for x in hist]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ids, y=[x["temperature"] for x in hist], name="Temperature", line=dict(color="#FF4B4B")))
        fig.add_trace(go.Scatter(x=ids, y=[x["speed"] for x in hist], name="Speed", line=dict(color="#0068C9")))
        fig.add_trace(go.Scatter(x=ids, y=[x["pressure"] for x in hist], name="Pressure", line=dict(color="#29B09D")))
        
        # Add Markers for Anomalies
        anom_x = [x["id"] for x in hist if x["status"] == "ANOMALY"]
        anom_y = [x["temperature"] for x in hist if x["status"] == "ANOMALY"]
        fig.add_trace(go.Scatter(x=anom_x, y=anom_y, mode="markers", name="Alert", marker=dict(color="red", size=12, symbol="x")))

        fig.update_layout(xaxis_title="Reading ID", yaxis_title="Sensor Values", margin=dict(l=0, r=0, t=30, b=0), height=350)
        st.plotly_chart(fig, use_container_width=True)

    with dist_col:
        pie = px.pie(
            values=[total_stats["normal"], total_stats["anomalies"]],
            names=["Normal", "Anomaly"],
            color=["Normal", "Anomaly"],
            color_discrete_map={"Normal": "#29B09D", "Anomaly": "#FF4B4B"}
        )
        pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350, showlegend=True)
        st.plotly_chart(pie, use_container_width=True)

    # Historical Table
    st.subheader("🕒 Recent Activity")
    st.table(st.session_state.history[-5:][::-1])

except Exception as e:
    st.warning("⚠️ Waiting for API connection... (Please ensure API is running at http://localhost:8000)")
    st.info("Run: `uvicorn src.api.main:app --reload` in your terminal.")
    # st.error(str(e))

time.sleep(2)
st.rerun()
