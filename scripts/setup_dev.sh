#!/bin/bash

# RTADS Development Environment Setup Script 🛡️
# This script automates the installation of dependencies and initialization of models.

echo "🚀 Initializing RTADS development environment..."

# 1. Check for Python
if ! command -v python3 &> /dev/null
then
    echo "❌ Error: Python3 not found. Please install Python 3.9+."
    exit 1
fi

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# 3. Install Dependencies
echo "🛠️ Installing dependencies from requirements.txt..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Initialize Database and Models
if [ -f "train.py" ]; then
    echo "🧠 Training initial ensemble models..."
    python3 train.py
else
    echo "⚠️ Warning: train.py not found. Skipping model initialization."
fi

echo "✅ Environment setup complete! You are ready to run RTADS."
echo "💡 To run the API: uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo "💡 To run the Dashboard: streamlit run src.dashboard/app.py"
