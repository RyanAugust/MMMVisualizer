#!/bin/bash

# Path to the virtual environment
VENV_PATH="${1:-../bikevenv}"
PYTHON_BIN="$VENV_PATH/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Error: Python not found at $PYTHON_BIN"
    echo "Usage: ./run.sh [path_to_venv]"
    exit 1
fi

echo "🚲 Starting BikeShop MMM Director using venv: $VENV_PATH"

# Function to kill background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $(jobs -p)
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# 1. Start the API in the background
echo "📡 Starting API on port 8000..."
$PYTHON_BIN api/main.py > api.log 2>&1 &

# Wait a moment for the API to initialize
sleep 2

# 2. Start the Streamlit UI in the foreground
echo "🎨 Starting UI on port 8501..."
$PYTHON_BIN -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
