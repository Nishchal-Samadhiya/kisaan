#!/usr/bin/env bash
set -e

echo "========================================================"
echo "Starting Kisan Mandi Intelligence (SIH26132)"
echo "AI Farmer Market & Price Discovery Platform"
echo "========================================================"

# Install dependencies if needed
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
