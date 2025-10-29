#!/bin/bash

# Hyperliquid Market Microstructure Dashboard Launcher

echo "🚀 Starting Hyperliquid Market Microstructure Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed!"
    echo "📦 Installing requirements..."
    pip3 install -r requirements.txt
    echo ""
fi

echo "✅ Launching dashboard..."
echo "📊 Open your browser to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run dashboard.py

