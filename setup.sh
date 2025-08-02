#!/bin/bash

# 🚀 AI Crypto Price Prediction System Setup Script
# This script helps set up the environment and install dependencies

echo "🚀 AI Crypto Price Prediction System Setup"
echo "==========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✓ pip3 found"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "crypto_env" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf crypto_env
fi

python3 -m venv crypto_env
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment."
    echo "On Ubuntu/Debian, try: sudo apt install python3-venv"
    echo "On macOS, try: brew install python3"
    exit 1
fi

echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source crypto_env/bin/activate

# Upgrade pip
echo ""
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
echo "This may take several minutes..."

# Install in stages to handle potential conflicts
echo "Installing core packages..."
pip install streamlit pandas numpy requests

echo "Installing data analysis packages..."
pip install yfinance plotly ta scikit-learn seaborn matplotlib scipy statsmodels

echo "Installing ML packages..."
pip install xgboost lightgbm joblib

echo "Installing deep learning packages..."
pip install tensorflow keras

echo "Installing crypto and sentiment analysis packages..."
pip install ccxt pycoingecko python-binance prophet textblob vaderSentiment

echo "Installing utility packages..."
pip install schedule python-telegram-bot

# Verify installation
echo ""
echo "🧪 Testing installation..."
python3 -c "
import streamlit
import pandas
import numpy
import tensorflow
import xgboost
import lightgbm
print('✓ All core packages imported successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "To run the application:"
    echo "1. Activate the virtual environment: source crypto_env/bin/activate"
    echo "2. Run the app: streamlit run app.py"
    echo ""
    echo "📚 Features included:"
    echo "  • Real-time crypto price fetching"
    echo "  • LSTM, XGBoost, LightGBM, Random Forest models"
    echo "  • Technical analysis with 20+ indicators"
    echo "  • Ensemble predictions and confidence scoring"
    echo "  • Interactive dashboard with Streamlit"
    echo ""
    echo "⚠️  Disclaimer: This tool is for educational purposes only."
    echo "   Always do your own research before making investment decisions."
else
    echo ""
    echo "❌ Some packages failed to install. Please check the error messages above."
    echo "You may need to install additional system dependencies."
fi