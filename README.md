# 🚀 AI Crypto Price Prediction System

An advanced cryptocurrency price prediction system using machine learning and real-time data analytics. This system combines multiple AI models, real-time data fetching, and comprehensive technical analysis to provide accurate price predictions and trading recommendations.

## 🌟 Features

### 🤖 Advanced AI/ML Models
- **LSTM Neural Networks** - Deep learning for sequential pattern recognition
- **XGBoost** - Gradient boosting for robust predictions
- **LightGBM** - Fast and efficient gradient boosting
- **Random Forest** - Ensemble learning for stability
- **Ensemble Predictions** - Combined model outputs for better accuracy

### 📊 Real-Time Data Integration
- **Multiple API Sources** - CoinGecko, Yahoo Finance, CCXT exchanges
- **Live Price Updates** - Real-time cryptocurrency prices
- **Market Sentiment Analysis** - News and social media sentiment scoring
- **Technical Indicators** - 20+ advanced technical analysis indicators

### 📈 Advanced Analytics
- **Feature Engineering** - 50+ engineered features including:
  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Volume patterns and ratios
  - Price patterns and positions
  - Lag features and rolling statistics
- **Confidence Intervals** - Prediction reliability scoring
- **Risk Assessment** - Volatility-based risk evaluation
- **Trading Recommendations** - AI-powered buy/sell/hold signals

### 🎯 Interactive Dashboard
- **Real-time Price Display** - Current prices with 24h changes
- **Prediction Visualization** - Interactive charts with multiple model predictions
- **Model Performance Metrics** - R², MAE, MSE for each model
- **Confidence Scoring** - Visual confidence indicators
- **Alert System** - Price alert notifications

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd crypto-price-predictor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.28+
- XGBoost 1.7+
- LightGBM 3.3+
- And more (see requirements.txt)

## 🚀 Usage

1. **Select Cryptocurrency** - Choose from 10 popular cryptocurrencies
2. **Configure Parameters** - Set prediction period, time horizon, and models
3. **Generate Predictions** - Click to train models and generate forecasts
4. **Analyze Results** - Review predictions, confidence scores, and recommendations
5. **Set Alerts** - Configure price alerts for monitoring

## 📊 Supported Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Cardano (ADA)
- Solana (SOL)
- Dogecoin (DOGE)
- Polygon (MATIC)
- Avalanche (AVAX)
- Chainlink (LINK)
- Polkadot (DOT)

## 🧠 Model Architecture

### LSTM Neural Network
- 3-layer LSTM with dropout regularization
- 100-100-50 neuron architecture
- Adam optimizer with learning rate scheduling
- Early stopping and learning rate reduction

### Ensemble Approach
- Multiple model predictions combined
- Weighted averaging based on performance
- Confidence scoring for reliability assessment

## 📈 Technical Indicators

The system uses 20+ technical indicators including:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillators
- Williams %R
- Commodity Channel Index (CCI)
- Average Directional Index (ADX)
- On-Balance Volume (OBV)
- Average True Range (ATR)

## ⚠️ Risk Disclaimer

**Important:** This tool is for educational and research purposes only. Cryptocurrency investments carry high risk:

- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- Consider consulting with a financial advisor
- Never invest more than you can afford to lose
- Cryptocurrency markets are highly volatile and unpredictable

## 🔧 Configuration

The system allows customization of:
- Prediction time horizon (1-30 days)
- Historical data period (6 months to 5 years)
- Model selection (enable/disable specific models)
- Alert thresholds and notifications

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

## 📄 License

This project is for educational purposes. Please ensure compliance with relevant financial regulations in your jurisdiction.
