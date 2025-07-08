import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="CryptoEdge - Advanced Crypto Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .signal-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff5858 0%, #f857a6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .signal-hold {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class CryptoAnalyzer:
    def __init__(self):
        self.crypto_symbols = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Binance Coin': 'BNB-USD',
            'Cardano': 'ADA-USD',
            'Solana': 'SOL-USD',
            'Dogecoin': 'DOGE-USD',
            'Polygon': 'MATIC-USD',
            'Avalanche': 'AVAX-USD',
            'Chainlink': 'LINK-USD',
            'Polkadot': 'DOT-USD'
        }
    
    def get_crypto_data(self, symbol, period='1y'):
        """Fetch cryptocurrency data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        if data is None or data.empty:
            return None
        
        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Middle'] = bb.bollinger_mavg()
        data['BB_Lower'] = bb.bollinger_lband()
        
        # Moving Averages
        data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
        data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()
        
        # Volume indicators
        data['Volume_SMA'] = ta.volume.VolumeSMAIndicator(data['Close'], data['Volume']).volume_sma()
        
        # Volatility
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        
        return data
    
    def generate_signals(self, data):
        """Generate trading signals based on technical indicators"""
        if data is None or data.empty:
            return None, None, None
        
        latest = data.iloc[-1]
        signals = []
        
        # RSI Signal
        if latest['RSI'] < 30:
            signals.append(('RSI', 'BUY', 'Oversold condition'))
        elif latest['RSI'] > 70:
            signals.append(('RSI', 'SELL', 'Overbought condition'))
        
        # MACD Signal
        if latest['MACD'] > latest['MACD_Signal']:
            signals.append(('MACD', 'BUY', 'Bullish crossover'))
        else:
            signals.append(('MACD', 'SELL', 'Bearish crossover'))
        
        # Bollinger Bands Signal
        if latest['Close'] < latest['BB_Lower']:
            signals.append(('BB', 'BUY', 'Price below lower band'))
        elif latest['Close'] > latest['BB_Upper']:
            signals.append(('BB', 'SELL', 'Price above upper band'))
        
        # Moving Average Signal
        if latest['SMA_20'] > latest['SMA_50']:
            signals.append(('MA', 'BUY', 'Short MA above Long MA'))
        else:
            signals.append(('MA', 'SELL', 'Short MA below Long MA'))
        
        # Overall signal
        buy_signals = sum(1 for _, signal, _ in signals if signal == 'BUY')
        sell_signals = sum(1 for _, signal, _ in signals if signal == 'SELL')
        
        if buy_signals > sell_signals:
            overall_signal = 'BUY'
        elif sell_signals > buy_signals:
            overall_signal = 'SELL'
        else:
            overall_signal = 'HOLD'
        
        # Calculate entry, stop loss, and take profit
        entry_price = latest['Close']
        atr = latest['ATR']
        
        if overall_signal == 'BUY':
            stop_loss = entry_price - (2 * atr)
            take_profit = entry_price + (3 * atr)
        elif overall_signal == 'SELL':
            stop_loss = entry_price + (2 * atr)
            take_profit = entry_price - (3 * atr)
        else:
            stop_loss = entry_price - (1.5 * atr)
            take_profit = entry_price + (1.5 * atr)
        
        return signals, overall_signal, {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        }
    
    def create_chart(self, data, symbol):
        """Create interactive chart with indicators"""
        if data is None or data.empty:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price & Indicators', 'RSI', 'MACD'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='red', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='red', width=1)),
            row=1, col=1
        )
        
        # Moving Averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram'),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_title='Date',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def backtest_strategy(self, data, initial_capital=10000):
        """Simple backtesting strategy"""
        if data is None or data.empty:
            return None
        
        data = data.copy()
        data['Position'] = 0
        data['Returns'] = data['Close'].pct_change()
        
        # Simple strategy: Buy when RSI < 30, Sell when RSI > 70
        for i in range(1, len(data)):
            if data.iloc[i]['RSI'] < 30 and data.iloc[i-1]['RSI'] >= 30:
                data.iloc[i, data.columns.get_loc('Position')] = 1
            elif data.iloc[i]['RSI'] > 70 and data.iloc[i-1]['RSI'] <= 70:
                data.iloc[i, data.columns.get_loc('Position')] = -1
            else:
                data.iloc[i, data.columns.get_loc('Position')] = data.iloc[i-1]['Position']
        
        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
        data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Cumulative_Returns']
        
        return data

def main():
    st.markdown('<h1 class="main-header">🚀 CryptoEdge Trading Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Crypto Analysis & Signal Generation for Mumbai Traders")
    
    analyzer = CryptoAnalyzer()
    
    # Sidebar
    st.sidebar.header("Trading Dashboard")
    
    # Crypto selection
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency:",
        list(analyzer.crypto_symbols.keys())
    )
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Select Time Period:",
        ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
    
    # Get data
    symbol = analyzer.crypto_symbols[selected_crypto]
    data = analyzer.get_crypto_data(symbol, period)
    
    if data is not None:
        # Calculate indicators
        data = analyzer.calculate_technical_indicators(data)
        
        # Generate signals
        signals, overall_signal, trade_params = analyzer.generate_signals(data)
        
        # Main dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Current Price")
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            st.metric(
                f"{selected_crypto}",
                f"${current_price:.2f}",
                f"{price_change_pct:.2f}%"
            )
        
        with col2:
            st.markdown("### Trading Signal")
            if overall_signal == 'BUY':
                st.markdown('<div class="signal-buy">🟢 BUY SIGNAL</div>', unsafe_allow_html=True)
            elif overall_signal == 'SELL':
                st.markdown('<div class="signal-sell">🔴 SELL SIGNAL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-hold">🟡 HOLD SIGNAL</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("### Risk/Reward")
            if trade_params:
                st.metric(
                    "R/R Ratio",
                    f"{trade_params['risk_reward']:.2f}",
                    "Good" if trade_params['risk_reward'] > 2 else "Review"
                )
        
        # Trade Parameters
        if trade_params:
            st.markdown("### 📊 Trade Parameters")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Entry Price", f"${trade_params['entry']:.2f}")
            with col2:
                st.metric("Stop Loss", f"${trade_params['stop_loss']:.2f}")
            with col3:
                st.metric("Take Profit", f"${trade_params['take_profit']:.2f}")
            with col4:
                risk_amount = abs(trade_params['entry'] - trade_params['stop_loss'])
                st.metric("Risk Amount", f"${risk_amount:.2f}")
        
        # Technical Indicators
        st.markdown("### 📈 Technical Analysis")
        
        # Create chart
        chart = analyzer.create_chart(data, selected_crypto)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Signals breakdown
        st.markdown("### 🎯 Signal Breakdown")
        if signals:
            signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Description'])
            st.dataframe(signal_df, use_container_width=True)
        
        # Key metrics
        st.markdown("### 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")
        with col2:
            st.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
        with col3:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        with col4:
            st.metric("ATR", f"{data['ATR'].iloc[-1]:.2f}")
        
        # Backtesting
        st.markdown("### 📈 Strategy Backtesting")
        backtest_data = analyzer.backtest_strategy(data)
        
        if backtest_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                final_value = backtest_data['Portfolio_Value'].iloc[-1]
                total_return = ((final_value - 10000) / 10000) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            
            with col2:
                max_drawdown = ((backtest_data['Portfolio_Value'].min() - backtest_data['Portfolio_Value'].max()) / backtest_data['Portfolio_Value'].max()) * 100
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            
            # Portfolio performance chart
            fig_portfolio = go.Figure()
            fig_portfolio.add_trace(go.Scatter(
                x=backtest_data.index,
                y=backtest_data['Portfolio_Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2)
            ))
            
            fig_portfolio.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                height=400
            )
            
            st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Risk Management
        st.markdown("### ⚠️ Risk Management Guidelines")
        st.info("""
        **Mumbai Trader's Risk Management Rules:**
        - Never risk more than 2% of your capital on a single trade
        - Always set stop losses before entering a trade
        - Consider INR conversion rates and timing for Indian markets
        - Keep track of your trades for tax purposes (30% flat tax in India)
        - Use proper position sizing based on your account size
        """)
        
        # Market sentiment
        st.markdown("### 📰 Market Sentiment")
        sentiment_score = np.random.randint(1, 100)  # Placeholder for real sentiment analysis
        
        if sentiment_score > 70:
            st.success(f"Market Sentiment: Bullish ({sentiment_score}/100)")
        elif sentiment_score < 30:
            st.error(f"Market Sentiment: Bearish ({sentiment_score}/100)")
        else:
            st.warning(f"Market Sentiment: Neutral ({sentiment_score}/100)")
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This is for educational purposes only. Always do your own research before trading.")

if __name__ == "__main__":
    main()
