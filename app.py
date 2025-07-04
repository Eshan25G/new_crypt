
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Plotly for interactive visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set page config
st.set_page_config(
    page_title="NIFTY 50 Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# NIFTY 50 Stock Symbols
NIFTY_50_STOCKS = {
    'Reliance Industries': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'State Bank of India': 'SBIN.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'ITC': 'ITC.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Larsen & Toubro': 'LT.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS',
    'Wipro': 'WIPRO.NS',
    'Nestle India': 'NESTLEIND.NS',
    'UltraTech Cement': 'ULTRACEMCO.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Titan Company': 'TITAN.NS',
    'HCL Technologies': 'HCLTECH.NS',
    'Tech Mahindra': 'TECHM.NS',
    'Oil & Natural Gas Corp': 'ONGC.NS',
    'NTPC': 'NTPC.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'Power Grid Corporation': 'POWERGRID.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Tata Steel': 'TATASTEEL.NS',
    'Adani Ports': 'ADANIPORTS.NS',
    'Coal India': 'COALINDIA.NS'
}

class StockPredictor:
    def __init__(self, symbol, period='2y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.scaler = MinMaxScaler()
        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            return True
        except Exception as e:
            st.error(f"Error fetching data for {self.symbol}: {str(e)}")
            return False

    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            return

        # Simple Moving Averages
        self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26).mean()

        # MACD
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()

        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * 2)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * 2)

        # Volume indicators
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=10).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']

        # Price change indicators
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['Price_Change_5'] = self.data['Close'].pct_change(5)
        self.data['Volatility'] = self.data['Price_Change'].rolling(window=10).std()

    def prepare_data_for_ml(self, sequence_length=60, test_size=0.2):
        """Prepare data for machine learning models"""
        if self.data is None:
            return None, None, None, None

        # Select features
        feature_columns = ['Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'RSI',
                          'MACD', 'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility']

        # Remove NaN values
        df = self.data[feature_columns].dropna()

        # Scale the data
        scaled_data = self.scaler.fit_transform(df)

        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Close price is the target

        X, y = np.array(X), np.array(y)

        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def build_gru_model(self, input_shape):
        """Build GRU model"""
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        # 1. LSTM Model
        status_text.text("Training LSTM model...")
        progress_bar.progress(10)
        lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        lstm_history = lstm_model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        self.models['LSTM'] = lstm_model
        progress_bar.progress(40)

        # 2. GRU Model
        status_text.text("Training GRU model...")
        gru_model = self.build_gru_model((X_train.shape[1], X_train.shape[2]))
        gru_history = gru_model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        self.models['GRU'] = gru_model
        progress_bar.progress(60)

        # 3. Traditional ML Models
        X_train_2d = X_train[:, -1, :]
        X_test_2d = X_test[:, -1, :]

        # Random Forest
        status_text.text("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X_train_2d, y_train)
        self.models['Random_Forest'] = rf_model
        progress_bar.progress(80)

        # Linear Regression
        status_text.text("Training Linear Regression model...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_2d, y_train)
        self.models['Linear_Regression'] = lr_model
        progress_bar.progress(100)

        status_text.text("All models trained successfully!")

        return lstm_history, gru_history

    def make_predictions(self, X_test):
        """Make predictions with all models"""
        # Deep learning models
        for model_name in ['LSTM', 'GRU']:
            if model_name in self.models:
                pred = self.models[model_name].predict(X_test, verbose=0)
                self.predictions[model_name] = pred.flatten()

        # Traditional ML models
        X_test_2d = X_test[:, -1, :]
        for model_name in ['Random_Forest', 'Linear_Regression']:
            if model_name in self.models:
                pred = self.models[model_name].predict(X_test_2d)
                self.predictions[model_name] = pred

    def calculate_metrics(self, y_true):
        """Calculate performance metrics"""
        for model_name, predictions in self.predictions.items():
            # Inverse transform predictions and actual values
            y_true_scaled = y_true.reshape(-1, 1)
            pred_scaled = predictions.reshape(-1, 1)

            # Create dummy array for inverse transform
            dummy_array = np.zeros((len(y_true_scaled), self.scaler.n_features_in_))
            dummy_array[:, 0] = y_true_scaled.flatten()
            y_true_actual = self.scaler.inverse_transform(dummy_array)[:, 0]

            dummy_array[:, 0] = pred_scaled.flatten()
            pred_actual = self.scaler.inverse_transform(dummy_array)[:, 0]

            # Calculate metrics
            mse = mean_squared_error(y_true_actual, pred_actual)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_actual, pred_actual)
            r2 = r2_score(y_true_actual, pred_actual)

            # Calculate directional accuracy
            actual_direction = np.sign(np.diff(y_true_actual))
            pred_direction = np.sign(np.diff(pred_actual))
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100

            self.metrics[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Directional_Accuracy': directional_accuracy
            }

    def plot_predictions(self):
        """Plot predictions vs actual values"""
        if not self.predictions:
            return

        # Get actual values for plotting
        actual_data = self.data['Close'].values[-len(list(self.predictions.values())[0]):]
        dates = self.data.index[-len(actual_data):]

        # Create comparison plot
        fig = go.Figure()

        # Add actual values
        fig.add_trace(
            go.Scatter(x=dates, y=actual_data, name='Actual Price',
                      line=dict(color='blue', width=2))
        )

        # Add predictions
        colors = ['red', 'green', 'orange', 'purple']
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            pred_scaled = predictions.reshape(-1, 1)
            dummy_array = np.zeros((len(pred_scaled), self.scaler.n_features_in_))
            dummy_array[:, 0] = pred_scaled.flatten()
            pred_actual = self.scaler.inverse_transform(dummy_array)[:, 0]

            fig.add_trace(
                go.Scatter(x=dates, y=pred_actual, name=f'{model_name}',
                          line=dict(color=colors[i % len(colors)], dash='dash'))
            )

        fig.update_layout(
            title=f'Stock Price Predictions for {self.symbol}',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            hovermode='x unified',
            height=600
        )

        return fig

    def predict_future(self, days=30):
        """Predict future stock prices"""
        if not self.models or self.data is None:
            return None, None

        # Get last sequence
        feature_columns = ['Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'RSI',
                          'MACD', 'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility']

        last_sequence = self.data[feature_columns].dropna().iloc[-60:].values
        last_sequence_scaled = self.scaler.transform(last_sequence)

        future_predictions = {}

        # Predict with LSTM
        if 'LSTM' in self.models:
            lstm_predictions = []
            current_sequence = last_sequence_scaled.copy()

            for _ in range(days):
                next_pred = self.models['LSTM'].predict(current_sequence.reshape(1, 60, -1), verbose=0)
                lstm_predictions.append(next_pred[0, 0])

                new_row = current_sequence[-1].copy()
                new_row[0] = next_pred[0, 0]
                current_sequence = np.vstack([current_sequence[1:], new_row])

            # Convert back to original scale
            dummy_array = np.zeros((len(lstm_predictions), self.scaler.n_features_in_))
            dummy_array[:, 0] = lstm_predictions
            lstm_predictions_actual = self.scaler.inverse_transform(dummy_array)[:, 0]
            future_predictions['LSTM'] = lstm_predictions_actual

        # Create future dates
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')

        return future_predictions, future_dates

@st.cache_data
def load_stock_data(symbol, period):
    """Cached function to load stock data"""
    ticker = yf.Ticker(symbol)
    return ticker.history(period=period)

def main():
    st.markdown('<div class="main-header">📈 NIFTY 50 Stock Predictor</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Stock selection
    stock_name = st.sidebar.selectbox(
        "Select Stock",
        list(NIFTY_50_STOCKS.keys()),
        index=0
    )
    
    stock_symbol = NIFTY_50_STOCKS[stock_name]
    
    # Period selection
    period = st.sidebar.selectbox(
        "Select Period",
        ["1y", "2y", "3y", "5y"],
        index=1
    )
    
    # Prediction days
    prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=7,
        max_value=60,
        value=30,
        step=1
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Analysis for {stock_name} ({stock_symbol})")
        
        # Display basic stock info
        try:
            stock_data = load_stock_data(stock_symbol, period)
            current_price = stock_data['Close'].iloc[-1]
            prev_price = stock_data['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Current Price", f"₹{current_price:.2f}", f"{change:.2f}")
            
            with metric_col2:
                st.metric("Change %", f"{change_pct:.2f}%")
            
            with metric_col3:
                st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")
            
            # Plot historical data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title=f'{stock_name} - Historical Price',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading stock data: {str(e)}")
    
    with col2:
        st.subheader("Quick Stats")
        
        if 'stock_data' in locals():
            # Calculate some quick statistics
            high_52w = stock_data['High'].max()
            low_52w = stock_data['Low'].min()
            avg_volume = stock_data['Volume'].mean()
            volatility = stock_data['Close'].pct_change().std() * 100
            
            st.write(f"**52W High:** ₹{high_52w:.2f}")
            st.write(f"**52W Low:** ₹{low_52w:.2f}")
            st.write(f"**Avg Volume:** {avg_volume:,.0f}")
            st.write(f"**Volatility:** {volatility:.2f}%")
    
    # Prediction section
    st.subheader("🔮 Stock Price Prediction")
    
    if st.button("Start Prediction Analysis", type="primary"):
        with st.spinner("Analyzing stock data and training models..."):
            try:
                # Initialize predictor
                predictor = StockPredictor(stock_symbol, period)
                
                # Fetch data
                if predictor.fetch_data():
                    st.success(f"Successfully fetched {len(predictor.data)} days of data")
                    
                    # Calculate technical indicators
                    predictor.calculate_technical_indicators()
                    
                    # Prepare data
                    X_train, X_test, y_train, y_test = predictor.prepare_data_for_ml()
                    
                    if X_train is not None:
                        st.info(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
                        
                        # Train models
                        predictor.train_models(X_train, X_test, y_train, y_test)
                        
                        # Make predictions
                        predictor.make_predictions(X_test)
                        
                        # Calculate metrics
                        predictor.calculate_metrics(y_test)
                        
                        # Display results
                        st.subheader("📊 Model Performance")
                        
                        # Metrics table
                        if predictor.metrics:
                            metrics_df = pd.DataFrame(predictor.metrics).T
                            st.dataframe(metrics_df.round(4))
                            
                            # Find best model
                            best_model = min(predictor.metrics.items(), key=lambda x: x[1]['RMSE'])
                            st.success(f"Best performing model: **{best_model[0]}** (RMSE: {best_model[1]['RMSE']:.4f})")
                        
                        # Plot predictions
                        st.subheader("📈 Prediction vs Actual")
                        fig = predictor.plot_predictions()
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Future predictions
                        st.subheader("🔮 Future Predictions")
                        future_pred, future_dates = predictor.predict_future(prediction_days)
                        
                        if future_pred:
                            # Plot future predictions
                            fig = go.Figure()
                            
                            # Historical data (last 60 days)
                            historical_data = predictor.data['Close'].iloc[-60:]
                            fig.add_trace(
                                go.Scatter(x=historical_data.index, y=historical_data.values,
                                          name='Historical', line=dict(color='blue'))
                            )
                            
                            # Future predictions
                            for model_name, predictions in future_pred.items():
                                fig.add_trace(
                                    go.Scatter(x=future_dates, y=predictions,
                                              name=f'{model_name} Future', line=dict(dash='dash'))
                                )
                            
                            fig.update_layout(
                                title=f'Future Stock Price Predictions for {stock_name}',
                                xaxis_title='Date',
                                yaxis_title='Price (₹)',
                                hovermode='x unified',
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display future predictions table
                            future_df = pd.DataFrame({
                                'Date': future_dates,
                                'Predicted_Price': future_pred['LSTM']
                            })
                            
                            st.subheader("📅 Future Price Predictions")
                            st.dataframe(future_df.head(10))
                            
                            # Download predictions
                            csv = future_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name=f"{stock_symbol}_predictions.csv",
                                mime="text/csv"
                            )
                    
                    else:
                        st.error("Error preparing data for machine learning")
                
                else:
                    st.error("Failed to fetch stock data")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666;'>
            <p>🔬 This application uses LSTM, GRU, and traditional ML models for stock prediction.</p>
            <p>⚠️ Predictions are for educational purposes only. Not financial advice.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
