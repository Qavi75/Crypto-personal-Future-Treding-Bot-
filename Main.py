# ✅ Required Libraries Install (Optimized)
!pip install ccxt==4.4.59 ta==0.11.0 aiohttp==3.10.11 numpy pandas tensorflow keras yfinance matplotlib scikit-learn --quiet

# ✅ Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU
import ccxt
import ta
from sklearn.preprocessing import MinMaxScaler
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
drive.mount('/content/drive')

# ✅ Real-Time Market Data Fetching (With Proxy Support)
def get_crypto_data(symbol="PEPE/USDT", exchange_name="kucoin", timeframe='1h', limit=200):
    try:
        exchange_class = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'options': {'adjustForTimeDifference': True}
        })
        ohlcv = exchange_class.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.dropna()
    except Exception as e:
        print(f"❌ Error fetching data from {exchange_name.upper()}: {str(e)[:100]}...")
        return pd.DataFrame()

# ✅ Fetch Data from Multiple Exchanges (Fallback Mechanism)
exchanges = ["kucoin", "okx", "gateio"]
dfs = []

for exchange in exchanges:
    data = get_crypto_data(exchange_name=exchange)
    if not data.empty:
        dfs.append(data)
        print(f"✅ Successfully fetched data from {exchange.upper()}")
        break  # Stop after first successful fetch

if not dfs:
    raise ValueError("🔥 All exchanges failed! Try VPN or different exchanges.")

okx_data = dfs[0]

# ✅ Enhanced Indicators Calculation with Dynamic Adjustments
def apply_indicators_with_dynamic_adjustments(df):
    # Trend Indicators
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['SMA_20'] = df['close'].rolling(20).mean()

    # Momentum Indicators
    df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14, fillna=True).rsi()
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, fillna=True)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Volatility Indicators
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True
    ).average_true_range()

    # Volume Indicators
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['close'], volume=df['volume'], fillna=True
    ).on_balance_volume()

    # Dynamic Thresholds
    df['Dynamic_RSI_Lower'] = 30 + (df['ATR'] / df['close'].rolling(20).mean()) * 100
    df['Dynamic_RSI_Upper'] = 70 - (df['ATR'] / df['close'].rolling(20).mean()) * 100

    return df.dropna()

okx_data = apply_indicators_with_dynamic_adjustments(okx_data)

# ✅ Advanced LSTM Model Architecture
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(okx_data[['close']])

sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential([
    Input(shape=(X.shape[1], 1)),
    Bidirectional(LSTM(100, return_sequences=True)),
    Dropout(0.3),
    GRU(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ✅ Enhanced Model Training
history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ✅ Professional Price Prediction
last_sequence = X[-1].reshape(1, X.shape[1], 1)
predicted_scaled = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
current_price = okx_data['close'].iloc[-1]

# ✅ Dynamic Signal System
signal_rules = {
    'BUY': [
        okx_data['RSI_14'].iloc[-1] < okx_data['Dynamic_RSI_Lower'].iloc[-1],
        okx_data['MACD'].iloc[-1] > okx_data['MACD_Signal'].iloc[-1],
        current_price > okx_data['SMA_20'].iloc[-1]
    ],
    'SELL': [
        okx_data['RSI_14'].iloc[-1] > okx_data['Dynamic_RSI_Upper'].iloc[-1],
        okx_data['MACD'].iloc[-1] < okx_data['MACD_Signal'].iloc[-1],
        current_price < okx_data['EMA_9'].iloc[-1]
    ]
}

signal = "HOLD"
if sum(signal_rules['BUY']) >= 2:
    signal = "BUY"
elif sum(signal_rules['SELL']) >= 2:
    signal = "SELL"

# ✅ Dynamic Adjustments for Risk Management
def calculate_dynamic_risk_with_adjustments(current, atr, signal, close_prices):
    trend_strength = (close_prices.ewm(span=9).mean() - close_prices.rolling(20).mean()).iloc[-1]
    risk_multiplier = 1 + abs(trend_strength / close_prices.std())

    params = {
        'BUY': {'stop': -2.0 * risk_multiplier, 'target': 3.0 * risk_multiplier},
        'SELL': {'stop': 2.0 * risk_multiplier, 'target': -3.0 * risk_multiplier}
    }
    stop_loss = current + (atr * params[signal]['stop'])
    take_profit = current + (atr * params[signal]['target'])
    return stop_loss, take_profit

stop_loss, take_profit = calculate_dynamic_risk_with_adjustments(
    current_price, okx_data['ATR'].iloc[-1], "BUY" if signal == "BUY" else "SELL", okx_data['close']
)
# ✅ Dynamic Signal System
signal_rules = {
    'BUY': [
        okx_data['RSI_14'].iloc[-1] < okx_data['Dynamic_RSI_Lower'].iloc[-1],
        okx_data['MACD'].iloc[-1] > okx_data['MACD_Signal'].iloc[-1],
        current_price > okx_data['SMA_20'].iloc[-1]
    ],
    'SELL': [
        okx_data['RSI_14'].iloc[-1] > okx_data['Dynamic_RSI_Upper'].iloc[-1],
        okx_data['MACD'].iloc[-1] < okx_data['MACD_Signal'].iloc[-1],
        current_price < okx_data['EMA_9'].iloc[-1]
    ]
}

signal = "HOLD"
if sum(signal_rules['BUY']) >= 2:
    signal = "BUY"
elif sum(signal_rules['SELL']) >= 2:
    signal = "SELL"

# ✅ Enhanced Visualization with Dynamic Adjustments
plt.figure(figsize=(16, 10))

# Price and Indicators with Risk Levels
plt.subplot(3, 1, 1)
plt.plot(okx_data['timestamp'], okx_data['close'], label='Price', color='royalblue')
plt.plot(okx_data['timestamp'], okx_data['EMA_9'], label='EMA 9', linestyle='--', color='orange')
plt.plot(okx_data['timestamp'], okx_data['SMA_20'], label='SMA 20', linestyle=':', color='green')
plt.scatter(okx_data['timestamp'].iloc[-1], predicted_price, color='red', s=100, label='Predicted Price')

# Add Stop Loss and Take Profit levels
plt.axhline(y=stop_loss, color='red', linestyle='--', label=f'Stop Loss: {stop_loss:.2f}')
plt.axhline(y=take_profit, color='green', linestyle='--', label=f'Take Profit: {take_profit:.2f}')

plt.title('Advanced Price Analysis with Dynamic Adjustments')
plt.xlabel('Timestamp')
plt.ylabel('Price (USDT)')
plt.legend()

# MACD and RSI Indicators with Dynamic Thresholds
plt.subplot(3, 1, 2)
plt.plot(okx_data['timestamp'], okx_data['MACD'], label='MACD', color='purple')
plt.plot(okx_data['timestamp'], okx_data['MACD_Signal'], label='Signal', color='darkorange')
plt.bar(okx_data['timestamp'], okx_data['RSI_14']-50, label='RSI-50', color='gray', alpha=0.3)
plt.axhline(okx_data['Dynamic_RSI_Lower'].iloc[-1], color='blue', linestyle='--', label='Dynamic RSI Lower')
plt.axhline(okx_data['Dynamic_RSI_Upper'].iloc[-1], color='green', linestyle='--', label='Dynamic RSI Upper')
plt.title('Momentum Indicators with Dynamic Thresholds')
plt.xlabel('Timestamp')
plt.ylabel('Indicator Value')
plt.legend()

# Volume and OBV Indicators
plt.subplot(3, 1, 3)
plt.bar(okx_data['timestamp'], okx_data['volume'], color='lightblue', label='Volume')
plt.plot(okx_data['timestamp'], okx_data['OBV'], color='darkblue', label='OBV')
plt.title('Volume and OBV Analysis')
plt.xlabel('Timestamp')
plt.ylabel('Volume')
plt.legend()

plt.tight_layout()
plt.show()

# ✅ Results Display
print(f"""
📊 Market Analysis Report with Dynamic Adjustments
───────────────────────────────────────────────────
• Current Price: {current_price:.2f} USDT
• Predicted Price: {predicted_price:.2f} USDT
• Trend Strength: {'Bullish' if okx_data['EMA_9'].iloc[-1] > okx_data['SMA_20'].iloc[-1] else 'Bearish'}

📈 Trading Signal: {signal}
  ├─ RSI: {okx_data['RSI_14'].iloc[-1]:.2f}
  ├─ Dynamic RSI Lower: {okx_data['Dynamic_RSI_Lower'].iloc[-1]:.2f}
  ├─ Dynamic RSI Upper: {okx_data['Dynamic_RSI_Upper'].iloc[-1]:.2f}
  ├─ MACD Cross: {'Bullish' if okx_data['MACD'].iloc[-1] > okx_data['MACD_Signal'].iloc[-1] else 'Bearish'}
  └─ Price Position: {'Above SMA20' if current_price > okx_data['SMA_20'].iloc[-1] else 'Below SMA20'}

⚖️ Risk Management
  ├─ Stop Loss: {stop_loss:.2f} USDT
  └─ Take Profit: {take_profit:.2f} USDT

💡 Recommendation: {'Hold Position' if signal == 'HOLD' else 'Execute '+signal+' Order'}
""")
