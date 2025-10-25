import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load processed data
df = pd.read_csv('outputs/processed_Bitcoin.csv', parse_dates=['Date'], index_col='Date')
df = df.dropna(subset=['Log_Return', 'Volatility_7d'])

# Create LSTM sequences
def create_sequences(data, target, window=30):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(target[i])
    return np.array(X), np.array(y)

log_returns = df['Log_Return'].values.reshape(-1, 1)
vol_target = df['Volatility_7d'].values

# Scale log returns
scaler = StandardScaler()
log_returns_scaled = scaler.fit_transform(log_returns)

# Create sequences
window_size = 30
X, y = create_sequences(log_returns_scaled, vol_target, window=window_size)

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print("Training LSTM...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Predict
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.6f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test, label='Actual Volatility')
plt.plot(y_pred, label='Predicted Volatility (LSTM)', alpha=0.7)
plt.title('Actual vs LSTM Predicted Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save model
model.save('outputs/lstm_volatility_model.h5')
print("✅ Model saved to outputs/lstm_volatility_model.h5")
