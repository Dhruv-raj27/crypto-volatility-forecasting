import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load processed data
df = pd.read_csv('outputs/processed_Bitcoin.csv', parse_dates=['Date'], index_col='Date')
df = df.dropna(subset=['Log_Return', 'Volatility_7d'])

# Load trained model
model = load_model('outputs/lstm_volatility_model.h5', compile=False)

# Prepare features
def create_sequences(data, window=30):
    X = []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
    return np.array(X)

log_returns = df['Log_Return'].values.reshape(-1, 1)
scaler = StandardScaler()
log_returns_scaled = scaler.fit_transform(log_returns)

window_size = 30
X_all = create_sequences(log_returns_scaled, window=window_size)

# Predict volatility with LSTM
pred_vol = model.predict(X_all).flatten()

# Align predictions with actual returns
returns = df['Log_Return'].values[window_size:]

# Simulate portfolios
unhedged = returns

# Hedging: reduce exposure when volatility is high
threshold = 0.04  # can tune this
hedged = []
for i in range(len(pred_vol)):
    if pred_vol[i] > threshold:
        # Apply hedge: reduce return by 70% or short exposure
        hedged.append(returns[i] * 0.3)
    else:
        hedged.append(returns[i])
hedged = np.array(hedged)

# Convert to cumulative returns
unhedged_cum = np.cumsum(unhedged)
hedged_cum = np.cumsum(hedged)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(unhedged_cum, label='Unhedged Portfolio')
plt.plot(hedged_cum, label='LSTM-Hedged Portfolio', linestyle='--')
plt.title('Portfolio Returns: Unhedged vs LSTM-Based Hedging')
plt.xlabel('Days')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def calculate_metrics(returns):
    cumulative_return = np.sum(returns)
    drawdowns = np.maximum.accumulate(np.cumsum(returns)) - np.cumsum(returns)
    max_drawdown = np.max(drawdowns)
    sharpe = np.mean(returns) / np.std(returns)
    return cumulative_return, max_drawdown, sharpe

un_cum, un_dd, un_sharpe = calculate_metrics(unhedged)
hed_cum, hed_dd, hed_sharpe = calculate_metrics(hedged)

print(f"📈 Unhedged  → Return: {un_cum:.2f}, Max Drawdown: {un_dd:.2f}, Sharpe: {un_sharpe:.2f}")
print(f"🛡️  Hedged    → Return: {hed_cum:.2f}, Max Drawdown: {hed_dd:.2f}, Sharpe: {hed_sharpe:.2f}")
