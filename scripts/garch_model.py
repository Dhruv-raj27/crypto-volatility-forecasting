import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Load data
df = pd.read_csv('./outputs/processed_Bitcoin.csv', parse_dates=['Date'], index_col='Date')
returns = df['Log_Return'].dropna()

# Fit GARCH(1,1)
model = arch_model(returns, vol='Garch', p=1, q=1)
res = model.fit(disp="off")

# Forecast volatility
vol_forecast = res.conditional_volatility
df['GARCH_Forecast'] = vol_forecast

# Plot
plt.figure(figsize=(14, 5))
plt.plot(df['Volatility_7d'], label='Realized Volatility (7-day)', color='green')
plt.plot(df['GARCH_Forecast'], label='GARCH(1,1) Forecast', color='red')
plt.title('Bitcoin Volatility: Realized vs GARCH Forecast')
plt.xlabel('Date')
plt.ylabel('Volatility (Standard Deviation)')
plt.legend()
plt.tight_layout()
plt.show()