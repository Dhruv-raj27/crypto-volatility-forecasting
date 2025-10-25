import os
import pandas as pd
import numpy as np

def preprocess_crypto_data(input_file, output_file, rolling_window=7):
    """
    Load, clean, and compute volatility features from a crypto CSV file.
    """
    # Load CSV
    df = pd.read_csv(input_file)

    # Convert Date column to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Keep only Open and Close prices
    df = df[['Open', 'Close']].dropna()

    # Calculate log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Calculate rolling volatility
    df[f'Volatility_{rolling_window}d'] = df['Log_Return'].rolling(window=rolling_window).std()

    # Drop initial NaNs (from rolling calc)
    df.dropna(inplace=True)

    # Save to CSV
    df.to_csv(output_file)
    print(f"✅ Processed data saved to: {output_file}")


if __name__ == "__main__":
    # Set input and output paths
    coin = "Bitcoin"
    input_path = f"data/coin_{coin}.csv"
    output_path = f"outputs/processed_{coin}.csv"

    # Run preprocessing
    preprocess_crypto_data(input_path, output_path)