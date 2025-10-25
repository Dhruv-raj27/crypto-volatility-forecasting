# 🪙 Crypto Volatility Forecasting

This project focuses on modeling and forecasting the **volatility of cryptocurrencies** using both **statistical models (GARCH)** and **deep learning models (LSTM)**. The primary asset studied so far is **Bitcoin (BTC)**, with plans to include other cryptocurrencies like Ethereum (ETH) and more.

---

## 📁 Project Structure

```
crypto-volatility-forecasting/
│
├── data/                 ← Raw CSV files (e.g., BTC.csv, ETH.csv)
├── notebooks/            ← Jupyter notebooks for EDA, modeling
├── scripts/              ← Python scripts for preprocessing and training
├── outputs/              ← Processed data, saved models, and graphs
├── requirements.txt      ← Python package dependencies
├── .gitignore            ← Files to exclude from version control
└── README.md             ← Project overview (this file)
```

---

## 🔍 Features Completed

### ✅ Data Preprocessing
- Cleaned and aligned historical price data
- Calculated **log returns** and **7-day rolling volatility**
- Saved processed dataset to `outputs/processed_Bitcoin.csv`

### ✅ Exploratory Data Analysis (EDA)
- Visualized Bitcoin close price over time
- Plotted daily log returns and realized volatility

### ✅ GARCH(1,1) Volatility Modeling
- Implemented using `arch` package
- Forecasted conditional volatility
- Compared with 7-day realized volatility in visual plots

### ✅ LSTM-Based Volatility Forecasting
- Trained an LSTM model using TensorFlow/Keras
- Used log returns as input to predict rolling volatility
- Evaluated and visualized predictions vs actual volatility
- Saved model as `outputs/lstm_volatility_model.h5`

### ✅ LSTM-Based Hedging Simulation
- Simulated a portfolio with and without hedging based on predicted volatility
- Compared cumulative returns and drawdowns

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/crypto-volatility-forecasting.git
cd crypto-volatility-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare data

Place your raw CSV files (e.g., BTC.csv, ETH.csv) in the `data/` folder.

Then run:

```bash
python scripts/data_preprocessing.py
```

### 4. Run models

To train and visualize models:

```bash
python scripts/garch_model.py              # GARCH model
python scripts/lstm_model.py               # LSTM model
python scripts/hedging_simulation.py       # Hedging simulation
```

---

## 📊 Results So Far

| Portfolio     | Return | Max Drawdown | Sharpe Ratio |
|---------------|--------|--------------|---------------|
| Unhedged      | 5.64   | 1.87         | 0.04          |
| LSTM-Hedged   | 4.04   | 1.81         | 0.04          |

> 📈 Visual results are available in the `notebooks/` folder or shown during script execution.

---

## 🔜 Next Steps

- Add ETH and other cryptocurrency volatility modeling
- Cross-compare GARCH vs LSTM across assets
- Enhance model performance with tuning and more features
- Evaluate risk metrics and hedge effectiveness

---

## 📚 Requirements

- Python ≥ 3.8
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib
- Scikit-learn, arch

> 📦 See `requirements.txt` for the complete list.
---