````markdown
# Stock-Price LSTM Pipeline

A minimal, easy-to-use deep learning pipeline for time-series forecasting of stock prices using LSTM (Long Short-Term Memory) neural networks. This project is designed for rapid experimentation and prototyping with any OHLCV (Open, High, Low, Close, Volume) CSV data. It features a two-stage workflow:

- **Training**: Learns from historical OHLCV data and saves a trained LSTM model and scaler.
- **Inference**: Loads the trained model to predict the next three periods of OHLC prices.

This project is ideal for traders, analysts, and ML enthusiasts who want a simple, script-based approach to stock price forecasting.

---

## Tech Stack

| Component      | Technology/Library    |
| -------------- | --------------------- |
| Language       | Python 3.9+           |
| Deep Learning  | TensorFlow (Keras)    |
| Data Handling  | pandas, numpy         |
| Preprocessing  | scikit-learn          |
| Model Storage  | joblib, Keras         |
| Environment    | Virtualenv/venv       |

---

## 1. Prerequisites

| Tool   | Tested Version |
|--------|----------------|
| Python | 3.9+           |
| pip    | 22+            |

---

## 2. Setup

```bash
# 1. Clone / copy the repo
git clone <your-repo> && cd <your-repo>

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
````

---

## 3. Folder Layout

```
project/
├── csv/                   # put your OHLCV CSVs here
│   └── aapl_1Y_1hour.csv  # any name is fine
├── train_model.py
├── infer_predict.py
├── requirements.txt
└── README.md
```

*The scripts always pick **the first** `.csv` they find in `./csv/`.*

---

## 4. Quick Start

### 4.1 Train the model

```bash
python train_model.py
```

* Builds a 2-layer LSTM, trains 50 epochs, and saves:

  * `model.h5` – trained network
  * `scaler.pkl` – fitted MinMaxScaler

### 4.2 Run inference

```bash
python infer_predict.py
```

* Loads the saved model/scaler
* Predicts the next **3** periods
* Prints a table and writes `future_predictions.csv`.

Example output:

```
📅 Future Predictions:
         Open        High         Low       Close                Date
0  218.12 ...  219.30 ...  217.55 ...  218.44 ...  2025-06-17 11:00:00
1  218.45 ...  219.70 ...  217.90 ...  218.80 ...  2025-06-17 12:00:00
2  218.78 ...  220.05 ...  218.20 ...  219.15 ...  2025-06-17 13:00:00
```

---

## 5. CSV Requirements

| Required Column                | Notes                               |
| ------------------------------ | ----------------------------------- |
| `date` (or `Date`)             | ISO-8601 with or without timezone   |
| `open`, `high`, `low`, `close` | Numeric OHLC prices                 |
| `volume`                       | Numeric (used during training only) |

> Extra columns (e.g., `average`, `barCount`, unnamed index) are ignored automatically.

---

## 6. Extending

* Change `n_steps`, add dropout, or predict more periods.
* To forecast daily bars instead of hourly, just supply daily-resolution CSVs.
* For a live dashboard, wrap `infer_predict.py` into a Flask/FastAPI service.
