# Stock-Price LSTM Pipeline

A minimal, easy-to-use deep learning pipeline for time-series forecasting of stock prices using LSTM (Long Short-Term Memory) neural networks. This project is designed for rapid experimentation and prototyping with any OHLCV (Open, High, Low, Close, Volume) CSV data.

**Features:**
- **Training:** Learns from historical OHLCV data and saves a trained LSTM model and scaler.
- **Inference:** Loads the trained model to predict the next three periods of OHLC prices.

Ideal for traders, analysts, and ML enthusiasts who want a simple, script-based approach to stock price forecasting.

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

## Requirements

- Python 3.9+
- pip (latest recommended)
- See `requirements.txt` for all Python dependencies

---

## Project Structure

```
LSTM/
├── data/                  # Place your OHLCV CSV files here
│   └── aapl_5Y_1day_....csv
├── train_model.py         # Script to train the LSTM model
├── infer_predict.py       # Script for inference/prediction
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

*The scripts use the first `.csv` file found in the `data/` folder.*

---

## Quickstart

```bash
# 1. Clone the repository
 git clone https://github.com/deKryptomancer/LSTM.git
 cd LSTM

# 2. Create and activate a virtual environment
 python -m venv venv
 venv\Scripts\activate      # On Windows

# 3. Install dependencies
 pip install -r requirements.txt

# 4. Train the model
 python train_model.py

# 5. Run inference
 python infer_predict.py
```

---

## Usage Examples

**Training:**
```
python train_model.py
```
- Trains a 2-layer LSTM on your data (in `data/`) and saves:
  - `model.h5` (trained model)
  - `scaler.pkl` (MinMaxScaler)

**Inference:**
```
python infer_predict.py
```
- Loads the model/scaler, predicts the next 3 periods, prints results, and writes `future_predictions.csv`.

---

## CSV Requirements

| Required Column                | Notes                               |
| ------------------------------ | ----------------------------------- |
| `date` (or `Date`)             | ISO-8601 with or without timezone   |
| `open`, `high`, `low`, `close` | Numeric OHLC prices                 |
| `volume`                       | Numeric (used during training only) |

> Extra columns (e.g., `average`, `barCount`, unnamed index) are ignored automatically.

---

## Extending

- Change `n_steps`, add dropout, or predict more periods in the scripts.
- To forecast daily bars instead of hourly, use daily-resolution CSVs.
- For a live dashboard, wrap `infer_predict.py` into a Flask/FastAPI service.

---

## Contributing

Pull requests and suggestions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE) (or specify your license here)

---

**Happy trading!** 
