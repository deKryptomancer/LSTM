# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Load CSV
csv_folder = './data'
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in ./csv/")
csv_path = os.path.join(csv_folder, csv_files[0])
print(f"📈 Training on: {csv_path}")

# Load & format
data = pd.read_csv(csv_path)
data = data.rename(columns={
    'date': 'Date', 'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
})
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
scaled_df = pd.DataFrame(scaled_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# Create sequences: input → next 5 steps of OHLC
def create_sequences(data, n_steps=30, n_future=5):
    X, y = [], []
    for i in range(len(data) - n_steps - n_future):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps:i + n_steps + n_future])
    return np.array(X), np.array(y)

n_steps = 30
n_future = 5
X, y = create_sequences(scaled_df[['Open', 'High', 'Low', 'Close']].values, n_steps, n_future)
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]

# Reshape y to (samples, 5*4) for dense layer
y_train = y_train.reshape((y_train.shape[0], -1))

# Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(n_steps, 4)),
    LSTM(50),
    Dense(n_future * 4)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save model
model.save("model.keras")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved.")
