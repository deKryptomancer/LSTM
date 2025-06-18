# infer_predict.py
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")

# Load latest CSV
csv_folder = './data'
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in ./csv/")
csv_path = os.path.join(csv_folder, csv_files[0])
print(f"🔮 Predicting with: {csv_path}")

# Preprocess
data = pd.read_csv(csv_path)
data = data.rename(columns={
    'date': 'Date', 'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
})
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
scaled_df = pd.DataFrame(scaled_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# Inference (5 steps ahead)
n_steps = 30
n_future = 5
last_sequence = scaled_df[['Open', 'High', 'Low', 'Close']].values[-n_steps:]
input_seq = last_sequence.reshape(1, n_steps, 4)

raw_output = model.predict(input_seq, verbose=0)[0]  # shape: (20,)
reshaped_output = raw_output.reshape(n_future, 4)

# Inverse scale
padded = np.hstack([reshaped_output, np.zeros((n_future, 1))])
inv_scaled = scaler.inverse_transform(padded)[:, :4]

# Add future timestamps
last_date = data['Date'].iloc[-1]
time_diff = data['Date'].diff().median()
future_dates = [last_date + (i + 1) * time_diff for i in range(n_future)]

# Output
pred_df = pd.DataFrame(inv_scaled, columns=['Open', 'High', 'Low', 'Close'])
pred_df['Date'] = future_dates

print("\n📅 Next 5 Predictions:")
print(pred_df)
pred_df.to_csv("future_predictions.csv", index=False)
