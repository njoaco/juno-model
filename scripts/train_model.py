import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
import cryptocompare
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

look_back = 60
epochs = 100
batch_size = 16

API_KEY = os.getenv("TWELVEDATA_API_KEY")
if not API_KEY:
    raise ValueError("TWELVEDATA_API_KEY not found in .env file")

save_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(save_dir, exist_ok=True)

def get_stock_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={API_KEY}&outputsize=2000"
    response = requests.get(url).json()
    if "values" not in response:
        error_message = response.get("message", "Unable to retrieve data.")
        raise ValueError(f"Error fetching data from TwelveData: {error_message}")
    df = pd.DataFrame(response["values"])
    df = df.rename(columns={"close": "close", "high": "high", "low": "low", "volume": "volumeto"})
    df = df.sort_values(by="datetime").reset_index(drop=True)
    return df

def main():
    asset_type = sys.argv[1] if len(sys.argv) > 1 else "1"
    asset_type = "crypto" if asset_type == "1" else "stock"
    
    symbol = input("Enter the asset symbol (e.g. BTC for crypto, AAPL for stock): ").upper()
    
    if asset_type == "crypto":
        current_price = cryptocompare.get_price(symbol, currency='USD')[symbol]['USD']
        print(f"\nCurrent price of {symbol}: ${current_price:.2f} USD")
        hist_data = cryptocompare.get_historical_price_day(symbol, currency="USD", limit=2000)
        df = pd.DataFrame(hist_data)
    else:
        df = get_stock_data(symbol)
        current_price = float(df.iloc[-1]['close'])
        print(f"\nCurrent price of {symbol}: ${current_price:.2f} USD")
    
    df = df.rename(columns={"close": "close", "high": "high", "low": "low", "volume": "volumeto"})
    df = df.dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[["close", "high", "low", "volumeto"]])
    
    scaler_path = os.path.join(save_dir, f"scaler_{symbol}.pkl")
    joblib.dump(scaler, scaler_path)
    
    X, y = [], []
    for i in range(len(df_scaled) - look_back - 30):
        X.append(df_scaled[i : i + look_back, 0])
        y.append(df_scaled[i + look_back : i + look_back + 30, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], label='Loss')
    ax.set_title(f"Training for {symbol} - Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    
    loss_history = []
    
    class LossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss_history.append(logs['loss'])
            line.set_data(range(len(loss_history)), loss_history)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(30)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    print(f"Training model for {symbol}...")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[LossCallback()])
    
    model_path = os.path.join(save_dir, f"model_{symbol}.h5")
    model.save(model_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
