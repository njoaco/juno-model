import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import requests
import cryptocompare
from dotenv import load_dotenv

load_dotenv()

window_size = 60


save_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(save_dir, exist_ok=True)

API_KEY = os.getenv("TWELVEDATA_API_KEY")
if not API_KEY:
    raise ValueError("TWELVEDATA_API_KEY not found in .env file")

def get_stock_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={API_KEY}&outputsize={window_size}"
    response = requests.get(url).json()
    if "values" not in response:
        raise ValueError("Error fetching data from TwelveData: " + response.get("message", "Unable to retrieve data."))
    df = pd.DataFrame(response["values"])
    df = df.rename(columns={"close": "close", "high": "high", "low": "low", "volume": "volumeto"})
    df = df.sort_values(by="datetime").reset_index(drop=True)
    return df

def main():
    prediction_dir = os.path.join(os.path.dirname(__file__), "..", "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    asset_type = sys.argv[1] if len(sys.argv) > 1 else "1"
    asset_type = "crypto" if asset_type == "1" else "stock"

    symbol = input("Enter the asset symbol (e.g. BTC for crypto, AAPL for stock): ").upper()

    if asset_type == "crypto":
        current_price = cryptocompare.get_price(symbol, currency='USD')[symbol]['USD']
        print(f"\nCurrent price of {symbol}: ${current_price:.2f} USD")
        hist_data = cryptocompare.get_historical_price_day(symbol, currency="USD", limit=window_size)
    else:
        df = get_stock_data(symbol)
        current_price = float(df.iloc[-1]['close'])
        print(f"\nCurrent price of {symbol}: ${current_price:.2f} USD")
        hist_data = df.to_dict('records')

    while True:
        try:
            days = int(input("Enter the number of days to predict (1-30): "))
            if 1 <= days <= 30:
                break
            print("The value must be between 1 and 30!")
        except ValueError:
            print("Invalid input!")

    scaler = joblib.load(os.path.join(save_dir, f"scaler_{symbol}.pkl"))
    model = tf.keras.models.load_model(os.path.join(save_dir, f"model_{symbol}.h5"))

    df = pd.DataFrame(hist_data)[['close', 'high', 'low', 'volumeto']]
    data_scaled = scaler.transform(df)

    input_sequence = data_scaled[-window_size:, 0].reshape(1, window_size, 1)
    predicted_sequence = model.predict(input_sequence)[0]

    dummy_data = np.zeros((30, 4))
    dummy_data[:, 0] = predicted_sequence
    predicted_prices = scaler.inverse_transform(dummy_data)[:, 0]

    predicted_price = predicted_prices[days-1]
    print(f"Prediction for {symbol} on day {days}: ${predicted_price:.2f} USD")
    
    percentage_change = ((predicted_price - current_price) / current_price) * 100
    
    if percentage_change > 5:
        recommendation = "Buy"
    elif percentage_change < -5:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
    
    print(f"Recommendation: {recommendation} (Change: {percentage_change:.2f}%)")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(prediction_dir, f"pred_{symbol}_{timestamp}.txt")

    report_content = f"""=== Prediction for {symbol} ===
Date/Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Current Price: ${current_price:.2f}
Predicted Day: {days}
Predicted Price: ${predicted_price:.2f}
Recommendation: {recommendation} (Change: {percentage_change:.2f}%)
30-Day History:
""" + "\n".join([f"Day {i+1}: ${price:.2f}" for i, price in enumerate(predicted_prices)])

    with open(filename, 'w') as f:
        f.write(report_content)

    print(f"\nPrediction saved to: {filename}")

if __name__ == "__main__":
    main()
