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

# Configuración
window_size = 60  # Ventana de datos pasados (60 días)
epochs = 100
batch_size = 16
API_KEY = "fddef7417bfd43abbd422118e0a38b10"

# Crear la carpeta de guardado si no existe
save_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(save_dir, exist_ok=True)

def get_stock_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={API_KEY}&outputsize={window_size}"
    response = requests.get(url).json()
    if "values" not in response:
        raise ValueError("Error al obtener datos de TwelveData")
    df = pd.DataFrame(response["values"])
    df = df.rename(columns={"close": "close", "high": "high", "low": "low", "volume": "volumeto"})
    df = df.sort_values(by="datetime").reset_index(drop=True)
    return df

def main():
    prediction_dir = os.path.join(os.path.dirname(__file__), "..", "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    asset_type = sys.argv[1] if len(sys.argv) > 1 else "1"
    asset_type = "crypto" if asset_type == "1" else "stock"

    symbol = input(f"Ingrese el símbolo del {asset_type} (ej. BTC para cripto, AAPL para stock): ").upper()

    if asset_type == "crypto":
        current_price = cryptocompare.get_price(symbol, currency='USD')[symbol]['USD']
        hist_data = cryptocompare.get_historical_price_day(symbol, currency="USD", limit=window_size)
    else:
        df = get_stock_data(symbol)
        current_price = float(df.iloc[-1]['close'])
        hist_data = df.to_dict('records')
    
    print(f"\nPrecio actual de {symbol}: ${current_price:.2f} USD")
    
    while True:
        try:
            days = int(input("Ingrese el número de días a predecir (1-30): "))
            if 1 <= days <= 30:
                break
            print("¡El valor debe estar entre 1 y 30!")
        except ValueError:
            print("¡Entrada inválida!")

    scaler = joblib.load(os.path.join(save_dir, f"scaler_{symbol}.pkl"))
    model = tf.keras.models.load_model(os.path.join(save_dir, f"model_{symbol}.h5"))
    
    df = pd.DataFrame(hist_data)[['close', 'high', 'low', 'volumeto']]
    data_scaled = scaler.transform(df)

    input_sequence = data_scaled[-window_size:, 0].reshape(1, window_size, 1)
    predicted_sequence = model.predict(input_sequence)[0]
    
    dummy_data = np.zeros((30, 4))
    dummy_data[:, 0] = predicted_sequence
    predicted_prices = scaler.inverse_transform(dummy_data)[:, 0]

    print(f"Predicción para {symbol} en el día {days}: ${predicted_prices[days-1]:.2f} USD")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(prediction_dir, f"pred_{symbol}_{timestamp}.txt")

    report_content = f"""=== Predicción {symbol} ===
    Fecha/Hora: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
    Precio actual: ${current_price:.2f}
    Día predicho: {days}
    Precio predicho: ${predicted_prices[days-1]:.2f}
    Histórico de 30 días:
    """ + "\n".join([f"Día {i+1}: ${price:.2f}" for i, price in enumerate(predicted_prices)])
    
    with open(filename, 'w') as f:
        f.write(report_content)
    
    print(f"\nPredicción guardada en: {filename}")

if __name__ == "__main__":
    main()
