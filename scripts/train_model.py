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

# Cargar variables de entorno del archivo .env
load_dotenv()

# Configuración
look_back = 60  # Ventana de datos pasados
epochs = 100
batch_size = 16

API_KEY = os.getenv("TWELVEDATA_API_KEY")
if not API_KEY:
    raise ValueError("No se encontró TWELVEDATA_API_KEY en el archivo .env")

# Crear la carpeta de guardado si no existe
save_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(save_dir, exist_ok=True)

def get_stock_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={API_KEY}&outputsize=2000"
    response = requests.get(url).json()
    if "values" not in response:
        error_message = response.get("message", "No se pudo obtener datos.")
        raise ValueError(f"Error al obtener datos de TwelveData: {error_message}")
    df = pd.DataFrame(response["values"])
    df = df.rename(columns={"close": "close", "high": "high", "low": "low", "volume": "volumeto"})
    df = df.sort_values(by="datetime").reset_index(drop=True)
    return df

def main():
    asset_type = sys.argv[1] if len(sys.argv) > 1 else "1"
    asset_type = "crypto" if asset_type == "1" else "stock"
    
    symbol = input(f"Ingrese el símbolo del {asset_type} (ej. BTC para cripto, AAPL para stock): ").upper()
    
    if asset_type == "crypto":
        # Obtener y mostrar el precio actual para criptomonedas
        current_price = cryptocompare.get_price(symbol, currency='USD')[symbol]['USD']
        print(f"\nPrecio actual de {symbol}: ${current_price:.2f} USD")
        hist_data = cryptocompare.get_historical_price_day(symbol, currency="USD", limit=2000)
        df = pd.DataFrame(hist_data)
    else:
        # Obtener datos para stocks y mostrar el precio actual (último cierre)
        df = get_stock_data(symbol)
        current_price = float(df.iloc[-1]['close'])
        print(f"\nPrecio actual de {symbol}: ${current_price:.2f} USD")
    
    # Asegurarse de que las columnas tengan el nombre esperado y eliminar valores nulos
    df = df.rename(columns={"close": "close", "high": "high", "low": "low", "volume": "volumeto"})
    df = df.dropna()
    
    # Normalización
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[["close", "high", "low", "volumeto"]])
    
    scaler_path = os.path.join(save_dir, f"scaler_{symbol}.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Crear secuencias para entrenamiento
    X, y = [], []
    for i in range(len(df_scaled) - look_back - 30):
        X.append(df_scaled[i : i + look_back, 0])
        y.append(df_scaled[i + look_back : i + look_back + 30, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Configuración del gráfico en tiempo real para el entrenamiento
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], label='Pérdida (Loss)')
    ax.set_title(f"Entrenamiento de {symbol} - Pérdida por Época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
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
            plt.pause(0.1)  # Pausa para actualizar el gráfico
    
    # Definición del modelo LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(30)  # Salida para 30 días
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    print(f"Entrenando el modelo para {symbol}...")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[LossCallback()])
    
    # Guardar el modelo
    model_path = os.path.join(save_dir, f"model_{symbol}.h5")
    model.save(model_path)
    
    print(f"Modelo guardado en {model_path}")
    print(f"Scaler guardado en {scaler_path}")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
