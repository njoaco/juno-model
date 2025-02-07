import os
import numpy as np
import pandas as pd
import cryptocompare
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configuración
window_size = 60  # Ventana de datos pasados (60 días)
epochs = 100
batch_size = 16

# Crear la carpeta de guardado si no existe
save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
os.makedirs(save_dir, exist_ok=True)

# Cargar scaler guardado


def main():

    crypto_symbol = input("Ingrese el símbolo de la criptomoneda (ej. BTC, ETH): ").upper()  # 

    scaler_path = os.path.join(save_dir, f"scaler_{crypto_symbol}.pkl")  # 🆕 Nombre dinámico
    scaler = joblib.load(scaler_path)

    model_path = os.path.join(save_dir, f"model_{crypto_symbol}.h5")  # 🆕 Nombre dinámico
    model = tf.keras.models.load_model(model_path)

    # Obtener los últimos 60 precios de cierre para la predicción
    print(f"Obteniendo los últimos datos de {crypto_symbol}...")
    hist_data = cryptocompare.get_historical_price_day(crypto_symbol, currency="USD", limit=window_size)
    df = pd.DataFrame(hist_data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    # Asegurarse de tener las columnas correctas en el DataFrame para usar con el scaler
    df = df[['close', 'high', 'low', 'volumeto']]  # Orden correcto (igual que en train_model.py)

    # Escalar los datos (incluir todas las columnas como en el entrenamiento)
    data_scaled = scaler.transform(df)

    # Crear la secuencia de entrada para la predicción (usando solo la columna 'close')
    input_sequence = data_scaled[-window_size:, 2]  # El índice 2 corresponde a 'close' (la columna de precios de cierre)

    # Redimensionar para que el modelo pueda aceptarlo
    input_sequence = input_sequence.reshape(1, window_size, 1)  # (1, 60, 1) para LSTM

    # Realizar la predicción
    predicted_price = model.predict(input_sequence)
    dummy_data = np.zeros((1, 4))
    dummy_data[:, 0] = predicted_price  # 'close' es la primera columna en el scaler
    predicted_price = scaler.inverse_transform(dummy_data)[0][0]  # Extraer solo el valor 'close'

    # Imprimir la predicción
    print(f"Predicción de precio para {crypto_symbol} a 30 días: {predicted_price}")

# Ejecutar la función principal
if __name__ == "__main__":
    main()
