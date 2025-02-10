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
    crypto_symbol = input("Ingrese el símbolo de la criptomoneda (ej. BTC, ETH): ").upper()
    
    # Solicitar días a predecir (1-30)
    while True:
        try:
            days = int(input("Ingrese el número de días a predecir (1-30): "))
            if 1 <= days <= 30:
                break
            print("¡El valor debe estar entre 1 y 30!")
        except ValueError:
            print("¡Entrada inválida!")

    # Cargar modelo y scaler
    scaler = joblib.load(os.path.join(save_dir, f"scaler_{crypto_symbol}.pkl"))
    model = tf.keras.models.load_model(os.path.join(save_dir, f"model_{crypto_symbol}.h5"))

    # Obtener datos y preprocesar
    hist_data = cryptocompare.get_historical_price_day(crypto_symbol, currency="USD", limit=window_size)
    df = pd.DataFrame(hist_data)[['close', 'high', 'low', 'volumeto']]
    data_scaled = scaler.transform(df)

    # Preparar entrada
    input_sequence = data_scaled[-window_size:, 0].reshape(1, window_size, 1)
    
    # Predecir los 30 días
    predicted_sequence = model.predict(input_sequence)[0]
    
    # Escalar inverso solo para 'close'
    dummy_data = np.zeros((30, 4))
    dummy_data[:, 0] = predicted_sequence
    predicted_prices = scaler.inverse_transform(dummy_data)[:, 0]

    # Mostrar predicción del día seleccionado
    print(f"Predicción para {crypto_symbol} en el día {days}: {predicted_prices[days-1]:.2f} USD")

# Ejecutar la función principal
if __name__ == "__main__":
    main()
