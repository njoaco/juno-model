# scripts/predict.py

import os
import numpy as np
import pandas as pd
import cryptocompare
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib  # Usaremos joblib en lugar de pickle

def download_crypto_data(symbol, currency='USD', limit=2100):
    """
    Descarga datos históricos diarios de la criptomoneda.
    Se descarga una cantidad mayor para asegurarse de contar con suficientes datos.
    """
    data = cryptocompare.get_historical_price_day(symbol, currency=currency, limit=limit)
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    crypto_symbol = 'XRP'
    currency = 'USD'
    window_size = 60
    forecast_horizon = 30  # Aunque el modelo fue entrenado para predecir 30 días en el futuro
    
    # Ubicación de los archivos del modelo y scaler
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
    model_filename = f"model_{crypto_symbol}.h5"
    model_path = os.path.join(model_dir, model_filename)
    
    scaler_filename = f"scaler_{crypto_symbol}.pkl"
    scaler_path = os.path.join(model_dir, scaler_filename)
    
    # Cargar el modelo y el scaler
    # Usamos compile=False para evitar problemas de recompilación
    model = load_model(model_path, compile=False)
    
    # Cargamos el scaler usando joblib.load (ya que se guardó con joblib.dump)
    scaler = joblib.load(scaler_path)
    
    # Descargar datos recientes (se necesitan al menos window_size + forecast_horizon días)
    df = download_crypto_data(crypto_symbol, currency, limit=window_size + forecast_horizon)
    data = df['close'].values.reshape(-1, 1)
    
    # Escalar los datos utilizando el scaler guardado
    data_scaled = scaler.transform(data)
    
    # Tomamos los últimos 'window_size' días para construir la secuencia de entrada
    input_sequence = data_scaled[-window_size:]
    input_sequence = input_sequence.reshape(1, window_size, 1)
    
    # Realizar la predicción (la salida estará escalada)
    prediction_scaled = model.predict(input_sequence)
    
    # Convertir la predicción a la escala original
    prediction = scaler.inverse_transform(prediction_scaled)
    print(f"Predicción para {crypto_symbol} a {forecast_horizon} días: {prediction[0][0]}")

if __name__ == '__main__':
    main()
