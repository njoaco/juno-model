import os
import numpy as np
import pandas as pd
import cryptocompare
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configuración
crypto_symbol = "XRP"
look_back = 60  # Ventana de datos pasados
epochs = 50
batch_size = 16

# Crear la carpeta de guardado si no existe
save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
os.makedirs(save_dir, exist_ok=True)

# Obtener datos históricos
print(f"Descargando datos para {crypto_symbol}...")
hist_data = cryptocompare.get_historical_price_day(crypto_symbol, currency="USD", limit=2000)
df = pd.DataFrame(hist_data)
df["time"] = pd.to_datetime(df["time"], unit="s")
df.set_index("time", inplace=True)

# Normalización
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[["close"]])

# Guardar el scaler
scaler_path = os.path.join(save_dir, f"scaler_{crypto_symbol}.pkl")
joblib.dump(scaler, scaler_path)

# Crear secuencias para entrenar
X, y = [], []
for i in range(len(df_scaled) - look_back - 30):
    X.append(df_scaled[i : i + look_back, 0])
    y.append(df_scaled[i + look_back + 30, 0])  # Predicción a 30 días

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Definir modelo LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Entrenar el modelo
print(f"Entrenando el modelo para {crypto_symbol}...")
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

# Guardar el modelo
model_path = os.path.join(save_dir, f"model_{crypto_symbol}.h5")
model.save(model_path)

print(f"Modelo guardado en {model_path}")
print(f"Scaler guardado en {scaler_path}")
