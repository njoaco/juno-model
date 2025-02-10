import os
import numpy as np
import pandas as pd
import cryptocompare
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# Configuración
crypto_symbol = input("Ingrese el símbolo de la criptomoneda (ej. BTC, ETH): ").upper()
look_back = 60  # Ventana de datos pasados
epochs = 100
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

# Verificar datos nulos o ceros
if df[["close", "high", "low", "volumeto"]].isnull().values.any():
    raise ValueError("Los datos contienen valores nulos.")

# Normalización
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[["close", "high", "low", "volumeto"]])

# Guardar el scaler
scaler_path = os.path.join(save_dir, f"scaler_{crypto_symbol}.pkl")
joblib.dump(scaler, scaler_path)

# Crear secuencias para entrenar
X, y = [], []
for i in range(len(df_scaled) - look_back - 30):
    X.append(df_scaled[i : i + look_back, 0])  # Ventana de entrada
    y.append(df_scaled[i + look_back : i + look_back + 30, 0])  # Secuencia de 30 días

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Definir modelo LSTM (salida de 30 neuronas)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(30)  # Salida para 30 días
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Configurar matplotlib para modo interactivo
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], label='Pérdida (Loss)')
ax.set_title(f"Entrenamiento de {crypto_symbol} - Pérdida por Época")
ax.set_xlabel("Época")
ax.set_ylabel("Pérdida")
ax.legend()
ax.grid(True)

# Lista para almacenar la pérdida
loss_history = []

# Callback para actualizar la pérdida
class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss_history.append(logs['loss'])
        line.set_data(range(len(loss_history)), loss_history)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)  # Pausa para actualizar el gráfico

# Entrenar el modelo
print(f"Entrenando el modelo para {crypto_symbol}...")
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[LossCallback()])

# Guardar el modelo
model_path = os.path.join(save_dir, f"model_{crypto_symbol}.h5")
model.save(model_path)

print(f"Modelo guardado en {model_path}")
print(f"Scaler guardado en {scaler_path}")

# Mantener la ventana abierta al finalizar
plt.ioff()
plt.show()