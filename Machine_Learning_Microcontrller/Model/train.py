import os
import pandas as pd
import tensorflow as tf

# Load dataset from Data folder
csv_path = os.path.join(os.path.dirname(__file__), "..", "Data", "sine_data.csv")
data = pd.read_csv(csv_path)

X_train = data[['x']].values
Y_train = data[['sin(x)']].values

# Define small model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=200, verbose=0)

# Save Keras model
keras_path = os.path.join(os.path.dirname(__file__), "sine_model.keras")
model.save(keras_path)
print(f"✅ Model trained and saved to {keras_path}")
