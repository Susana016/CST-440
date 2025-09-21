import os
import tensorflow as tf

base_dir = os.path.dirname(__file__)
keras_path = os.path.join(base_dir, "sine_model.keras")
tflite_path = os.path.join(base_dir, "sine_model.tflite")

model = tf.keras.models.load_model(keras_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Model converted and saved to {tflite_path}")
