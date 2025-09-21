import os
import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(__file__)
tflite_path = os.path.join(base_dir, "..", "Model", "sine_model.tflite")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

x_test = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
y_true = np.sin(x_test)

y_pred = []

for val in x_test:
    input_data = np.array([[val]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    y_pred.append(prediction)

y_pred = np.array(y_pred)

mse = np.mean((y_true - y_pred) ** 2)
mae = np.mean(np.abs(y_true - y_pred))

print("Model Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

for i in range(0, len(x_test), 20):
    print(f"x = {x_test[i]:.3f}, predicted = {y_pred[i]:.4f}, true = {y_true[i]:.4f}")
