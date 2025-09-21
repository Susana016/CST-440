import os
import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(__file__)
tflite_path = os.path.join(base_dir, "Model", "sine_model.tflite")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_values = np.linspace(0, np.pi, 10, dtype=np.float32)

print("Running inference on TFLite model:")
for val in test_values:
    input_data = np.array([[val]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    print(f"x = {val:.3f}, predicted sin(x) = {prediction:.4f}, true sin(x) = {np.sin(val):.4f}")
