import numpy as np
import pandas as pd
import os

# Generate input data RADIANS
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

# Create a DataFrame
data = pd.DataFrame({
    'x': x,
    'sin(x)': y
})

# Save Data
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, "sine_data.csv")

data.to_csv(file_path, index=False)

print(f"CSV file saved at: {file_path}")


