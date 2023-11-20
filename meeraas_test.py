import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'diabetic_data.csv'  
data = pd.read_csv(file_path)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data['weight'], data['admission type'], alpha=0.5)
plt.title('Scatter Plot of Weight vs Admission Type')
plt.xlabel('Weight')
plt.ylabel('Admission Type')
plt.grid(True)
plt.show()
