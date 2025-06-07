import pickle
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

print(os.getcwd())
# Load the content of the provided pickle file to inspect it
file_path = 'town3_waypoints_turn_right_1.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)


# Define the path to save the CSV file
csv_file_path = 'town3_waypoints_turn_right_1.csv'

# Extract location_x and location_y from the list of dictionaries
coordinates = [(item['location_x'], item['location_y']) for item in data]

# Write the coordinates to a CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['location_x', 'location_y'])  # Write the header
    csv_writer.writerows(coordinates)  # Write the rows


# Load the CSV file with the waypoints
csv_file_path = 'town3_waypoints_turn_right_1.csv'
waypoints = pd.read_csv(csv_file_path)

# Extract location_x and location_y
x = waypoints['location_x'].values
y = waypoints['location_y'].values

# Apply a smoothing spline to the data
smoothing_factor = 0.001  # Adjustable parameter
spl_x = UnivariateSpline(np.arange(len(x)), x, s=smoothing_factor)
spl_y = UnivariateSpline(np.arange(len(y)), y, s=smoothing_factor)

# Generate smoothed points
smoothed_x = spl_x(np.arange(len(x)))
smoothed_y = spl_y(np.arange(len(y)))

# Plot the original and smoothed points
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='Original Points', alpha=0.6)
plt.plot(smoothed_x, smoothed_y, 'b-', label='Smoothed Points', linewidth=2)
plt.title('Original vs Smoothed Waypoints')
plt.xlabel('location_x')
plt.ylabel('location_y')
plt.legend()
plt.grid(True)
plt.show()

# Calculate curvature
dx = np.gradient(smoothed_x)
dy = np.gradient(smoothed_y)
ddx = np.gradient(dx)
ddy = np.gradient(dy)

curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5


# Create a DataFrame for the smoothed data
smoothed_data = pd.DataFrame({
    'smoothed_x': smoothed_x,
    'smoothed_y': smoothed_y,
    'curvature': curvature
})

# Define the path to save the smoothed data as a CSV file
smoothed_csv_file_path = 'town4_long_r0d2.csv'

# Save the smoothed data to a CSV file
smoothed_data.to_csv(smoothed_csv_file_path, index=False)


