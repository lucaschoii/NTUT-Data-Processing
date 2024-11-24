import matplotlib.pyplot as plt
import pickle
from process import remove_drift, bandpass_filter, segment_data

with (open("data.pkl", "rb")) as openfile:
    data = pickle.load(openfile)
x = data[0]
y = data[1]
z = data[2]

# Remove drift from the data
x_drift_removed = remove_drift(x)
y_drift_removed = remove_drift(y)
z_drift_removed = remove_drift(z)

# Apply bandpass filter to the data
x_filtered = bandpass_filter(x_drift_removed)
y_filtered = bandpass_filter(y_drift_removed)
z_filtered = bandpass_filter(z_drift_removed)

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)

plt.plot(x, label='X')
plt.plot(x_drift_removed, label='X (Drift Removed)')
plt.plot(x_filtered, label='X (Filtered)')

plt.title("X Accelerometer Data")
plt.legend()


plt.subplot(3, 1, 2)
plt.plot(y, label='Y')
plt.plot(y_drift_removed, label='Y (Drift Removed)')
plt.plot(y_filtered, label='Y (Filtered)')

plt.title("Y Accelerometer Data")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(z, label='Z')
plt.plot(z_drift_removed, label='Z (Drift Removed)')
plt.plot(z_filtered, label='Z (Filtered)')

plt.title("Z Accelerometer Data")
plt.legend()

plt.show()