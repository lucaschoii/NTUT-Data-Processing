import numpy as np
from scipy.signal import lfilter, firwin, butter
import matplotlib.pyplot as plt
import pickle


def remove_drift(data, window_size=50):
    '''
    Removes drift from data using a moving average filter.

    Args:
        data: array of accelerometer data
        window_size: size of the moving average window

    Returns:
        drift_removed_data: array of accelerometer data with drift removed
    
    '''
    data = np.array(data)
    
    # Create a moving average filter
    b = np.ones(window_size) / window_size
    a = np.array([1, -0.8])

    # Apply the filter to the data
    drift_removed_data = lfilter(b, a, data)

    return drift_removed_data


def bandpass_filter(data, fs=500, lowcut=3, highcut=8, numtaps=101, order=4):
    '''
    Applies a bandpass filter to data.

    Args:
        data: array of accelerometer data
        fs: sampling frequency
        lowcut: low cutoff frequency
        highcut: high cutoff frequency
        numtaps: number of filter taps

    Returns:
        filtered_data: array of accelerometer data with drift removed
    
    '''
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered_data = lfilter(b, a, data)
    return filtered_data

def segment_data(data, window_size, overlap_ratio):
    '''
    Segments the data into overlapping windows.

    Args:
        data: array of accelerometer data
        window_size: size of the window
        overlap_ratio: ratio of overlap between windows

    Returns:
        segmented_data: array of segmented data
    
    '''
    segmented_data = []
    stride = int(window_size * (1 - overlap_ratio))
    for i in range(0, len(data) - window_size, stride):
        segment = data[i:i + window_size]
        segmented_data.append(segment)

    return np.array(segmented_data)


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






