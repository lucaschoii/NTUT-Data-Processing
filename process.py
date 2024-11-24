import numpy as np
from scipy.signal import lfilter, firwin, butter



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









