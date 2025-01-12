import numpy as np
from scipy.signal import lfilter, firwin, butter

from scipy.signal import find_peaks

from spectrum import pburg

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


def bandpass_filter(data, fs=500, lowcut=1, highcut=30, numtaps=101, order=4):
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

def detect_peak_frequency(window, fs=100, low_freq=3, high_freq=8, ar_order=6):
    '''
    Detects the peak frequency in a window of data.

    Args:
        window: array of accelerometer data
        fs: sampling frequency
        low_freq: low cutoff frequency
        high_freq: high cutoff frequency
        ar_order: autoregressive model order
    
    Returns:
        peak_frequency: peak frequency in the window, otherwise 1
    '''
    # Compute the power spectral density using the Burg method  
    psd = pburg(window, ar_order, sampling=fs, NFFT=512)
    f = np.asarray(psd.frequencies())
    psd = psd.psd

    # Filter frequencies within the specified range
    freq_mask = (f >= low_freq) & (f <= high_freq)
    filtered_freqs = f[freq_mask]
    filtered_psd = psd[freq_mask]
    
    if len(filtered_freqs) == 0:
        return 1  # Return 1 if no frequencies are within the range

    # Find the peak frequency within the filtered range
    peaks, _ = find_peaks(filtered_psd)
    if len(peaks) == 0:
        return 1  # Return 1 if no peak is found

    peak_index = peaks[np.argmax(filtered_psd[peaks])]
    peak_frequency = filtered_freqs[peak_index]
    
    return peak_frequency








