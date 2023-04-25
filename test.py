import numpy as np
from numpy.lib.stride_tricks import as_strided

# Define the original array and window size
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window_size = 3
overlap_size = 2

# Calculate the number of windows and the stride size
n_windows = 1 + (len(arr) - window_size) // overlap_size
stride_size = arr.strides[0] * overlap_size

# Create a view of the array with overlapping windows
windows = as_strided(arr, shape=(n_windows, window_size), strides=(stride_size, arr.strides[0]))

# Concatenate the windows along a new axis
concatenated = np.concatenate(windows, axis=0)

# Print the original array, windows, and concatenated array
print("Original array:", arr)
print("Overlapping windows:", windows)
print("Concatenated array:", concatenated)