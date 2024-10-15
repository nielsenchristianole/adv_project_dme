from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def binary_search_closest(arr, target):
    """Binary search to find the closest element to the target in a sorted array."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    
    # Check the two closest values (low and high) after the loop
    if low >= len(arr): low = len(arr) - 1
    if high < 0: high = 0
    
    # Return the index of the closest value
    if abs(arr[low] - target) < abs(arr[high] - target):
        return low
    else:
        return high

def find_closest_folder(csv_path, query_longitude, query_latitude):
    # Step 1: Load the CSV
    df = pd.read_csv(csv_path)
    
    # Step 2: Extract sorted longitudes and latitudes
    longitudes = df['longitudes']
    latitudes = df['latitudes'].dropna()

    # Step 3: Perform binary search on longitudes
    closest_lon_idx = binary_search_closest(longitudes, query_longitude)
    
    # Step 4: Perform binary search on latitudes within the closest longitude
    closest_lat_idx = binary_search_closest(latitudes, query_latitude)
    
    # Step 5: Find the corresponding idx and create folder structure string    
    return f'{closest_lat_idx}/{closest_lon_idx}'


def show_plot(dataset_folder, path_folder, height_map=False):
    data_path = Path(dataset_folder) / 'data' / Path(path_folder)
    plot_path = data_path / 'plot.png'
    if plot_path.exists() and not height_map:
        img = Image.open(plot_path)
        plt.imshow(img)
        plt.axis('off')
    else:
        height_map = np.load(data_path / 'height_map.npy')
        plt.imshow(height_map)
    plt.show()
    
    
if __name__ == '__main__':
    # Example usage:
    csv_file = 'data/undistorted_data_ortho_2/lonlat_id.csv'
    query_lon, query_lat = 8.196084, 59.853582

  # Example longitude and latitude near the wrap-around boundary
    closest_folder = find_closest_folder(csv_file, query_lon, query_lat)
    print(f"The closest folder is: {closest_folder}")
    
    dataset_folder = 'data/undistorted_data_ortho_2'
    show_plot(dataset_folder, closest_folder, height_map=False)
