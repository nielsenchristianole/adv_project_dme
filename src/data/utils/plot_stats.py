import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def load_json_files(base_dir, include_patches_wo_cities):
    data = []
    for lat_folder in tqdm(Path(base_dir).iterdir(), desc='Processing latitudes'):
        if not lat_folder.is_dir():
            continue
        for lon_folder in lat_folder.iterdir():
            if not lon_folder.is_dir():
                continue
            json_metafile_path = lon_folder / 'metadata.json'
            if not json_metafile_path.exists():
                continue
            if not include_patches_wo_cities and not (lon_folder / 'cities.csv').exists():
                continue
            with open(json_metafile_path, 'r') as f:
                metadata = json.load(f)
                data.append(metadata)
    return data

def plot_histograms(data):
    nr_cities = []
    height_variance = []
    land_area_ratio = []

    for entry in data:
        nr_cities.append(entry.get('nr_cities', 0))
        height_variance_value = entry.get('height_variance', np.nan)
        if not np.isnan(height_variance_value):
            height_variance.append(height_variance_value)
        land_area_ratio_value = entry.get('land_area_ratio', np.nan)
        if land_area_ratio_value != np.nan:
            land_area_ratio.append(land_area_ratio_value)

    # Plot histogram for nr_cities
    plt.figure()
    plt.hist(nr_cities, bins=20, edgecolor='black')
    plt.title('Histogram of Number of Cities')
    plt.xlabel('Number of Cities')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plot histogram for height_variance
    plt.figure()
    plt.hist(height_variance, bins=20, edgecolor='black')
    plt.title('Histogram of Height Variance')
    plt.xlabel('Height Variance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plot histogram for land_area_ratio
    plt.figure()
    plt.hist(land_area_ratio, bins=20, edgecolor='black')
    plt.title('Histogram of Land Area Ratio')
    plt.xlabel('Land Area Ratio')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    # Example usage
    include_patches_wo_cities=False
    base_dir = 'data/undistorted_data_ortho_2/data'
    data = load_json_files(base_dir, include_patches_wo_cities)
    plot_histograms(data)