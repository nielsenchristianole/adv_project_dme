import os
import json
import pandas as pd
from pathlib import Path

# import matplotlib
# matplotlib.use('Agg') # Slim chance that matplotlib will crash the file, this is a fix
import cv2
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.projections import calc_tangent_xy_to_lat_lon, calc_lat_lon_to_tangent_xy, EARTH_RADIUS, EARTH_CIRCUMFERENCE

"""
This file will compute undistorted patches of the world in a predefined size.

"""

def filter_chunk(chunk, left_lon, right_lon, bot_lat, top_lat):
    if left_lon > right_lon:
        # Handle wrap-around at the date line
        return chunk[
            ((chunk['lon'] >= left_lon) | (chunk['lon'] <= right_lon)) &
            (chunk['lat'] >= bot_lat) & (chunk['lat'] <= top_lat)
        ]
    else:
        # Normal case
        return chunk[
            (chunk['lon'] >= left_lon) & (chunk['lon'] <= right_lon) &
            (chunk['lat'] >= bot_lat) & (chunk['lat'] <= top_lat)
        ]
        

class HeightMapProjector:
    def __init__(self, 
                 height_netcdf_path: os.PathLike,
                 city_csv_path: os.PathLike,
                 city_types: list[str],
                 output_dir: str,
                 km_per_px_resolution: float |str,
                 data_size_px: int,
                 plot: bool = False,
                 **kwargs
                 ):
        self.plot = plot
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)

        self.city_types = city_types
        self.city_csv_path = Path(city_csv_path)
        
        self.height_netcdf_path = Path(height_netcdf_path)
        self.km_per_px_resolution = km_per_px_resolution

        self.patch_size_px = int(data_size_px * np.sqrt(2))
        if type(self.km_per_px_resolution) != str:
            self.patch_size_km = self.km_per_px_resolution * self.patch_size_px
        
    def _compute_lonlats(self) -> tuple[np.ndarray, np.ndarray]:        
        # Compute delta longitude in degrees
        delta_longitude = np.degrees(self.patch_size_km / EARTH_RADIUS)
            
        # We assume delta latitude is the same as delta longitude
        longitudes = np.arange(-180, 180, delta_longitude)
        latitudes = np.arange(-90, 90, delta_longitude)
        
        return longitudes, latitudes, delta_longitude
    
    def _save_lonlat_id_csv(self, filename: os.PathLike) -> None:
        longitudes, latitudes, _ = self._compute_lonlats()
        
        # Pad to fit dataframe structure
        max_length = max(len(longitudes), len(latitudes))
        longitudes_extended = np.pad(longitudes, (0, max_length - len(longitudes)), constant_values=np.nan)
        latitudes_extended = np.pad(latitudes, (0, max_length - len(latitudes)), constant_values=np.nan)

        df = pd.DataFrame({
            'idx': np.arange(max_length),
            'longitudes': longitudes_extended,
            'latitudes': latitudes_extended
        })

        # Save to CSV
        df.to_csv(filename, index=False)
        
    def compute_height_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Main function to compute height maps. Saves a metadata file for each patch,
        and a csv for the longitudes and latitudes and the id folder of them.

        Returns:
            tuple[np.ndarray, np.ndarray]: Longitudes and latitudes used for the centers
        """
        # Load the height map data
        self.map = nc.Dataset(self.height_netcdf_path, 'r')
        self.hm_lon = self.map.variables['lon'][:].data
        self.hm_lat = self.map.variables['lat'][:].data
        self.elevation = self.map.variables['elevation'][:].data
                
        if self.km_per_px_resolution == 'dynamic':
            self.km_per_px_resolution = EARTH_CIRCUMFERENCE / self.elevation.shape[1]

        print("[STATUS] computing lon lats started")
        longitudes, latitudes, delta_longitude = self._compute_lonlats()
        self._save_lonlat_id_csv(self.output_dir / 'lonlat_id.csv')
        
        dataset_meta = {
            'scaled_patch_size_px': self.patch_size_px,
            'scaled_patch_size_km': self.patch_size_km,
            'global_max_height_km': float(np.max(self.elevation)),
            'global_min_height_km': float(np.min(self.elevation)), 
            'delta_longitude': float(delta_longitude),
        }
        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(dataset_meta, f, indent=4)
            
        print("[STATUS] computing lon lats done")
            
        for i, lat in enumerate(tqdm(latitudes, desc='Processing longitudes')):
            lat_path = self.output_dir / 'data' / str(i) 
            for j, lon in enumerate(tqdm(longitudes, desc='Processing latitudes')):
                lon_path = lat_path / str(j)
                
                # Project the height map
                projected_image, lonlat_bounds = self.project_and_interpolate(
                    central_lon=lon, 
                    central_lat=lat,
                    tolerance=delta_longitude + 0.5
                )
                # Skip patch
                if projected_image is None and lonlat_bounds is None:
                    continue
                
                lat_path.mkdir(exist_ok=True)
                lon_path.mkdir(exist_ok=True)
                np.save(lon_path / 'height_map.npy', projected_image)
                metadata = {
                    'central_lonlat': [lon, lat],
                    'lonlat_bounds': list(lonlat_bounds),
                    'height_variance': float(np.var(projected_image[projected_image >= 0])),
                    'min_height': float(np.min(projected_image)), 
                    'max_height': float(np.max(projected_image)),
                    'land_area_ratio': float(np.count_nonzero(projected_image >= 0) / projected_image.size)
                }
                with open(lon_path / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
    
    def _extract_height_map_slice(self, height_map: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
        """Returns a sliced height map and boundary box (not of the height map!) of the latitudes and longitudes for the projection"""
        # Find the min/max lat/lon that defines the region of interest
        bot_lat, top_lat = np.min(latitudes), np.max(latitudes)
        left_lon, right_lon = np.min(longitudes), np.max(longitudes)

        # Handle longitude wrap-around (crossing the dateline)
        if right_lon - left_lon > 181: # This should only be possible when going from -180 to 180
            right_lon = np.max(longitudes[longitudes < 0]) # West side
            left_lon = np.min(longitudes[longitudes > 0]) # East side

        lon_idx_left = np.searchsorted(self.hm_lon, left_lon, side='left')
        lon_idx_right = np.searchsorted(self.hm_lon, right_lon, side='right')
        lat_idx_bot = np.searchsorted(self.hm_lat, bot_lat, side='left') 
        lat_idx_top = np.searchsorted(self.hm_lat, top_lat, side='right')    
            
        # Extract slice from height map (handle wrap-around for longitude)
        if left_lon > right_lon:  # If dateline crossing is detected
            height_map_slice = np.concatenate(
                (height_map[lat_idx_bot:lat_idx_top, lon_idx_left:], height_map[lat_idx_bot:lat_idx_top, :lon_idx_right]),
                axis=1
            )
        else:
            height_map_slice = height_map[lat_idx_bot:lat_idx_top, lon_idx_left:lon_idx_right]
        
        return height_map_slice, left_lon, top_lat, right_lon, bot_lat
    
    def project_and_interpolate(self, central_lon, central_lat, tolerance) -> tuple[np.ndarray, tuple[float]]:
        """
        Will return None, None if the patch is close to the poles of the earth.
        Else it will return the projected image and the lonlat bounds (left_lon, top_lat, right_lon, bot_lat)
        """
        # Calculate the pixel coordinates
        x_vals = np.linspace(-self.patch_size_km / 2, self.patch_size_km / 2, self.patch_size_px)
        y_vals = np.linspace(-self.patch_size_km / 2, self.patch_size_km / 2, self.patch_size_px)

        x_plane, y_plane = np.meshgrid(x_vals, y_vals)

        # Project plane coordinates to lat/lon on the sphere
        longitudes, latitudes = calc_tangent_xy_to_lat_lon(x_plane, y_plane, central_lat, central_lon)
        # Skip any patches that are close to the poles and skip those
        if np.any((latitudes >= 90-tolerance) | (latitudes <= tolerance-90)):
            return None, None

        # Extract the height map slice based on lat/lon plane
        height_map_slice, left_lon, top_lat, right_lon, bot_lat = self._extract_height_map_slice(self.elevation, latitudes, longitudes)
        lonlat_bounds = (left_lon, top_lat, right_lon, bot_lat)
        
        # Adjust longitudes for wrap-around at the date line
        if left_lon > right_lon:
            longitudes[longitudes < left_lon] += 360
            right_lon += 360

        # Normalize the longitudes and latitudes to the range of the height map slice
        src_x = ((longitudes - left_lon) / (right_lon - left_lon) * (height_map_slice.shape[1] - 1)).astype(np.float32)
        src_y = ((latitudes - top_lat) / (bot_lat - top_lat) * (height_map_slice.shape[0] - 1)).astype(np.float32)
        # Use OpenCV remap for interpolation
        projected_image = cv2.remap(height_map_slice, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-4242)

        return projected_image, lonlat_bounds

    def compute_cities(self, longitudes: np.ndarray, latitudes: np.ndarray) -> None:
        """
        Computes cities and copies them over to a csv. If a patch has no city, it won't have a csv file.
        """    
        cities = pd.read_csv(self.city_csv_path)

        for i, lat in enumerate(tqdm(latitudes, desc='Processing latitudes')):
            lat_path = self.output_dir / 'data' / str(i)
            for j, lon in enumerate(tqdm(longitudes, desc='Processing longitudes')):
                lon_path = lat_path / str(j)
                json_metafile_path = lon_path / 'metadata.json'
                if not json_metafile_path.exists():
                    continue

                with open(json_metafile_path, 'r') as f:
                    metadata = json.load(f)
                    left_lon, top_lat, right_lon, bot_lat = metadata['lonlat_bounds']

                # Compute cities within the lonlat boundaries
                cities_within_boundaries = filter_chunk(cities, left_lon, right_lon, bot_lat, top_lat)
                
                # SKip if there are no cities within the boundaries
                if len(cities_within_boundaries) == 0:
                    metadata['nr_cities'] = 0
                    with open(json_metafile_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    continue
                
                T_x, T_y = calc_lat_lon_to_tangent_xy(cities_within_boundaries['lat'].values, cities_within_boundaries['lon'].values, lat, lon, self.patch_size_km, self.patch_size_px)
                valid_indices = (T_x >= 0) & (T_x < self.patch_size_px) & (T_y >= 0) & (T_y < self.patch_size_px)
                cities_within_boundaries = cities_within_boundaries[valid_indices]
                cities_within_boundaries['T_x'] = T_x[valid_indices]
                cities_within_boundaries['T_y'] = T_y[valid_indices]
                
                metadata['nr_cities'] = len(cities_within_boundaries)
                with open(json_metafile_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                # Save the filtered cities to a CSV file
                cities_within_boundaries.to_csv(lon_path / 'cities.csv', index=False)

    def plot_height_map_w_cities(self) -> None:
        """
        Plots the undistorted height map with the undistorted cities on top of it.
        """
        # Loop through latitude and longitude folders
        for _, lat_path in enumerate(tqdm((self.output_dir / 'data').iterdir(), desc='Processing latitudes')):
            for _, lon_path in enumerate(tqdm(lat_path.iterdir(), desc='Processing longitudes')):
                json_metafile_path = lon_path / 'metadata.json'
                cities_csv_path = lon_path / 'cities.csv'
                if not json_metafile_path.exists() or not cities_csv_path.exists():
                    continue

                # Read the height map
                height_map = np.load(lon_path / 'height_map.npy')
                masked_height_map = np.ma.masked_less_equal(height_map, 0)
                # Read the cities
                cities = pd.read_csv(cities_csv_path)
                plt.imshow(masked_height_map)
                plt.scatter(cities['T_x'], cities['T_y'], color='red')
                plt.savefig(lon_path / 'plot.png')
                plt.close()
        
    def run(self) -> None:
        self.compute_height_maps()
        longitudes, latitudes, _ = self._compute_lonlats()
        self.compute_cities(longitudes, latitudes)
        if self.plot:
            self.plot_height_map_w_cities()


if __name__ == '__main__':
    import yaml
    
    with open('src/data/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("[STATUS] Config loaded")
    
    projector = HeightMapProjector(**config)
    
    print("[STATUS] Projector initialized")
    
    projector.run()
 
