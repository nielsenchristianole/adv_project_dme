import numpy as np
from typing import Tuple

# Earth constants in km
EARTH_CIRCUMFERENCE = 40075
EARTH_RADIUS = 6378.137


def calc_tangent_xy_to_lat_lon(x_plane, y_plane, central_lat, central_lon, sphere_radius=EARTH_RADIUS):
    # Convert central lat/lon to radians
    central_lat_rad = np.radians(central_lat)
    central_lon_rad = np.radians(central_lon)

    # Calculate distance (rho) from the center of the plane for each point
    rho = np.sqrt(x_plane**2 + y_plane**2)

    # Handle small rho values to avoid division by zero
    rho[rho == 0] = 1e-10
    c = 2 * np.arctan(rho / (2 * sphere_radius))
    
    # Calculate latitudes and longitudes
    lat = np.arcsin(np.cos(c) * np.sin(central_lat_rad) + (y_plane * np.sin(c) * np.cos(central_lat_rad)) / rho)
    lon = central_lon_rad + np.arctan2(x_plane * np.sin(c), rho * np.cos(central_lat_rad) * np.cos(c) - y_plane * np.sin(central_lat_rad) * np.sin(c))

    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    # Normalize longitudes to the range [-180, 180]
    lon_deg = (lon_deg + 180) % 360 - 180

    return lon_deg, lat_deg

def calc_lat_lon_to_tangent_xy(lat: float, 
                               lon: float, 
                               central_lat: float, 
                               central_lon: float, 
                               patch_size_km: float = None, 
                               patch_size_px: int = None,
                               sphere_radius=EARTH_RADIUS,
                               mode='px_topleft') -> Tuple[float,float]:
    """Projects a given latitude and longitude on a tangent plane from a central latlon
    Can output either pixel coordinates from topleft origin or km coordinates in the center, by
    setting mode to 'px_topleft' or 'km_center' respectively.
    
    Args:
        lat (float): Latitude to project
        lon (float): Longitude to project
        central_lat (float): Latitude of the origin of the tangent plane
        central_lon (float): Longitude of the origin of the tangent plane
        patch_size_km (float): Tangent plane size in km
        patch_size_px (int): Tangent plane size in px
        sphere_radius (float): Radius of sphere used. Defaults to EARTH_RADIUS.
        mode (str): Either 'px_topleft' or 'km_center'. Defaults to 'px_topleft'.

    Returns:
        Tuple[float,float]: (x,y) from given origin and units
    """
    lat = np.radians(lat)
    lon = np.radians(lon)
    central_lat = np.radians(central_lat)
    central_lon = np.radians(central_lon)

    x = sphere_radius * np.cos(lat) * np.sin(lon - central_lon)
    y = sphere_radius * (np.cos(central_lat) * np.sin(lat) - np.sin(central_lat) * np.cos(lat) * np.cos(lon - central_lon))

    if mode == 'km_center':
        return x,y
    elif mode == 'px_topleft':
        # Calculate the pixel coordinates
        x_pixel = ((x + patch_size_km  / 2) / patch_size_km  * patch_size_px)
        y_pixel = ((-y + patch_size_km  / 2) / patch_size_km  * patch_size_px)
        return x_pixel, y_pixel
    else:
        AssertionError(f"Unknown mode: {mode}")