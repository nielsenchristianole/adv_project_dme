from typing import Optional

import numpy as np
import shapely
import geopandas as gpd
import scipy.spatial


def LLHtoECEF(lat: np.ndarray, lon: np.ndarray, alt: Optional[np.ndarray]=None) -> np.ndarray:
    # see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html
    if alt is None:
        alt = np.zeros_like(lat)
    
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    rad = np.float64(6_378_137.0)[None]        # Radius of the Earth (in meters)
    f = np.float64(1.0 / 298.257223563)[None]  # Flattening factor WGS84 Model
    cosLat = np.cos(lat)
    sinLat = np.sin(lat)
    FF     = (1.0 - f) ** 2
    C      = 1 / np.sqrt(cosLat ** 2 + FF * sinLat ** 2)
    S      = C * FF

    x = (rad * C + alt) * cosLat * np.cos(lon)
    y = (rad * C + alt) * cosLat * np.sin(lon)
    z = (rad * S + alt) * sinLat

    return np.stack((x, y, z), axis=0)


def get_diameter(series: gpd.GeoSeries) -> float:
    if isinstance(series.geometry, shapely.Point):
        return 0.
    coords = np.array(list(series.geometry.coords))
    coords = LLHtoECEF(coords[:, 1], coords[:, 0])
    return scipy.spatial.distance.pdist(coords.T).max()


def get_perimeter(series: gpd.GeoSeries) -> float:
    if isinstance(series.geometry, shapely.Point):
        return 0.
    coords = np.array(list(series.geometry.coords))
    coords = LLHtoECEF(coords[:, 1], coords[:, 0])
    return np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()


def get_coords(series: gpd.GeoSeries) -> np.ndarray:

    if isinstance(series.geometry, shapely.Point):
        return np.zeros((2, 1))

    coords = np.array(list(series.geometry.coords))
    coords = LLHtoECEF(coords[:, 1], coords[:, 0])
    center = np.array(series.geometry.centroid.coords).flatten()
    center = LLHtoECEF(center[[1]], center[[0]])
    center /= np.linalg.norm(center)

    coords = coords - (center * (center.T @ coords))
    idx = np.argmax(np.linalg.norm(center - coords, axis=0))

    vec1 = coords[:, [idx]]
    vec1 /= np.linalg.norm(vec1)
    vec2 = np.cross(center, vec1, axis=0)
    vec2 /= np.linalg.norm(vec2)

    basechange = np.hstack((vec1, vec2))
    return basechange.T @ coords


def get_area(series: gpd.GeoSeries) -> float:
    if isinstance(series.geometry, shapely.Point):
        return 0.
    coords = get_coords(series)
    return type(series.geometry)(coords.T).area


def transform_geom(series: gpd.GeoSeries) -> shapely.geometry:
    coords = get_coords(series)
    if isinstance(series.geometry, shapely.Point):
        return shapely.Point(coords.squeeze(1))
    return type(series.geometry)(coords.T)
