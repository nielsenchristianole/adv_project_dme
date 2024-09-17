from os import PathLike
from pathlib import Path

import shapely
import numpy as np
import geopandas as gpd
import netCDF4 as nc
import matplotlib.pyplot as plt

from src.data.utils.shapely_utils import apply_modulo_to_geom, project_coords_to_axis


class DatasetPreprocessor:

    def __init__(self,
        chart_dir: PathLike,
        file_names: list[str]|None=None,
        *,
        max_mem_size: int|None = None,
        dtype: type=np.int16) -> None:
        
        self.dtype = dtype
        self.max_mem_size = max_mem_size
        chart_dir = Path(chart_dir)

        # arbitrary constants
        self.eps = 0.0001
        self.pole_coord = 90 - self.eps
        
        # what files we're working with
        self.file_names = file_names or [
            'GEBCO_2024',
            'GEBCO_2024_sub_ice_topo',
            'GEBCO_2024_TID']
        
        # extract the lon and lat from the first file
        file = nc.Dataset(chart_dir / f'{self.file_names[0]}.nc', 'r')
        self.lon: np.ndarray = file.variables['lon'][:].data
        self.lat: np.ndarray = file.variables['lat'][:].data

        # extract the array of interest from the data
        ignore = {'lon', 'lat', 'crs'}
        self.filemap = {
            name: (_d := nc.Dataset(chart_dir / f'{name}.nc', 'r')).variables[(set(_d.variables.keys()) - ignore).pop()]
            for name in self.file_names[1:]}
        self.filemap[self.file_names[0]] = file.variables[(set(file.variables.keys()) - ignore).pop()]

    def _read_array(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        *,
        x_step: int = 1,
        y_step: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes the slices and asserts that we don't use too much memory and returns the data
        """

        width = x_max - x_min
        height = y_max - y_min
        assert (self.max_mem_size is None) or (width * height // (y_step * x_step) <= self.max_mem_size), 'Memory error'

        out_data = np.zeros(
            (len(self.file_names), height // y_step, width // x_step), dtype=self.dtype)
        for i, name in enumerate(self.file_names):
            out_data[i] = self.filemap[name][y_min:y_max:y_step, x_min:x_max:x_step]

        # add the lat and lon
        indexes = np.stack(np.meshgrid(self.lat[y_min:y_max:y_step], self.lon[x_min:x_max:x_step], indexing='ij'), axis=0)
        return out_data, indexes

    def read_maps(
        self,
        geom: shapely.MultiPolygon | shapely.Polygon,
        *,
        use_smallest_longitudinal_section: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        geom: shapely.MultiPolygon | shapely.Polygon
            - the geometry used to query the data
        use_smallest_longitudinal_section: bool
            - use the narrow or wide longitudinal section
        """

        lon_min, lat_min, lon_max, lat_max = geom.bounds
        y_min = np.where(self.lat < lat_min)[0][-1].item() if np.any(self.lat < lat_min) else 0
        y_max = np.where(self.lat > lat_max)[0][0].item() if np.any(self.lat > lat_max) else self.lat.size
        y_max = min(y_max + 1, self.lat.size)

        # contains north pole
        if geom.convex_hull.contains(shapely.Point(0, self.pole_coord)):
            x_min = 0
            x_max = self.lon.size
            y_max = self.lat.size

            return self._read_array(x_min, x_max, y_min, y_max, x_step=10)

        # contains south pole
        elif geom.convex_hull.contains(shapely.Point(0, -self.pole_coord)):
            x_min = 0
            x_max = self.lon.size
            y_min = 0

            return self._read_array(x_min, x_max, y_min, y_max, x_step=10)

        moved_geom = apply_modulo_to_geom(geom, mod_x=360)
        normal_width = abs(geom.bounds[0] - geom.bounds[2])
        moved_width = abs(moved_geom.bounds[0] - moved_geom.bounds[2])

        # check if the section is more narrow if crossing the international date line
        if (normal_width - moved_width > self.eps) == use_smallest_longitudinal_section:
            
            # get the longitude of every coord
            coords = project_coords_to_axis(geom, x_axis=True)
            coords = np.sort(coords)
            gaps = np.diff(coords)
            gap_idx = np.argmax(gaps)

            # left side
            x_min = 0
            lon_max = coords[gap_idx]
            x_max = np.where(self.lon > lon_max)[0][0].item() if np.any(self.lon > lon_max) else self.lon.size
            x_max = min(x_max + 1, self.lon.size)

            left = self._read_array(x_min, x_max, y_min, y_max)

            # right side
            lon_min = coords[gap_idx]
            x_min = np.where(self.lon < lon_min)[0][-1].item() if np.any(self.lon < lon_min) else 0
            x_max = self.lon.size

            right = self._read_array(x_min, x_max, y_min, y_max)

            return np.concatenate((left, right), axis=-1)

        # get the indices
        x_min = np.where(self.lon < lon_min)[0][-1].item()
        x_max = np.where(self.lon > lon_max)[0][0].item()
        x_max = min(x_max + 1, self.lon.size)

        return self._read_array(x_min, x_max, y_min, y_max)


if __name__ == '__main__':
    esri_dir = Path('data/vectors/')
    path = esri_dir / 'ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'
    gdf = gpd.read_file(path)


    dataset_preprocessor = DatasetPreprocessor('data/charts/')
    antarctica_idx = np.stack(gdf['geometry'].apply(lambda geom: np.array(geom.centroid.xy)).to_list()).squeeze(-1)[:, 1].argmin(axis=0)
    chart, xy = dataset_preprocessor.read_maps(gdf.loc[antarctica_idx].geometry)
    plt.matshow(chart[0])
    plt.show()
