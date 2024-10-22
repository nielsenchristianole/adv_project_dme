from pathlib import Path

import cv2
import tqdm
import numpy as np
import netCDF4
import geopandas as gpd
import pandas as pd
import shapely



PATH = './data/gebco_flat/data/'
OUT_DIR = './data/height_contours/df_128/'
height_offsets = np.arange(0, 10_000, 25)
sanity_check = 0
diameter_range = (80, 128)


data_dir = Path(PATH)

data_frames = list()
lat_list = list(data_dir.iterdir())

if sanity_check:
    height_offsets = height_offsets[:2]
    lat_list = np.random.choice(lat_list, sanity_check, replace=False)

total = 0
for lat_dir in (pbar := tqdm.tqdm(lat_list, 'lat', leave=True)):
    lat_idx = int(lat_dir.name)

    lon_list = list(lat_dir.iterdir())
    if sanity_check:
        lon_list = np.random.choice(lon_list, sanity_check, replace=False)

    for lon_dir in tqdm.tqdm(lon_list, 'lon', leave=False):
        lon_idx = int(lon_dir.name)

        _path = lon_dir / 'height_map.npy'
        height_map = np.load(_path)

        for offset in tqdm.tqdm(height_offsets, 'offsets', leave=False):

            mask = (height_map > offset)

            if not mask.any():
                break

            contours, hierarchy = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue

            # we want hills, not valleys
            outer = (hierarchy[0, :, 3] == -1)
            contours = [contours[i] for i in range(len(contours)) if outer[i]]

            geoms = map(lambda x: x.squeeze(1), contours)
            geoms = filter(lambda x: x.shape[0] >= 4, geoms)
            geoms = map(lambda x: shapely.Polygon(x), geoms)
            geoms = filter(lambda x: x.is_valid, geoms)

            gdf = gpd.GeoDataFrame(geometry=list(geoms))

            if gdf.empty:
                continue

            gdf['diameter'] = 2 * gdf.minimum_bounding_radius()
            gdf = gdf[gdf['diameter'].between(*diameter_range)].reset_index(drop=True).copy()

            if gdf.empty:
                continue

            gdf['lat'] = lat_idx
            gdf['lon'] = lon_idx
            gdf['offset'] = offset
            gdf['path'] = str(_path)

            centers = gdf.minimum_bounding_circle().centroid
            gdf['center_x'] = centers.x
            gdf['center_y'] = centers.y

            data_frames.append(gdf)

            total += len(gdf)
            pbar.set_postfix_str(f'Total: {total}')


df = pd.concat(data_frames, ignore_index=True)
gdf = gpd.GeoDataFrame(df)

out_dir = Path(OUT_DIR)
out_dir.mkdir(exist_ok=True, parents=True)

gdf.to_file(out_dir / 'df.shp')
