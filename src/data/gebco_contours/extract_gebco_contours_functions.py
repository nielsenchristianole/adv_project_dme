from pathlib import Path
from typing import List, Tuple, Union

import cv2
import tqdm
import numpy as np
import netCDF4
import geopandas as gpd
import pandas as pd
import shapely

from src.data.gebco_contours.utils import get_coords


def extract_contours(
    out_root_dir: Path,
    nc_files: List[str],
    sub_dirs: List[str],
    height_offsets: np.ndarray,
    reload_gebco: bool=True
) -> None:
    """
    load in the height map and extract and save the contours
    """

    for path, out_dir in zip(nc_files, tqdm.tqdm(sub_dirs, 'datasets', leave=False)):

        path = Path(path)
        out_dir = out_root_dir / out_dir
        out_dir.mkdir(exist_ok=True, parents=True)

        _df = netCDF4.Dataset(path, 'r')
        lat = _df.variables['lat'][:].data
        lon = _df.variables['lon'][:].data

        np.save(
            out_dir / 'lat.npy',
            lat)
        np.save(
            out_dir / 'lon.npy',
            lon)
        
        if not reload_gebco:
            gebco_data = _df.variables['elevation'][:].data

        for offset in tqdm.tqdm(height_offsets, 'contours', leave=False):

            _out = out_dir / f'{offset.item():0>4}'
            _out.mkdir(exist_ok=True)

            out_cont = _out / 'contours.npy'
            out_hier = _out / 'hierarchy.npy'
            out_idxs = _out / 'idxs.npy'

            if out_cont.exists() and out_hier.exists() and out_idxs.exists():
                continue

            # I know this is inefficient, byt I am almost maxing out my memory
            # and don't want to store anything unneeded. This script only has to
            # run once
            contours, hierarchy = cv2.findContours(
                (
                    (_df.variables['elevation'][:].data if reload_gebco else gebco_data) > offset
                ).astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            
            # we assume that no contours will be found in subsequent heights
            if not contours:
                break

            idxs = np.array([0] + [c.shape[0] for c in contours], dtype=np.int32)
            idxs = np.cumsum(idxs)
            
            # shape = (n, 1, (lon, lat))
            contours = np.concatenate(contours, axis=0).squeeze(1)
            contours = np.stack(
                (
                    lon[contours[:, 0]],
                    lat[contours[:, 1]]
                ),
                axis=1)

            np.save(
                out_cont,
                contours)
            np.save(
                out_hier,
                hierarchy)
            np.save(
                out_idxs,
                idxs)


def combine_contours(
    out_root_dir: Path,
    sub_dirs: List[str],
    diameter_range: Tuple[int, int],
    save_intermediate: bool=False
) -> gpd.GeoDataFrame:
    """
    read all the saved contours, filter and save them in a geodataframe
    """

    current_len = 0
    dataframes = list()
    for sub_dir in tqdm.tqdm(sub_dirs, 'dataset', leave=False):

        pbar = tqdm.tqdm(list((out_root_dir / sub_dir).iterdir()), 'creating dataset', leave=False)
        for _dir in pbar:

            if not _dir.is_dir():
                continue

            cont_path = _dir / 'contours.npy'
            hier_path = _dir / 'hierarchy.npy'
            idxs_path = _dir / 'idxs.npy'

            if not cont_path.exists() or not hier_path.exists() or not idxs_path.exists():
                continue

            contours = np.load(cont_path)
            hierarchy = np.load(hier_path).squeeze(0).T
            idxs = np.load(idxs_path)

            geometries = list()
            for idx0, idx1 in zip(
                tqdm.tqdm(idxs[:-1], 'iterating contours', leave=False),
                idxs[1:]):

                c = contours[idx0:idx1]
                if c.shape[0] > 3:
                    geometries.append(
                        shapely.LinearRing(c))
                elif c.shape[0] > 1:
                    geometries.append(
                        shapely.LineString(c))
                else:
                    geometries.append(
                        shapely.Point(c.squeeze(0)))
            
            _df = gpd.GeoDataFrame(geometry=geometries)
            _df['c_idx'] = np.arange(len(_df), dtype=np.int32)
            _df['c_next'] = hierarchy[0]
            _df['c_previous'] = hierarchy[1]
            _df['c_1srchild'] = hierarchy[2]
            _df['c_parent'] = hierarchy[3]
            _df['num_coords'] = _df.geometry.apply(lambda x: len(x.coords))

            _df = _df[_df['num_coords'] >= 4].copy()

            if not len(_df):
                continue
            try:
                # think it might have due to no length. Unsure
                _polygons = gpd.GeoDataFrame(geometry=[shapely.Polygon(coords.T) for coords in _df.apply(get_coords, axis=1)], index=_df.index)
            except Exception:
                print('error with:', sub_dir, _dir)
                continue

            _df['diameter'] = 2 * _polygons.minimum_bounding_radius()
            _mask = _df['diameter'].between(*diameter_range)
            _polygons = _polygons[_mask].copy()
            _df = _df[_mask].copy()

            if not len(_df):
                continue

            _df['center_lon'] = _df.centroid.x
            _df['center_lat'] = _df.centroid.y

            _df.geometry = _polygons.geometry
            _df['perim_3d'] = _df.geometry.length
            _df['area'] = _df.geometry.area

            _df['dataset'] = sub_dir
            _df['height'] = int(_dir.name)

            dataframes.append(_df)
            current_len += len(_df)
            pbar.set_postfix({'len': current_len})

            if save_intermediate:
                try:
                    (_dir / 'df').mkdir(exist_ok=True)
                    _df.to_file(_dir / 'df' /'df.shp')
                except Exception as e:
                    print('error with:', sub_dir, _dir, e)
                    continue

    df = pd.concat(dataframes, ignore_index=True)
    df = gpd.GeoDataFrame(df)
    print(f'Found {len(df)} contours')

    return df


def filter_df(df: gpd.GeoDataFrame, iou_threshold: float, filteree_dataset: str, min_num_coords: int=4) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    """
    iou_threshold: float
        The filtering similarity threshold
    filteree_dataset: str
        Which dataset to remove geometries from
    min_num_coords: int
        Minimum number of coordinates for a geometry to be considered (remember that last coordinate is duplicated)
    """

    df = df.copy()
    keep = np.zeros(len(df), dtype=bool)

    # filter out to small geometries
    df['num_coords'] = df.geometry.apply(lambda x: len(x.exterior.coords))
    df = df[(df['num_coords'] >= min_num_coords).to_numpy()].copy()

    # filter out invalid geometries
    df = df[np.array([shapely.Polygon(geom).is_valid for geom in df.geometry])].copy()
    df['keep'] = np.ones(len(df), dtype=bool)

    filteree_mask = (df['dataset'] == filteree_dataset).to_numpy()
    for idx, geom in enumerate(pbar := tqdm.tqdm(df[filteree_mask].geometry.to_list(), 'filtering', leave=False)):
        geom = shapely.Polygon(geom)

        i = df[~filteree_mask].geometry.apply(lambda other: geom.intersection(shapely.Polygon(other)).area)
        u = df[~filteree_mask].geometry.apply(lambda other: geom.union(shapely.Polygon(other)).area)
        if (i / u).max() > iou_threshold:
            df.loc[df[filteree_mask].index[idx], 'keep'] = False

    df = df[df['keep']].copy().drop(columns=['keep', 'num_coords'])

    keep[df.index] = True

    return df.reset_index(drop=True).copy(), keep
