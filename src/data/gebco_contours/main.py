from pathlib import Path

import numpy as np
import geopandas as gpd

from src.data.gebco_contours.extract_gebco_contours_functions import extract_contours, combine_contours, filter_df
from src.data.gebco_contours.utils import transform_geom

# True=slow, False=uses a lot of memory
RELOAD_GEBCO = True

out_root_dir = Path('./data/contours/')
# sub_dirs = ['contours_orig', 'contours_sub_ice']
sub_dirs = ['contours_orig']
nc_files = ['./data/charts/GEBCO_2024.nc', './data/charts/GEBCO_2024_sub_ice_topo.nc']
height_offsets = np.arange(0, 9000, 25, np.int16)

mult = 2

diameter_range = (mult * 80_000, mult * 128_000)


# # create contours from the height map
# extract_contours(
#     out_root_dir=out_root_dir,
#     nc_files=nc_files,
#     sub_dirs=sub_dirs,
#     height_offsets=height_offsets,
#     reload_gebco=RELOAD_GEBCO)

# # read in all the contours and create a dataframe
df_out = combine_contours(
    out_root_dir=out_root_dir,
    sub_dirs=sub_dirs,
    diameter_range=diameter_range,
    save_intermediate=False)

try:
    # save
    _out = out_root_dir / f'df_{mult * 64}'
    _out.mkdir(exist_ok=True, parents=True)
    df_out.to_file(_out / 'df.shp')
except Exception as e:
    import pdb
    pdb.set_trace()

# try:
#     # filter and save
#     df_filtered, _ = filter_df(df_out, iou_threshold=0.9, filteree_dataset='contours_sub_ice')
#     _out = out_root_dir / 'df_postfilter'
#     _out.mkdir(exist_ok=True, parents=True)
#     df_filtered.to_file(_out / 'df.shp')
# except Exception as e:
#     import pdb
#     pdb.set_trace()
