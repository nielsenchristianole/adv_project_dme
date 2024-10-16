from pathlib import Path
from typing import List, Tuple, Union

import cv2
import tqdm
import numpy as np
import netCDF4
import geopandas as gpd
import pandas as pd
import shapely



sub_dirs = ['contours_orig', 'contours_sub_ice']
nc_files = ['./data/charts/GEBCO_2024.nc', './data/charts/GEBCO_2024_sub_ice_topo.nc']




file = netCDF4.Dataset(nc_files[0], 'r').variables['elevation'][::2, ::2].data != netCDF4.Dataset(nc_files[1], 'r').variables['elevation'][::2, ::2].data

import matplotlib.pyplot as plt
plt.imshow(file[::-1])
plt.colorbar()
plt.show()