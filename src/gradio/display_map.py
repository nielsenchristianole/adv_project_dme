from typing import Literal
from enum import IntEnum

import cv2
import numpy as np
import shapely
import matplotlib as mpl
from skimage import measure
from PIL import Image

from src.gradio.demo_types import Town, RoadGraph, TOWN_TYPES
from src.gradio.gradio_utils import GetIcon, TownNameSampler


# what color to use for water
WATER_COLOR = (255 * np.array(mpl.colormaps['terrain'](0.130)[:3])).astype(np.uint8)

# what color to use for land
LAND_COLOR_THRESH = 0.226
TERRAIN_COLORMAP = np.array([mpl.colormaps['terrain'](i)[:3] for i in np.linspace(LAND_COLOR_THRESH, 1, 256)])
TERRAIN_COLORMAP = (255 * TERRAIN_COLORMAP).astype(np.uint8)

COASTAL_TOWN_COLOR = np.array([0, 0, 255])
NON_COASTAL_TOWN_COLOR = np.array([255, 255, 255])

ICON_GETTER = GetIcon()
class RETURN_STEP_OPTIONS(IntEnum):
    empty = 0
    shape = 1
    height_map = 2
    towns = 3
    roads = 4


def plot_map(
    shape: shapely.MultiPolygon|None=None,
    height_map: np.ndarray|None=None,
    towns: list[Town]|None=None,
    roads: RoadGraph|None=None,
    return_step: Literal['empty', 'shape', 'height_map', 'towns', 'roads']='roads',
    *,
    resolution: int|None=None,
    max_height: float|None=None,
    num_contour_levels: int=4,
    contour_pixel_width: int=3,
    contour_color: tuple[int,int,int]=(170,170,170),
    min_contour_length: int=10,
    shape_pixel_width: int=10,
    shape_color: tuple[int,int,int]=(0,0,0),
    contour_is_probably_not_closed_threshold: float|None=None,
    towns_pixel_width: int|None=None,
):
    """
    Get an array which is the map of the town
    """
    return_step = RETURN_STEP_OPTIONS[return_step]

    # fix resolution param
    if resolution is None:
        if height_map is not None:
            resolution = max(height_map.shape[:2])
        elif shape is not None:
            resolution = shape.bounds[2:] if isinstance(shape, shapely.MultiPolygon) else np.ceil(np.array([np.max(s, axis=0).squeeze(0) for s in shape]).max()).astype(int).item()
        else:
            resolution = 2000

    # return empty map
    if return_step == RETURN_STEP_OPTIONS.empty:
        return np.ones((resolution, resolution, 1), dtype=np.uint8) * WATER_COLOR[None, None, :]

    # define various param defaults
    if contour_is_probably_not_closed_threshold is None:
        contour_is_probably_not_closed_threshold = resolution * 0.1
    if towns_pixel_width is None:
        towns_pixel_width = resolution // 20


    # only generate the height map if given by return step
    map_so_far = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    if return_step >= RETURN_STEP_OPTIONS.height_map:

        if max_height is None:
            max_height = height_map.max()

        # height map for use
        simple_height_map_non_clipped = height_map.copy().astype(np.float32)
        out_resolution = (
            int(resolution * height_map.shape[0] / max(height_map.shape[:2])),
            int(resolution * height_map.shape[1] / max(height_map.shape[:2])))
        simple_height_map_non_clipped = cv2.resize(simple_height_map_non_clipped, out_resolution[::-1], interpolation=cv2.INTER_LINEAR)
        simple_height_map = np.clip(simple_height_map_non_clipped, a_min=0, a_max=None)

        # where to draw sea
        sea_mask = simple_height_map <= 0

        # color the map
        map_color = np.empty((*out_resolution, 3), dtype=np.uint8)
        for i in range(3):
            map_color[..., i] = cv2.applyColorMap((simple_height_map * (255 / max_height)).astype(np.uint8), TERRAIN_COLORMAP[:, i])
        map_color[sea_mask] = WATER_COLOR

        # draw contours on the map
        map_so_far = map_color.copy()
        # contour_levels = np.linspace(0, simple_height_map.max(), num_contours+2)[1:-1]
        contour_levels = np.quantile(simple_height_map[~sea_mask], np.linspace(0, 1, num_contour_levels+2)[1:-1])
        contours = list()
        for level in contour_levels:
            _conts = measure.find_contours(simple_height_map, level=level)
            _conts = [c[:,None,::-1].astype(np.int32) for c in _conts if c.shape[0] > min_contour_length] # filter contours and correct format
            _conts = [np.concatenate((c,c[::-1]), axis=0) if np.linalg.norm(c[0] - c[-1]) > contour_is_probably_not_closed_threshold else c for c in _conts] # reverse contours which arent closed
            contours.extend(_conts)
        map_so_far = cv2.drawContours(map_so_far , contours, -1, contour_color, contour_pixel_width)


    # get the shape in correct format
    if isinstance(shape, shapely.MultiPolygon):
        shape_line = list()
        for pol in shape:
            shape_line.append(list(pol.exterior.coords))
            for interior in pol.interiors:
                shape_line.append(list(interior.coords))
        shape_line = [np.array(s, dtype=np.int32)[:,None,::-1] for s in shape_line]
    if (shape is None) and (return_step >= RETURN_STEP_OPTIONS.height_map):
        shape_line = measure.find_contours(simple_height_map_non_clipped, level=0)
        shape_line = [s[:,None,::-1].astype(np.int32) for s in shape_line if s.shape[0] > 3]
        shape_line = [np.concatenate((s,s[::-1]), axis=0) if np.linalg.norm(s[0] - s[-1]) > contour_is_probably_not_closed_threshold else s for s in shape_line]
        # np.savez('assets/defaults/shape.npz', *shape_line)
    elif isinstance(shape, list):
        shape_line = shape
    else:
        raise ValueError()

    # draw the shape on the map
    map_so_far = cv2.drawContours(map_so_far, shape_line, -1, shape_color, shape_pixel_width)

    # generate some random towns if none is given
    if return_step >= RETURN_STEP_OPTIONS.towns:

        # draw the towns on the map
        map_so_far = Image.fromarray(map_so_far.copy(), mode='RGB').convert('RGBA')
        for town in towns:
            icon = ICON_GETTER.get(town['town_type'], towns_pixel_width, COASTAL_TOWN_COLOR if town['is_coastal'] else NON_COASTAL_TOWN_COLOR)
            icon = Image.fromarray(icon, mode='RGBA')
            location = (int(town['xyz'][0]) - icon.size[0] // 2, int(town['xyz'][1]) - icon.size[1] // 2)
            map_so_far.paste(icon, location, mask=icon.split()[3])

        map_so_far = np.array(map_so_far.convert('RGB'))

    # draw the roads
    if return_step >= RETURN_STEP_OPTIONS.roads:
        pass

    return map_so_far


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    height_map = np.load('assets/defaults/height_map.npy')
    shape = list(np.load('assets/defaults/shape.npz').values())
    full_chart = plot_map(shape, None, None, None, return_step='shape')

    plt.imshow(full_chart)
    plt.axis('off')
    plt.show()
