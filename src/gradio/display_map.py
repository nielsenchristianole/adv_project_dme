from typing import Literal, Optional, List, Tuple
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
    poly_shape = 1
    height_map = 2
    towns = 3
    roads = 4
    change_poly_shape = 5


def plot_map(
    shape: Optional[np.ndarray]=None, # Binary image of the shape of the map
    height_map: Optional[np.ndarray]=None,
    towns: Optional[List[Town]]=None,
    roads: Optional[RoadGraph]=None,
    return_step: Literal['empty', 'shape', 'height_map', 'towns', 'roads', 'change_poly_shape']='roads',
    *,
    polygon_dict: Optional[dict]=None,
    resolution: Optional[int]=None,
    max_height: Optional[float]=None,
    num_contour_levels: int=4,
    contour_pixel_width: int=3,
    contour_color: Tuple[int,int,int]=(170,170,170),
    min_contour_length: int=10,
    poly_shape_pixel_width: int=10,
    poly_shape_color: Tuple[int,int,int]=(0,0,0),
    contour_is_probably_not_closed_threshold: Optional[float]=None,
    towns_pixel_width: Optional[int]=None,
):
    """
    Get an array which is the map of the town
    """
    
    poly_shape = list(np.load('assets/defaults/shape.npz').values())
    
    return_step = RETURN_STEP_OPTIONS[return_step]

    # fix resolution param
    if resolution is None:
        if height_map is not None:
            resolution = max(height_map.shape[:2])
        elif poly_shape is not None:
            resolution = poly_shape.bounds[2:] if isinstance(poly_shape, shapely.MultiPolygon) else np.ceil(np.array([np.max(s, axis=0).squeeze(0) for s in poly_shape]).max(axis=0)).astype(int)[::-1]
            map_so_far = np.full((*resolution, 3), 255, dtype=np.uint8)
            resolution = max(resolution).item()
        else:
            resolution = 2000

    # --------------------------------- Empty map -------------------------------- #
    if return_step == RETURN_STEP_OPTIONS.empty:
        return np.ones((resolution, resolution, 1), dtype=np.uint8) * WATER_COLOR[None, None, :]

    # define various param defaults
    if contour_is_probably_not_closed_threshold is None:
        contour_is_probably_not_closed_threshold = resolution * 0.1
    if towns_pixel_width is None:
        towns_pixel_width = resolution // 20
        
    # --------------------------- Change Shape by Poly --------------------------- #
    if return_step == RETURN_STEP_OPTIONS.change_poly_shape and polygon_dict:
        out_resolution = (
            int(resolution * shape.shape[0] / max(shape.shape[:2])),
            int(resolution * shape.shape[1] / max(shape.shape[:2])))
        shape = cv2.resize(shape, out_resolution[::-1], interpolation=cv2.INTER_NEAREST)
        sea_mask = shape == 0
        
        map_so_far = np.empty((*shape.shape, 3), dtype=np.uint8)
        map_so_far[sea_mask] = WATER_COLOR
        map_so_far[~sea_mask] = TERRAIN_COLORMAP[0]
        
        # Draw User Polygon
        poly_points = polygon_dict['poly_points']
        poly_closed = polygon_dict['poly_closed']

        if poly_points: # if there are points
            # Draw lines between points
            for i in range(len(poly_points) - 1):
                start_point = (int(poly_points[i][0]), int(poly_points[i][1]))
                end_point = (int(poly_points[i + 1][0]), int(poly_points[i + 1][1]))
                cv2.line(map_so_far, start_point, end_point, color=(0, 255, 0), thickness=2)
                
            if poly_closed:
                # Draw a line between the last point and the first point to close the polygon
                if len(poly_points) > 1:
                    start_point = (int(poly_points[-1][0]), int(poly_points[-1][1]))
                    end_point = (int(poly_points[0][0]), int(poly_points[0][1]))
                    cv2.line(map_so_far, start_point, end_point, color=(0, 255, 0), thickness=2)
                # Draw a transparent polygon
                poly_points_np = np.array(poly_points, dtype=np.int32)
                overlay = map_so_far.copy()
                cv2.fillPoly(overlay, [poly_points_np], color=(0, 255, 0, 128))  # RGBA color with transparency
                map_so_far = cv2.addWeighted(overlay, 0.5, map_so_far, 0.5, 0)
            else:
                # Draw points
                for point in poly_points:
                    cv2.circle(map_so_far, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 0), thickness=-1)
        
        return map_so_far

    # -------------------------------- Height map -------------------------------- #
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
            colormapped = cv2.applyColorMap((simple_height_map * (255 / max_height)).astype(np.uint8), TERRAIN_COLORMAP[:, i])
            map_color[..., i] = colormapped
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


    # ------------------------ Contours and polyline shape ----------------------- #
    if isinstance(poly_shape, shapely.MultiPolygon):
        poly_shape_line = list()
        for pol in poly_shape:
            poly_shape_line.append(list(pol.exterior.coords))
            for interior in pol.interiors:
                poly_shape_line.append(list(interior.coords))
        poly_shape_line = [np.array(s, dtype=np.int32)[:,None,::-1] for s in poly_shape_line]
    if (poly_shape is None) and (return_step >= RETURN_STEP_OPTIONS.height_map):
        poly_shape_line = measure.find_contours(simple_height_map_non_clipped, level=0)
        poly_shape_line = [s[:,None,::-1].astype(np.int32) for s in poly_shape_line if s.shape[0] > 3]
        poly_shape_line = [np.concatenate((s,s[::-1]), axis=0) if np.linalg.norm(s[0] - s[-1]) > contour_is_probably_not_closed_threshold else s for s in poly_shape_line]
        # np.savez('assets/defaults/poly_shape.npz', *poly_shape_line)
    elif isinstance(poly_shape, list):
        poly_shape_line = poly_shape
    else:
        raise ValueError()

    # draw the poly_shape on the map
    map_so_far = cv2.drawContours(map_so_far, poly_shape_line, -1, poly_shape_color, poly_shape_pixel_width)

    # ----------------------------------- Towns ---------------------------------- #
    if return_step >= RETURN_STEP_OPTIONS.towns:
        # draw the towns on the map
        map_so_far = Image.fromarray(map_so_far.copy(), mode='RGB').convert('RGBA')
        for town in towns:
            icon = ICON_GETTER.get(town['town_type'], towns_pixel_width, COASTAL_TOWN_COLOR if town['is_coastal'] else NON_COASTAL_TOWN_COLOR)
            icon = Image.fromarray(icon, mode='RGBA')
            location = (int(town['xyz'][0]) - icon.size[0] // 2, int(town['xyz'][1]) - icon.size[1] // 2)
            map_so_far.paste(icon, location, mask=icon.split()[3])

        map_so_far = np.array(map_so_far.convert('RGB'))

    # ----------------------------------- Roads ---------------------------------- #
    if return_step >= RETURN_STEP_OPTIONS.roads:
        pass

    return map_so_far


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    height_map = np.load('assets/defaults/height_map.npy')
    poly_shape = list(np.load('assets/defaults/poly_shape.npz').values())
    full_chart = plot_map(height_map=None, poly_shape=poly_shape, return_step='poly_shape')

    plt.imshow(full_chart)
    plt.axis('off')
    plt.show()
