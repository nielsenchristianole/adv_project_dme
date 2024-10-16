import json
import copy
from typing import Literal, Dict, List

import cv2
import numpy as np

from src.gradio.demo_types import TOWN_TYPE, TOWN_TYPES
from src.gradio.gradio_configs import TOWN_NAMES_PATH, ICON_DIR


# load in town names and assert the structure
with open(TOWN_NAMES_PATH, 'r') as f:
    TOWN_NAMES: Dict[TOWN_TYPE, Dict[Literal['coastal', 'landlocked'], List[str]]] = json.load(f)
    _all_n = set()
    for _tt, _d2 in TOWN_NAMES.items():
        assert _tt in TOWN_TYPES
        for _lt, _n in _d2.items():
            assert _lt in ['coastal', 'landlocked']
            assert isinstance(_n, list)
            _n = set(_n)
            assert _all_n.isdisjoint(_n)
            _all_n.update(_n)


class TownNameSampler():
    """
    Class to sample town names from a list of town names, and generate new names if the list is empty and keep track of used names
    """

    def __init__(self):
        self.town_names = copy.deepcopy(TOWN_NAMES)
        self.overused_idx = 0

    def pop(self, town_type: TOWN_TYPE, is_coastal: bool) -> str:
        """
        Sample a town name from the list of town names and remove it from the list
        """
        possible_names = self.town_names[town_type]['coastal' if is_coastal else 'landlocked']
        if possible_names:
            idx = np.random.randint(len(possible_names))
            return possible_names.pop(idx)
        
        name = f'{town_type.capitalize()}_{self.overused_idx}'
        self.overused_idx += 1
        return 'Costal_' + name if is_coastal else name
    
    def reset(self):
        """
        Reset the town names to the original list
        """
        self.town_names = copy.deepcopy(TOWN_NAMES)
        self.overused_idx = 0



class GetIcon:
    """
    Class to get the icon for a town
    """

    def __init__(self):
        
        name2mask = dict()
        for town_type in TOWN_TYPES:
            path = ICON_DIR / f'{town_type}.png'
            if not path.exists():
                raise FileNotFoundError(f'Icon for town type {town_type} not found at {path}')
            im = cv2.imread(str(path), flags=cv2.IMREAD_UNCHANGED)
            name2mask[town_type] = (im[..., 3] != 0).astype(np.uint8)
        self.name2mask: Dict[TOWN_TYPE, np.ndarray] = name2mask
    
    def get(self, town_type: TOWN_TYPE, icon_size: int, color: np.ndarray) -> np.ndarray:

        if not isinstance(color, np.ndarray):
            color = np.array(color)
        if color.shape == (3,):
            color = np.concatenate([color, [255]], axis=0).astype(np.uint8)

        mask = self.name2mask[town_type]
        mask = cv2.resize(mask, (icon_size, icon_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        out_im = np.empty((icon_size, icon_size, 4), dtype=np.uint8)
        out_im[mask] = color[None,:]
        out_im[~mask] = 0
        return out_im
