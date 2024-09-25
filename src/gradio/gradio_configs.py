from typing import Literal
from pathlib import Path


TOWN_TYPE = Literal['hamlet', 'village', 'town', 'city']
TOWN_TYPES: list[TOWN_TYPE] = ['hamlet', 'village', 'town', 'city']

TOWN_NAMES_PATH = Path('assets/town_names.json')
ICON_DIR = Path('assets/icons')
CLOSE_ICON = ICON_DIR / 'close.png'
