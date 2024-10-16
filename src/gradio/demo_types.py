from typing import Literal, TypedDict, Tuple, Dict

import shapely

from src.gradio.gradio_configs import TOWN_TYPE, TOWN_TYPES



class Town(TypedDict):
    town_type: TOWN_TYPE
    is_coastal: bool
    xyz: Tuple[float, float, float] # pixel space
    town_name: str


class RoadNode(TypedDict):
    is_city: bool
    xyz: Tuple[float, float, float] # in pixels


class RoadEdge(TypedDict):
    line: shapely.LineString
    connected_nodes: Tuple[str, str]


class RoadGraph(TypedDict):
    nodes: Dict[str, RoadNode]
    edges: Dict[str, RoadEdge]

    @classmethod
    def empty(cls) -> 'RoadGraph':
        this = cls(nodes=dict(), edges=dict())
        return this
