from typing import Literal, TypedDict

import shapely

from src.gradio.gradio_configs import TOWN_TYPE, TOWN_TYPES



class Town(TypedDict):
    town_type: TOWN_TYPE
    is_coastal: bool
    xyz: tuple[float, float, float] # pixel space
    town_name: str


class RoadNode(TypedDict):
    is_city: bool
    xyz: tuple[float, float, float] # in pixels


class RoadEdge(TypedDict):
    line: shapely.LineString
    connected_nodes: tuple[str, str]


class RoadGraph(TypedDict):
    nodes: dict[str, RoadNode]
    edges: dict[str, RoadEdge]

    @classmethod
    def empty(cls) -> 'RoadGraph':
        this = cls(nodes=dict(), edges=dict())
        return this
