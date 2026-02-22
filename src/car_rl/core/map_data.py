from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

from car_rl.core.types import CarState


@dataclass
class Segment:
    p1: tuple[float, float]
    p2: tuple[float, float]


@dataclass
class DirectedLine:
    p1: tuple[float, float]
    p2: tuple[float, float]
    forward: tuple[float, float]


@dataclass
class TrackMap:
    name: str
    walls: list[Segment]
    start_pose: CarState
    start_line: DirectedLine
    finish_line: DirectedLine
    centerline: list[tuple[float, float]]



def _read_point(values: Sequence[float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def _read_segment(values: Sequence[Sequence[float]]) -> Segment:
    return Segment(p1=_read_point(values[0]), p2=_read_point(values[1]))


def _read_directed_line(data: dict) -> DirectedLine:
    return DirectedLine(
        p1=_read_point(data["p1"]),
        p2=_read_point(data["p2"]),
        forward=_read_point(data["forward"]),
    )


def load_map(path: Union[str, Path]) -> TrackMap:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    start_pose_data = data["start_pose"]
    start_pose = CarState(
        x=float(start_pose_data["x"]),
        y=float(start_pose_data["y"]),
        yaw=float(start_pose_data["yaw"]),
        v=float(start_pose_data.get("v", 0.0)),
        delta=float(start_pose_data.get("delta", 0.0)),
    )

    walls = [_read_segment(seg) for seg in data["walls"]]
    return TrackMap(
        name=str(data["name"]),
        walls=walls,
        start_pose=start_pose,
        start_line=_read_directed_line(data["start_line"]),
        finish_line=_read_directed_line(data["finish_line"]),
        centerline=[_read_point(p) for p in data.get("centerline", [])],
    )
