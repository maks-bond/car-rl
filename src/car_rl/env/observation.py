from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from car_rl.core.geometry import ray_segment_intersection_distance
from car_rl.core.map_data import TrackMap
from car_rl.core.types import CarState, VehicleLimits


@dataclass
class BoundaryObservationConfig:
    num_rays: int = 15
    fov_deg: float = 180.0
    max_ray_distance: float = 20.0


class BoundaryObservationBuilder:
    def __init__(
        self,
        track_map: TrackMap,
        limits: VehicleLimits,
        config: BoundaryObservationConfig,
    ) -> None:
        if config.num_rays < 2:
            raise ValueError("num_rays must be >= 2")
        self.track_map = track_map
        self.limits = limits
        self.config = config

        fov = math.radians(config.fov_deg)
        if config.num_rays == 1:
            self._ray_angles = [0.0]
        else:
            step = fov / (config.num_rays - 1)
            start = -0.5 * fov
            self._ray_angles = [start + i * step for i in range(config.num_rays)]

    def observe(self, state: CarState) -> Dict[str, object]:
        origin = (state.x, state.y)
        ray_distances: List[float] = []
        ray_distances_norm: List[float] = []

        for rel_angle in self._ray_angles:
            ray_angle = state.yaw + rel_angle
            direction = (math.cos(ray_angle), math.sin(ray_angle))
            d = self._cast_ray(origin, direction)
            ray_distances.append(d)
            ray_distances_norm.append(d / self.config.max_ray_distance)

        v_scale = max(abs(self.limits.v_min), abs(self.limits.v_max), 1e-6)
        delta_scale = max(abs(self.limits.delta_min), abs(self.limits.delta_max), 1e-6)

        return {
            "v_norm": state.v / v_scale,
            "delta_norm": state.delta / delta_scale,
            "ray_distances": ray_distances,
            "ray_distances_norm": ray_distances_norm,
        }

    def _cast_ray(self, origin: tuple[float, float], direction: tuple[float, float]) -> float:
        best = self.config.max_ray_distance
        for wall in self.track_map.walls:
            d = ray_segment_intersection_distance(origin, direction, wall.p1, wall.p2)
            if d is not None and d < best:
                best = d
        return best
