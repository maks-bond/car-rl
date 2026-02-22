from __future__ import annotations

import math
from typing import Sequence

from car_rl.agents.base import Agent
from car_rl.core.types import Action


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class EngineeredLaneFollowerAgent(Agent):
    def __init__(
        self,
        centerline: Sequence[tuple[float, float]],
        lookahead_distance: float = 1.8,
        target_speed: float = 2.2,
        max_steer: float = 0.6,
        max_delta_dot: float = 1.2,
    ) -> None:
        if len(centerline) < 2:
            raise ValueError("centerline must contain at least 2 points")

        self.centerline = list(centerline)
        self.lookahead_distance = lookahead_distance
        self.target_speed = target_speed
        self.max_steer = max_steer
        self.max_delta_dot = max_delta_dot
        self._cursor = 0

    def act(self, obs: dict) -> Action:
        x = float(obs["x"])
        y = float(obs["y"])
        yaw = float(obs["yaw"])
        v = float(obs["v"])
        delta = float(obs["delta"])

        self._cursor = self._nearest_index((x, y), self._cursor)
        target = self._lookahead_target((x, y), self._cursor)

        desired_heading = math.atan2(target[1] - y, target[0] - x)
        heading_error = _wrap_angle(desired_heading - yaw)

        desired_delta = _clamp(2.2 * heading_error, -self.max_steer, self.max_steer)
        delta_dot = _clamp(4.0 * (desired_delta - delta), -self.max_delta_dot, self.max_delta_dot)

        turn_factor = min(abs(heading_error) / 0.8, 1.0)
        speed_target = max(0.9, self.target_speed * (1.0 - 0.75 * turn_factor))
        a = _clamp(1.2 * (speed_target - v), -2.5, 1.6)

        return Action(a=a, delta_dot=delta_dot)

    def _nearest_index(self, p: tuple[float, float], start_idx: int) -> int:
        best_idx = start_idx
        best_dist = float("inf")
        upper = min(len(self.centerline), start_idx + 10)
        for i in range(start_idx, upper):
            d = _distance(p, self.centerline[i])
            if d < best_dist:
                best_dist = d
                best_idx = i

        # Allow recovery if car deviates far from the planned corridor.
        if best_dist > 4.0:
            for i, wp in enumerate(self.centerline):
                d = _distance(p, wp)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        return best_idx

    def _lookahead_target(self, p: tuple[float, float], idx: int) -> tuple[float, float]:
        traveled = _distance(p, self.centerline[idx])
        i = idx
        while i + 1 < len(self.centerline):
            segment = _distance(self.centerline[i], self.centerline[i + 1])
            if traveled + segment >= self.lookahead_distance:
                return self.centerline[i + 1]
            traveled += segment
            i += 1
        return self.centerline[-1]
