from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from car_rl.core.dynamics import step_bicycle
from car_rl.core.geometry import dot, point_in_convex_polygon, segment_intersects
from car_rl.core.map_data import DirectedLine, TrackMap
from car_rl.core.types import Action, CarState, RewardConfig, VehicleLimits, VehicleParams


@dataclass
class StepResult:
    state: CarState
    reward: float
    done: bool
    event: Optional[str]


class Simulator:
    def __init__(
        self,
        track_map: TrackMap,
        params: VehicleParams,
        limits: VehicleLimits,
        reward_config: RewardConfig,
        dt: float,
    ) -> None:
        self.track_map = track_map
        self.params = params
        self.limits = limits
        self.reward_config = reward_config
        self.dt = dt
        self.state = track_map.start_pose

    def reset(self) -> CarState:
        self.state = CarState(
            x=self.track_map.start_pose.x,
            y=self.track_map.start_pose.y,
            yaw=self.track_map.start_pose.yaw,
            v=self.track_map.start_pose.v,
            delta=self.track_map.start_pose.delta,
        )
        return self.state

    def step(self, action: Action) -> StepResult:
        prev = self.state
        nxt = step_bicycle(prev, action, self.params, self.limits, self.dt)

        reward = self.reward_config.step_penalty
        done = False
        event: Optional[str] = None

        if self._is_collision(nxt):
            reward += self.reward_config.collision
            done = True
            event = "collision"
        elif self._crossed_line(prev, nxt, self.track_map.finish_line):
            if self._is_forward_crossing(prev, nxt, self.track_map.finish_line):
                reward += self.reward_config.finish
                done = True
                event = "finish"
        elif self._crossed_line(prev, nxt, self.track_map.start_line):
            if not self._is_forward_crossing(prev, nxt, self.track_map.start_line):
                reward += self.reward_config.backward_start
                done = True
                event = "backward_start"

        self.state = nxt
        return StepResult(state=nxt, reward=reward, done=done, event=event)

    def _is_collision(self, state: CarState) -> bool:
        car_corners = self._car_corners(state)
        car_edges = (
            (car_corners[0], car_corners[1]),
            (car_corners[1], car_corners[2]),
            (car_corners[2], car_corners[3]),
            (car_corners[3], car_corners[0]),
        )

        for wall in self.track_map.walls:
            for e1, e2 in car_edges:
                if segment_intersects(e1, e2, wall.p1, wall.p2):
                    return True

            # If a wall endpoint lands inside the car body, treat as collision.
            if point_in_convex_polygon(wall.p1, car_corners) or point_in_convex_polygon(wall.p2, car_corners):
                return True
        return False

    def _crossed_line(self, prev: CarState, nxt: CarState, line: DirectedLine) -> bool:
        return segment_intersects((prev.x, prev.y), (nxt.x, nxt.y), line.p1, line.p2)

    def _is_forward_crossing(self, prev: CarState, nxt: CarState, line: DirectedLine) -> bool:
        movement = (nxt.x - prev.x, nxt.y - prev.y)
        return dot(movement, line.forward) > 0.0

    def _car_corners(self, state: CarState) -> tuple[tuple[float, float], ...]:
        x_rear = -self.params.rear_overhang
        x_front = self.params.wheelbase + self.params.front_overhang
        half_w = 0.5 * self.params.width

        return (
            self._local_to_world(state, x_rear, -half_w),
            self._local_to_world(state, x_front, -half_w),
            self._local_to_world(state, x_front, half_w),
            self._local_to_world(state, x_rear, half_w),
        )

    @staticmethod
    def _local_to_world(state: CarState, lx: float, ly: float) -> tuple[float, float]:
        c = math.cos(state.yaw)
        s = math.sin(state.yaw)
        return (
            state.x + lx * c - ly * s,
            state.y + lx * s + ly * c,
        )
