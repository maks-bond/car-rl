from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from car_rl.core.dynamics import step_bicycle
from car_rl.core.geometry import dot, point_to_segment_distance, segment_intersects
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
        p = (state.x, state.y)
        for wall in self.track_map.walls:
            if point_to_segment_distance(p, wall.p1, wall.p2) <= self.params.radius:
                return True
        return False

    def _crossed_line(self, prev: CarState, nxt: CarState, line: DirectedLine) -> bool:
        return segment_intersects((prev.x, prev.y), (nxt.x, nxt.y), line.p1, line.p2)

    def _is_forward_crossing(self, prev: CarState, nxt: CarState, line: DirectedLine) -> bool:
        movement = (nxt.x - prev.x, nxt.y - prev.y)
        return dot(movement, line.forward) > 0.0
