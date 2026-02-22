from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from car_rl.core.simulator import Simulator, StepResult
from car_rl.core.types import Action, CarState
from car_rl.env.observation import BoundaryObservationBuilder


class CarEnv:
    def __init__(
        self,
        simulator: Simulator,
        max_steps: int = 1000,
        boundary_observer: Optional[BoundaryObservationBuilder] = None,
    ) -> None:
        self.simulator = simulator
        self.max_steps = max_steps
        self.boundary_observer = boundary_observer
        self.step_count = 0

    def reset(self) -> dict:
        self.step_count = 0
        state = self.simulator.reset()
        return self._obs(state)

    def step(self, action: Action) -> tuple[dict, float, bool, dict]:
        self.step_count += 1
        res: StepResult = self.simulator.step(action)
        timed_out = self.step_count >= self.max_steps
        done = res.done or timed_out
        info = {"event": res.event, "timed_out": timed_out, "state": asdict(res.state)}
        return self._obs(res.state), res.reward, done, info

    def _obs(self, state: CarState) -> dict:
        if self.boundary_observer is not None:
            return self.boundary_observer.observe(state)
        return asdict(state)
