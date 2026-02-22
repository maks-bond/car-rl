from __future__ import annotations

from dataclasses import asdict

from car_rl.core.simulator import Simulator, StepResult
from car_rl.core.types import Action, CarState


class CarEnv:
    def __init__(self, simulator: Simulator, max_steps: int = 1000) -> None:
        self.simulator = simulator
        self.max_steps = max_steps
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
        info = {"event": res.event, "timed_out": timed_out}
        return self._obs(res.state), res.reward, done, info

    @staticmethod
    def _obs(state: CarState) -> dict:
        return asdict(state)
