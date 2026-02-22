from __future__ import annotations

from car_rl.agents.base import Agent
from car_rl.core.types import Action


class ConstantActionAgent(Agent):
    def __init__(self, a: float, delta_dot: float) -> None:
        self._action = Action(a=a, delta_dot=delta_dot)

    def act(self, obs: dict) -> Action:
        _ = obs
        return self._action
