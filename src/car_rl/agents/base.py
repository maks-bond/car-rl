from __future__ import annotations

from abc import ABC, abstractmethod

from car_rl.core.types import Action


class Agent(ABC):
    @abstractmethod
    def act(self, obs: dict) -> Action:
        raise NotImplementedError
