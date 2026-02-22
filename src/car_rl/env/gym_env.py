from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from car_rl.apps.common import create_env
from car_rl.core.types import Action
from car_rl.env.features import PolicyInputAdapter


class CarGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, map_path: str, observation_mode: str = "boundary") -> None:
        super().__init__()
        self._env = create_env(map_path, observation_mode=observation_mode)
        self._adapter = PolicyInputAdapter(observation_mode=observation_mode)
        self._observation_mode = observation_mode

        obs0 = self._env.reset()
        vec0 = self._adapter.transform(obs0)

        limits = self._env.simulator.limits
        self.action_space = spaces.Box(
            low=np.asarray([limits.a_min, limits.delta_dot_min], dtype=np.float32),
            high=np.asarray([limits.a_max, limits.delta_dot_max], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=vec0.shape,
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        obs = self._env.reset()
        vec = self._adapter.transform(obs)
        info = {
            "observation_mode": self._observation_mode,
            "observation_dict": obs,
            "state": asdict(self._env.simulator.state),
        }
        return vec, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, reward, done, info = self._env.step(Action(a=float(action[0]), delta_dot=float(action[1])))
        vec = self._adapter.transform(obs)

        timed_out = bool(info.get("timed_out", False))
        terminated = bool(done and not timed_out)
        truncated = bool(timed_out)

        out_info = dict(info)
        out_info["observation_mode"] = self._observation_mode
        out_info["observation_dict"] = obs
        return vec, float(reward), terminated, truncated, out_info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def make_car_gym_env(map_path: str, observation_mode: str = "boundary") -> CarGymEnv:
    return CarGymEnv(map_path=map_path, observation_mode=observation_mode)
