from __future__ import annotations

from pathlib import Path
from typing import Union

from car_rl.core.map_data import load_map
from car_rl.core.simulator import Simulator
from car_rl.core.types import RewardConfig, VehicleLimits, VehicleParams
from car_rl.env.environment import CarEnv


DEFAULT_DT = 0.05
DEFAULT_MAX_STEPS = 1000


def create_env(map_path: Union[str, Path]) -> CarEnv:
    track_map = load_map(map_path)

    params = VehicleParams(
        wheelbase=2.7,
        radius=0.35,
    )
    limits = VehicleLimits(
        a_min=-3.0,
        a_max=2.0,
        v_min=-2.0,
        v_max=8.0,
        delta_min=-0.6,
        delta_max=0.6,
        delta_dot_min=-1.2,
        delta_dot_max=1.2,
    )
    rewards = RewardConfig(
        finish=100.0,
        collision=-100.0,
        backward_start=-100.0,
        step_penalty=-0.01,
    )

    simulator = Simulator(
        track_map=track_map,
        params=params,
        limits=limits,
        reward_config=rewards,
        dt=DEFAULT_DT,
    )
    return CarEnv(simulator=simulator, max_steps=DEFAULT_MAX_STEPS)
