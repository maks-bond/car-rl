from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarState:
    x: float
    y: float
    yaw: float
    v: float
    delta: float


@dataclass
class Action:
    a: float
    delta_dot: float


@dataclass
class VehicleLimits:
    a_min: float
    a_max: float
    v_min: float
    v_max: float
    delta_min: float
    delta_max: float
    delta_dot_min: float
    delta_dot_max: float


@dataclass
class VehicleParams:
    wheelbase: float
    front_overhang: float = 1.0
    rear_overhang: float = 0.95
    width: float = 1.65


@dataclass
class RewardConfig:
    finish: float
    collision: float
    backward_start: float
    step_penalty: float
