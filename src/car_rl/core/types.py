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
    radius: float


@dataclass
class RewardConfig:
    finish: float
    collision: float
    backward_start: float
    step_penalty: float
