from __future__ import annotations

import math

from car_rl.core.types import Action, CarState, VehicleLimits, VehicleParams


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clip_action(action: Action, limits: VehicleLimits) -> Action:
    return Action(
        a=clip(action.a, limits.a_min, limits.a_max),
        delta_dot=clip(action.delta_dot, limits.delta_dot_min, limits.delta_dot_max),
    )


def step_bicycle(
    state: CarState,
    action: Action,
    params: VehicleParams,
    limits: VehicleLimits,
    dt: float,
) -> CarState:
    action = clip_action(action, limits)

    x_dot = state.v * math.cos(state.yaw)
    y_dot = state.v * math.sin(state.yaw)
    yaw_dot = 0.0 if abs(params.wheelbase) < 1e-9 else (state.v / params.wheelbase) * math.tan(state.delta)

    next_x = state.x + x_dot * dt
    next_y = state.y + y_dot * dt
    next_yaw = state.yaw + yaw_dot * dt
    next_v = clip(state.v + action.a * dt, limits.v_min, limits.v_max)
    next_delta = clip(state.delta + action.delta_dot * dt, limits.delta_min, limits.delta_max)

    return CarState(x=next_x, y=next_y, yaw=next_yaw, v=next_v, delta=next_delta)
