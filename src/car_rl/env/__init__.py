from car_rl.env.environment import CarEnv
from car_rl.env.features import PolicyInputAdapter

__all__ = ["CarEnv", "PolicyInputAdapter"]

try:
    from car_rl.env.gym_env import CarGymEnv, make_car_gym_env

    __all__.extend(["CarGymEnv", "make_car_gym_env"])
except ModuleNotFoundError:
    # gymnasium is optional at import time; install project deps to enable it.
    pass
