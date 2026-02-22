from __future__ import annotations

import argparse

import numpy as np

from car_rl.maps.registry import get_map_path, list_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Gymnasium wrapper")
    parser.add_argument("--map", default="straight_corridor", choices=list_maps())
    parser.add_argument("--observation-mode", default="boundary", choices=["state", "boundary"])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    try:
        from car_rl.env.gym_env import make_car_gym_env
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gymnasium is not installed in this environment. Run `pip install -e .` to install project dependencies."
        ) from exc

    env = make_car_gym_env(str(get_map_path(args.map)), observation_mode=args.observation_mode)
    obs, info = env.reset()

    print(f"obs_shape={obs.shape} action_shape={env.action_space.shape} mode={args.observation_mode}")
    print(f"obs_sample={obs[:min(8, obs.shape[0])].tolist()}")
    print(f"info_keys={sorted(info.keys())}")

    total_reward = 0.0
    for step in range(args.steps):
        action = np.asarray([1.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"step={step} reward={reward:.3f} terminated={terminated} truncated={truncated} "
            f"event={info.get('event')}"
        )
        if terminated or truncated:
            break

    print(f"total_reward={total_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
