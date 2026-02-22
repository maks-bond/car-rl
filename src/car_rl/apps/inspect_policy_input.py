from __future__ import annotations

import argparse

from car_rl.agents.constant import ConstantActionAgent
from car_rl.apps.common import create_env
from car_rl.env.features import PolicyInputAdapter
from car_rl.maps.registry import get_map_path, list_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect policy input vector shapes and values")
    parser.add_argument("--map", default="straight_corridor", choices=list_maps())
    parser.add_argument("--observation-mode", default="boundary", choices=["state", "boundary"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--delta-dot", type=float, default=0.0)
    args = parser.parse_args()

    env = create_env(get_map_path(args.map), observation_mode=args.observation_mode)
    adapter = PolicyInputAdapter(observation_mode=args.observation_mode)
    agent = ConstantActionAgent(a=args.a, delta_dot=args.delta_dot)

    obs = env.reset()
    vec = adapter.transform(obs)
    print(f"mode={args.observation_mode} dim={adapter.feature_dim}")
    print("features:", ", ".join(adapter.feature_names()))
    print("step=reset vec=", vec.tolist())

    for step in range(1, args.steps + 1):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        vec = adapter.transform(obs)
        print(f"step={step} reward={reward:.3f} done={done} event={info['event']} vec={vec.tolist()}")
        if done:
            break


if __name__ == "__main__":
    main()
