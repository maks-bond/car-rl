from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from car_rl.env.gym_env import make_car_gym_env
from car_rl.maps.registry import get_map_path, list_maps


def evaluate(model, env, episodes: int, deterministic: bool) -> Dict[str, float]:
    returns = []
    lengths = []
    success = 0
    collisions = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        event = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            done = bool(terminated or truncated)
            event = info.get("event")

        returns.append(ep_return)
        lengths.append(ep_len)
        if event == "finish":
            success += 1
        if event == "collision":
            collisions += 1
        print(f"episode={ep} return={ep_return:.3f} len={ep_len} event={event}")

    n = max(1, episodes)
    return {
        "mean_return": float(sum(returns) / n),
        "mean_length": float(sum(lengths) / n),
        "success_rate": float(success / n),
        "collision_rate": float(collisions / n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SB3 PPO policy")
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument("--map", default="straight_corridor", choices=list_maps())
    parser.add_argument("--observation-mode", default="boundary", choices=["state", "boundary"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save-json", default="")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is not installed. Run `pip install -e .` in your venv first."
        ) from exc

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    env = make_car_gym_env(str(get_map_path(args.map)), observation_mode=args.observation_mode)
    model = PPO.load(str(model_path))

    metrics = evaluate(model, env, episodes=args.episodes, deterministic=not args.stochastic)
    print("summary:", metrics)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved metrics to {out_path}")


if __name__ == "__main__":
    main()
