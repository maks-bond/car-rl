from __future__ import annotations

"""
PPO trainer for this project.

Important: PPO optimization itself is NOT implemented in this file; this file orchestrates
training/evaluation/checkpointing around Stable-Baselines3's PPO implementation.

Primary references:
- PPO paper (Schulman et al., 2017): https://arxiv.org/abs/1707.06347
- SB3 PPO docs: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- SB3 PPO source: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
- Spinning Up PPO guide: https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from car_rl.env.gym_env import make_car_gym_env
from car_rl.maps.registry import get_map_path, list_maps


def evaluate_model(model, env, episodes: int, deterministic: bool) -> Dict[str, float]:
    # Deterministic policy evaluation to track learning progress and regressions.
    returns = []
    lengths = []
    success = 0
    collisions = 0

    for _ in range(episodes):
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

    n = max(1, episodes)
    return {
        "mean_return": float(sum(returns) / n),
        "mean_length": float(sum(lengths) / n),
        "success_rate": float(success / n),
        "collision_rate": float(collisions / n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CarGymEnv")
    parser.add_argument("--map", default="straight_corridor", choices=list_maps())
    parser.add_argument("--observation-mode", default="boundary", choices=["state", "boundary"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", default="auto", help="SB3/PyTorch device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--eval-stochastic",
        action="store_true",
        help="Use stochastic policy sampling during eval (default is deterministic).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--tensorboard-log", default="runs/tb")
    args = parser.parse_args()

    try:
        # PPO algorithm class, callbacks, and vectorized env wrappers are provided by SB3.
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is not installed. Run `pip install -e .` in your venv first."
        ) from exc

    map_path = str(get_map_path(args.map))

    run_name = f"sb3ppo_{args.map}_{args.observation_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    def make_train_env():
        # Monitor records episode-level stats consumed by SB3 logger.
        return Monitor(make_car_gym_env(map_path=map_path, observation_mode=args.observation_mode))

    # SB3 PPO expects a VecEnv. DummyVecEnv with one env keeps setup simple and deterministic.
    train_env = DummyVecEnv([make_train_env for _ in range(max(1, args.n_envs))])
    # Separate eval env so callback evaluation does not disturb training rollout state.
    eval_env = make_car_gym_env(map_path=map_path, observation_mode=args.observation_mode)

    class EvalMetricsCallback(BaseCallback):
        # Project-specific callback for periodic evaluation and model selection.
        def __init__(
            self,
            eval_env,
            eval_freq: int,
            eval_episodes: int,
            out_dir: Path,
            verbose: int = 1,
        ) -> None:
            super().__init__(verbose=verbose)
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.eval_episodes = eval_episodes
            self.out_dir = out_dir
            self.best_success = -1.0
            self.best_return = float("-inf")
            self.history: list[dict[str, Any]] = []
            self.last_eval_timestep = 0

        def _on_step(self) -> bool:
            if self.eval_freq <= 0:
                return True
            if (self.num_timesteps - self.last_eval_timestep) < self.eval_freq:
                return True
            self.last_eval_timestep = int(self.num_timesteps)

            metrics = evaluate_model(
                self.model,
                self.eval_env,
                self.eval_episodes,
                deterministic=not args.eval_stochastic,
            )
            row = {
                "timesteps": int(self.num_timesteps),
                **metrics,
            }
            self.history.append(row)

            if self.verbose:
                print(
                    "[eval] "
                    f"t={row['timesteps']} return={row['mean_return']:.3f} "
                    f"len={row['mean_length']:.1f} success={row['success_rate']:.2f} "
                    f"collision={row['collision_rate']:.2f}"
                )

            # Primary criterion: success_rate, tie-breaker: mean_return.
            better_success = metrics["success_rate"] > self.best_success
            equal_success = abs(metrics["success_rate"] - self.best_success) < 1e-12
            better_return = metrics["mean_return"] > self.best_return
            if better_success or (equal_success and better_return):
                self.best_success = metrics["success_rate"]
                self.best_return = metrics["mean_return"]
                self.model.save(str(self.out_dir / "best_model"))
                if self.verbose:
                    print(
                        f"[eval] new best: success={self.best_success:.2f} "
                        f"return={self.best_return:.3f}; saved best_model.zip"
                    )

            with (self.out_dir / "eval_history.json").open("w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

            return True

    class TimestepsCheckpointCallback(BaseCallback):
        def __init__(self, save_freq: int, out_dir: Path, verbose: int = 1) -> None:
            super().__init__(verbose=verbose)
            self.save_freq = max(1, save_freq)
            self.out_dir = out_dir
            self.last_save_timestep = 0
            (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        def _on_step(self) -> bool:
            if (self.num_timesteps - self.last_save_timestep) < self.save_freq:
                return True
            self.last_save_timestep = int(self.num_timesteps)
            path = self.out_dir / "checkpoints" / f"ppo_t{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose:
                print(f"[ckpt] saved {path}.zip")
            return True

    checkpoint_cb = TimestepsCheckpointCallback(save_freq=args.checkpoint_freq, out_dir=run_dir, verbose=1)
    eval_cb = EvalMetricsCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        out_dir=run_dir,
        verbose=1,
    )

    # PPO network and hyperparameters:
    # - policy="MlpPolicy": feed-forward actor-critic
    # - n_steps: rollout length before each update
    # - batch_size: minibatch size for policy/value optimization
    # - gamma: reward discount
    # - learning_rate: optimizer step size
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
        verbose=1,
    )

    config = {
        "map": args.map,
        "map_path": map_path,
        "observation_mode": args.observation_mode,
        "timesteps": args.timesteps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "checkpoint_freq": args.checkpoint_freq,
        "n_envs": args.n_envs,
        "device": args.device,
        "eval_stochastic": args.eval_stochastic,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "ent_coef": args.ent_coef,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "tensorboard_log": args.tensorboard_log,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # SB3 handles rollout collection, GAE, clipped objective optimization, etc.
    # This script focuses on experiment control and metrics persistence.
    model.learn(total_timesteps=args.timesteps, callback=CallbackList([checkpoint_cb, eval_cb]))
    model.save(str(run_dir / "final_model"))

    final_metrics = evaluate_model(
        model,
        eval_env,
        args.eval_episodes,
        deterministic=not args.eval_stochastic,
    )
    with (run_dir / "final_eval.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Training complete. Outputs in: {run_dir}")
    print(f"Model device: {model.device}")
    print(f"Final eval: {final_metrics}")


if __name__ == "__main__":
    main()
