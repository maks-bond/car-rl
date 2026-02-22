from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from car_rl.core.types import Action
from car_rl.env.gym_env import make_car_gym_env
from car_rl.maps.registry import get_map_path, list_maps
from car_rl.viz.websocket_stream import WebSocketFrameStream


def _build_frame(
    *,
    env,
    state: dict,
    episode: int,
    step: int,
    action: Action,
    reward: float,
    total_reward: float,
    done: bool,
    event: Optional[str],
    control_mode: str,
) -> dict[str, Any]:
    track_map = env._env.simulator.track_map
    params = env._env.simulator.params

    return {
        "episode": episode,
        "step": step,
        "car": state,
        "action": {"a": action.a, "delta_dot": action.delta_dot},
        "reward": reward,
        "total_reward": total_reward,
        "done": done,
        "event": event,
        "map": track_map.name,
        "pose_reference": "rear_axle_center",
        "control_mode": control_mode,
        "vehicle": {
            "wheelbase": params.wheelbase,
            "front_overhang": params.front_overhang,
            "rear_overhang": params.rear_overhang,
            "width": params.width,
        },
        "map_data": {
            "walls": [{"p1": wall.p1, "p2": wall.p2} for wall in track_map.walls],
            "start_line": {"p1": track_map.start_line.p1, "p2": track_map.start_line.p2},
            "finish_line": {"p1": track_map.finish_line.p1, "p2": track_map.finish_line.p2},
        },
    }


async def simulation_loop(stream: WebSocketFrameStream, args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is not installed. Run `pip install -e .` in your venv first."
        ) from exc

    map_path = str(get_map_path(args.map))
    env = make_car_gym_env(map_path=map_path, observation_mode=args.observation_mode)
    model = PPO.load(args.model, device=args.device)

    episode = 0
    obs_vec, info = env.reset()
    state = dict(info.get("state", asdict(env._env.simulator.state)))
    step = 0
    total_reward = 0.0
    done = False
    last_event: Optional[str] = None
    last_reward = 0.0
    last_action = Action(a=0.0, delta_dot=0.0)
    mode = "paused"
    last_snapshot_time = time.monotonic()

    await stream.publish(
        _build_frame(
            env=env,
            state=state,
            episode=episode,
            step=step,
            action=last_action,
            reward=last_reward,
            total_reward=total_reward,
            done=done,
            event=last_event,
            control_mode=mode,
        )
    )

    while True:
        commands = stream.drain_commands()
        mode_before = mode
        for cmd in commands:
            if cmd.get("type") != "control":
                continue
            name = cmd.get("command")
            if name == "pause":
                mode = "paused"
            elif name == "play":
                mode = "play"
            elif name == "step":
                mode = "step"
            elif name == "play_to_end":
                mode = "play_to_end"

        if mode != mode_before:
            await stream.publish(
                _build_frame(
                    env=env,
                    state=state,
                    episode=episode,
                    step=step,
                    action=last_action,
                    reward=last_reward,
                    total_reward=total_reward,
                    done=done,
                    event=last_event,
                    control_mode=mode,
                )
            )
            last_snapshot_time = time.monotonic()

        if mode == "paused":
            now = time.monotonic()
            if now - last_snapshot_time > 0.5:
                await stream.publish(
                    _build_frame(
                        env=env,
                        state=state,
                        episode=episode,
                        step=step,
                        action=last_action,
                        reward=last_reward,
                        total_reward=total_reward,
                        done=done,
                        event=last_event,
                        control_mode=mode,
                    )
                )
                last_snapshot_time = now
            await asyncio.sleep(0.02)
            continue

        if done:
            episode += 1
            obs_vec, info = env.reset()
            state = dict(info.get("state", asdict(env._env.simulator.state)))
            step = 0
            total_reward = 0.0
            done = False
            last_event = None
            last_reward = 0.0
            last_action = Action(a=0.0, delta_dot=0.0)
            await stream.publish(
                _build_frame(
                    env=env,
                    state=state,
                    episode=episode,
                    step=step,
                    action=last_action,
                    reward=last_reward,
                    total_reward=total_reward,
                    done=done,
                    event=last_event,
                    control_mode=mode,
                )
            )
            last_snapshot_time = time.monotonic()

        action_arr, _ = model.predict(obs_vec, deterministic=not args.stochastic)
        obs_vec, reward, terminated, truncated, info = env.step(action_arr)

        step += 1
        total_reward += reward
        done = bool(terminated or truncated)
        last_event = info.get("event")
        last_reward = float(reward)
        last_action = Action(a=float(action_arr[0]), delta_dot=float(action_arr[1]))
        state = dict(info.get("state", asdict(env._env.simulator.state)))

        await stream.publish(
            _build_frame(
                env=env,
                state=state,
                episode=episode,
                step=step,
                action=last_action,
                reward=last_reward,
                total_reward=total_reward,
                done=done,
                event=last_event,
                control_mode=mode,
            )
        )
        last_snapshot_time = time.monotonic()

        if done and mode == "play_to_end":
            mode = "paused"
            await stream.publish(
                _build_frame(
                    env=env,
                    state=state,
                    episode=episode,
                    step=step,
                    action=last_action,
                    reward=last_reward,
                    total_reward=total_reward,
                    done=done,
                    event=last_event,
                    control_mode=mode,
                )
            )
            last_snapshot_time = time.monotonic()

        if mode == "step":
            mode = "paused"
            await stream.publish(
                _build_frame(
                    env=env,
                    state=state,
                    episode=episode,
                    step=step,
                    action=last_action,
                    reward=last_reward,
                    total_reward=total_reward,
                    done=done,
                    event=last_event,
                    control_mode=mode,
                )
            )
            last_snapshot_time = time.monotonic()

        await asyncio.sleep(1.0 / max(1.0, args.fps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained SB3 policy in the web viewer")
    parser.add_argument("--model", required=True, help="Path to SB3 model .zip")
    parser.add_argument("--map", default="straight_corridor", choices=list_maps())
    parser.add_argument("--observation-mode", default="boundary", choices=["state", "boundary"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cpu", help="SB3/PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--stochastic", action="store_true", help="Sample stochastic policy actions")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    stream = WebSocketFrameStream(host="127.0.0.1", port=8765)
    await asyncio.gather(stream.start(), simulation_loop(stream, args))


if __name__ == "__main__":
    asyncio.run(main())
