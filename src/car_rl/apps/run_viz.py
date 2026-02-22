from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

from car_rl.agents.constant import ConstantActionAgent
from car_rl.apps.common import create_env
from car_rl.core.types import Action
from car_rl.viz.websocket_stream import WebSocketFrameStream


def _build_frame(
    *,
    env,
    obs: dict,
    episode: int,
    step: int,
    action: Action,
    reward: float,
    total_reward: float,
    done: bool,
    event: Optional[str],
    control_mode: str,
) -> dict[str, Any]:
    track_map = env.simulator.track_map
    params = env.simulator.params

    return {
        "episode": episode,
        "step": step,
        "car": obs,
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


async def simulation_loop(stream: WebSocketFrameStream, fps: float = 30.0) -> None:
    map_path = Path(__file__).resolve().parent.parent / "maps" / "straight_corridor.json"
    env = create_env(map_path)
    agent = ConstantActionAgent(a=1.0, delta_dot=0.1)

    episode = 0
    obs = env.reset()
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
            obs=obs,
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
                    obs=obs,
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
                        obs=obs,
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
            obs = env.reset()
            step = 0
            total_reward = 0.0
            done = False
            last_event = None
            last_reward = 0.0
            last_action = Action(a=0.0, delta_dot=0.0)
            await stream.publish(
                _build_frame(
                    env=env,
                    obs=obs,
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

        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        step += 1
        total_reward += reward
        last_event = info.get("event")
        last_reward = reward
        last_action = action

        await stream.publish(
            _build_frame(
                env=env,
                obs=obs,
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
                    obs=obs,
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
                    obs=obs,
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

        await asyncio.sleep(1.0 / fps)


async def main() -> None:
    stream = WebSocketFrameStream(host="127.0.0.1", port=8765)
    await asyncio.gather(stream.start(), simulation_loop(stream))


if __name__ == "__main__":
    asyncio.run(main())
