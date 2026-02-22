from __future__ import annotations

import asyncio
from pathlib import Path

from car_rl.agents.constant import ConstantActionAgent
from car_rl.apps.common import create_env
from car_rl.viz.websocket_stream import WebSocketFrameStream


async def simulation_loop(stream: WebSocketFrameStream, fps: float = 30.0) -> None:
    map_path = Path(__file__).resolve().parent.parent / "maps" / "straight_corridor.json"
    env = create_env(map_path)
    agent = ConstantActionAgent(a=1.0, delta_dot=0.1)

    episode = 0
    while True:
        obs = env.reset()
        total_reward = 0.0

        for step in range(env.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            frame = {
                "episode": episode,
                "step": step,
                "car": obs,
                "action": {"a": action.a, "delta_dot": action.delta_dot},
                "reward": reward,
                "total_reward": total_reward,
                "done": done,
                "event": info["event"],
                "map": env.simulator.track_map.name,
            }
            await stream.publish(frame)

            if done:
                await asyncio.sleep(1.0)
                break
            await asyncio.sleep(1.0 / fps)

        episode += 1


async def main() -> None:
    stream = WebSocketFrameStream(host="127.0.0.1", port=8765)
    await asyncio.gather(stream.start(), simulation_loop(stream))


if __name__ == "__main__":
    asyncio.run(main())
