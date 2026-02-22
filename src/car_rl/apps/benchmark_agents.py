from __future__ import annotations

import argparse
from typing import Optional

from car_rl.agents.constant import ConstantActionAgent
from car_rl.agents.engineered import EngineeredLaneFollowerAgent
from car_rl.apps.common import create_env
from car_rl.maps.registry import get_map_path, list_maps


def run_episode(env, agent) -> tuple[float, Optional[str], int]:
    obs = env.reset()
    total_reward = 0.0
    for step in range(env.max_steps):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            return total_reward, info.get("event"), step + 1
    return total_reward, None, env.max_steps


def benchmark(map_name: str, episodes: int) -> None:
    env = create_env(get_map_path(map_name))
    centerline = env.simulator.track_map.centerline

    agents = [("constant", ConstantActionAgent(a=1.0, delta_dot=0.0))]
    if centerline:
        agents.append(("engineered", EngineeredLaneFollowerAgent(centerline=centerline)))

    for agent_name, agent in agents:
        success = 0
        collisions = 0
        returns: list[float] = []
        steps: list[int] = []

        for _ in range(episodes):
            total_reward, event, used_steps = run_episode(env, agent)
            returns.append(total_reward)
            steps.append(used_steps)
            if event == "finish":
                success += 1
            if event == "collision":
                collisions += 1

        avg_return = sum(returns) / len(returns)
        avg_steps = sum(steps) / len(steps)
        print(
            f"map={map_name:16s} agent={agent_name:10s} "
            f"success_rate={success / episodes:.2f} collision_rate={collisions / episodes:.2f} "
            f"avg_return={avg_return:8.2f} avg_steps={avg_steps:6.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline agents across maps")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--maps", nargs="*", default=list_maps())
    args = parser.parse_args()

    for map_name in args.maps:
        benchmark(map_name, args.episodes)


if __name__ == "__main__":
    main()
