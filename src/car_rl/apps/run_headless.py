from __future__ import annotations

import argparse

from car_rl.agents.constant import ConstantActionAgent
from car_rl.agents.engineered import EngineeredLaneFollowerAgent
from car_rl.apps.common import create_env
from car_rl.maps.registry import get_map_path, list_maps


def build_agent(agent_name: str, env, args: argparse.Namespace):
    if agent_name == "constant":
        return ConstantActionAgent(a=args.a, delta_dot=args.delta_dot)
    if agent_name == "engineered":
        if args.observation_mode != "state":
            raise ValueError("engineered agent requires observation_mode=state")
        centerline = env.simulator.track_map.centerline
        if not centerline:
            raise ValueError("selected map has no centerline waypoints for engineered agent")
        return EngineeredLaneFollowerAgent(centerline=centerline)
    raise ValueError(f"unknown agent '{agent_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run headless simulation episodes")
    parser.add_argument("--map", default="straight_corridor", choices=list_maps())
    parser.add_argument("--agent", default="constant", choices=["constant", "engineered"])
    parser.add_argument("--observation-mode", default="state", choices=["state", "boundary"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--a", type=float, default=1.0, help="constant action acceleration")
    parser.add_argument("--delta-dot", type=float, default=0.0, help="constant steering angle rate")
    args = parser.parse_args()

    map_path = get_map_path(args.map)
    env = create_env(map_path, observation_mode=args.observation_mode)
    agent = build_agent(args.agent, env, args)

    success = 0
    collisions = 0
    for episode in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        final_event = None

        for step in range(env.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if step % 20 == 0 or done:
                if args.observation_mode == "state":
                    print(
                        f"ep={episode} step={step} x={obs['x']:.2f} y={obs['y']:.2f} v={obs['v']:.2f} "
                        f"reward={reward:.2f} total={total_reward:.2f} event={info['event']}"
                    )
                else:
                    rays = obs["ray_distances_norm"]
                    min_ray = min(rays) if rays else 0.0
                    print(
                        f"ep={episode} step={step} v_norm={obs['v_norm']:.2f} delta_norm={obs['delta_norm']:.2f} "
                        f"min_ray_norm={min_ray:.2f} reward={reward:.2f} total={total_reward:.2f} event={info['event']}"
                    )
            if done:
                final_event = info["event"]
                break

        if final_event == "finish":
            success += 1
        if final_event == "collision":
            collisions += 1
        print(
            f"episode={episode} done={done} event={final_event} total_reward={total_reward:.2f} steps={step+1}"
        )

    print(
        f"summary map={args.map} agent={args.agent} episodes={args.episodes} "
        f"success_rate={success / max(1, args.episodes):.2f} collisions={collisions}"
    )


if __name__ == "__main__":
    main()
