from __future__ import annotations

from pathlib import Path

from car_rl.agents.constant import ConstantActionAgent
from car_rl.apps.common import create_env


def main() -> None:
    map_path = Path(__file__).resolve().parent.parent / "maps" / "straight_corridor.json"
    env = create_env(map_path)
    agent = ConstantActionAgent(a=1.0, delta_dot=0.0)

    obs = env.reset()
    total_reward = 0.0

    for step in range(env.max_steps):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if step % 20 == 0 or done:
            print(
                f"step={step} x={obs['x']:.2f} y={obs['y']:.2f} v={obs['v']:.2f} "
                f"reward={reward:.2f} total={total_reward:.2f} event={info['event']}"
            )
        if done:
            break

    print(f"done={done} total_reward={total_reward:.2f} steps={step+1}")


if __name__ == "__main__":
    main()
