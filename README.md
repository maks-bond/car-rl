# car-rl

Starter project for reinforcement-learning experiments with a headless bicycle-model car simulator.

## What is included

- Headless simulator (`src/car_rl/core/simulator.py`)
- Kinematic bicycle dynamics with action/state limits (`src/car_rl/core/dynamics.py`)
- Map format and first map (`src/car_rl/maps/straight_corridor.json`)
- RL-style env wrapper (`src/car_rl/env/environment.py`)
- Constant action baseline agent (`src/car_rl/agents/constant.py`)
- Live WebSocket frame stream (`src/car_rl/viz/websocket_stream.py`)
- Minimal browser visualization (`web/index.html`)

## Model definition

State:
- `x, y, yaw, v, delta`
- Pose convention: `x, y` is the center of the rear axle (not the geometric center of the body)

Control:
- `a` (acceleration)
- `delta_dot` (steering angle rate)

Dynamics:
- `x_dot = v * cos(yaw)`
- `y_dot = v * sin(yaw)`
- `yaw_dot = v / L * tan(delta)`
- `v_dot = a`
- `delta_dot = u_delta_dot`

## Reward/events

- Finish line crossed in forward direction: `+100` and terminate
- Collision with wall: `-100` and terminate
- Start line crossed backward direction: `-100` and terminate
- Step penalty: `-0.01`

## Quickstart (Most Important)

Use Python 3.11+ (recommended) and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Run headless simulation:

```bash
PYTHONPATH=src python3 -m car_rl.apps.run_headless
```

Run web visualization (2 terminals):

Terminal A (simulation + websocket server):
```bash
PYTHONPATH=src python3 -m car_rl.apps.run_viz
```

Terminal B (static web server):

```bash
python3 -m http.server 8080
```

Open in browser:

- `http://127.0.0.1:8080/web/`

## UI controls (web viewer)

- `Step`: execute exactly one simulator step, then pause
- `Play`: run continuously
- `Pause`: pause immediately
- `Play To End`: run until episode termination, then auto-pause before next episode

The panel also shows:
- episode, step, reward, event
- control mode
- speed (`v`)
- steering angle (`delta`) in degrees

## Collision model

- Collision is checked against the **oriented car body box**, not a point/circle.
- Car geometry is defined in `VehicleParams` (`wheelbase`, `front_overhang`, `rear_overhang`, `width`).

## Project layout

- `src/car_rl/core/`: dynamics, geometry, map loading, simulator
- `src/car_rl/env/`: RL-style environment wrapper
- `src/car_rl/agents/`: baseline agents
- `src/car_rl/apps/`: runnable entrypoints
- `src/car_rl/maps/`: map files
- `src/car_rl/viz/`: websocket stream code
- `web/`: browser renderer UI

## Troubleshooting

- `python: command not found`: use `python3` in commands.
- `ModuleNotFoundError: websockets`: activate venv and run `pip install -e .`.
- No updates in browser: ensure `run_viz` is running and refresh the page.

## Next steps

1. Add more maps in `src/car_rl/maps/`
2. Replace `ConstantActionAgent` with engineered policy
3. Add Gymnasium API compatibility and train PPO/SAC
4. Add checkpoints and dense progress rewards
