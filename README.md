# car-rl

Starter project for reinforcement-learning experiments with a headless bicycle-model car simulator.

## What is included

- Headless simulator (`src/car_rl/core/simulator.py`)
- Kinematic bicycle dynamics with action/state limits (`src/car_rl/core/dynamics.py`)
- Map format and first map (`src/car_rl/maps/straight_corridor.json`)
- Additional maps: `easy_turn`, `s_curve`, `maze_small`
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

Use Python 3.11+ and create a virtual environment:

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

Run headless with map/agent selection:

```bash
PYTHONPATH=src python3 -m car_rl.apps.run_headless --map easy_turn --agent engineered --episodes 3
```

Available maps can be discovered from `src/car_rl/maps/`.
Available agents:
- `constant`
- `engineered` (centerline-following baseline)

Observation modes:
- `state` (default): returns `x, y, yaw, v, delta`
- `boundary`: returns boundary-based features only (no `x, y, yaw`)

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

## Baseline benchmark

Run constant vs engineered policy over all maps:

```bash
PYTHONPATH=src python3 -m car_rl.apps.benchmark_agents --episodes 5
```

Example: run specific maps only:

```bash
PYTHONPATH=src python3 -m car_rl.apps.benchmark_agents --episodes 5 --maps straight_corridor easy_turn s_curve
```

## Boundary-based learning features

To run boundary-only observations:

```bash
PYTHONPATH=src python3 -m car_rl.apps.run_headless --map straight_corridor --agent constant --observation-mode boundary
```

Boundary observation vector/schema:
- `v_norm`: normalized speed
- `delta_norm`: normalized steering angle
- `ray_distances`: raw wall distances from lidar-style rays
- `ray_distances_norm`: distances normalized by max ray range

Ray setup:
- 15 rays
- 180-degree field of view centered on vehicle heading
- max range: 20 m

Notes:
- This mode does not include centerline features.
- `engineered` agent requires `--observation-mode state`.

## Policy-network input adapter (PPO/SAC-ready)

Use `PolicyInputAdapter` to convert observations into fixed-size `numpy.float32` vectors.

State mode vector (dim = 6):
- `[x, y, sin_yaw, cos_yaw, v, delta]`

Boundary mode vector (default dim = 17):
- `[v_norm, delta_norm, ray_00_norm, ..., ray_14_norm]`

Quick inspection command:

```bash
PYTHONPATH=src python3 -m car_rl.apps.inspect_policy_input --map straight_corridor --observation-mode boundary --steps 3
```

Python usage:

```python
from car_rl.apps.common import create_env
from car_rl.env.features import PolicyInputAdapter
from car_rl.maps.registry import get_map_path

env = create_env(get_map_path("straight_corridor"), observation_mode="boundary")
adapter = PolicyInputAdapter(observation_mode="boundary")

obs = env.reset()
vec = adapter.transform(obs)  # np.ndarray shape: (17,), dtype=float32
```

## Gymnasium wrapper (vector observations)

Use `CarGymEnv` when plugging directly into RL libraries that expect Gymnasium API:
- `reset() -> (obs_vec, info)`
- `step(action) -> (obs_vec, reward, terminated, truncated, info)`
- `action` is `[a, delta_dot]`

Smoke test:

```bash
PYTHONPATH=src python3 -m car_rl.apps.run_gym_smoke --map straight_corridor --observation-mode boundary --steps 30
```

Python usage:

```python
from car_rl.env.gym_env import make_car_gym_env
from car_rl.maps.registry import get_map_path

env = make_car_gym_env(str(get_map_path("straight_corridor")), observation_mode="boundary")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([1.0, 0.0])
```

## Train PPO (Stable-Baselines3)

Install/update dependencies:

```bash
pip install -e .
```

Quick training run:

```bash
PYTHONPATH=src python3 -m car_rl.apps.train_sb3_ppo \
  --map straight_corridor \
  --observation-mode boundary \
  --timesteps 50000 \
  --eval-freq 5000 \
  --eval-episodes 5
```

Training outputs are saved under `runs/ppo_<map>_<mode>_<timestamp>/`:
- `config.json`
- `checkpoints/*.zip`
- `best_model.zip`
- `final_model.zip`
- `eval_history.json`
- `final_eval.json`

Evaluate a trained model:

```bash
PYTHONPATH=src python3 -m car_rl.apps.eval_policy \
  --model runs/<run_name>/best_model.zip \
  --map straight_corridor \
  --observation-mode boundary \
  --episodes 20
```

Suggested first test plan:
1. Train on `straight_corridor` until success rate is near 1.0.
2. Evaluate on `straight_corridor` to confirm stable performance.
3. Train/evaluate on `easy_turn`, then `s_curve`.

## Where PPO Is Implemented

PPO optimization is provided by Stable-Baselines3, not handwritten in this repository.

- Orchestration in this repo: `src/car_rl/apps/train_sb3_ppo.py`
- SB3 PPO class (actual algorithm implementation):
  - Docs: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
  - Source: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py

Related SB3 internals worth reading:
- On-policy base loop (rollout/update flow): https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py
- Rollout buffer (advantages/returns storage): https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py

Core PPO theory:
- PPO paper (Schulman et al., 2017): https://arxiv.org/abs/1707.06347
- OpenAI Spinning Up PPO explanation: https://spinningup.openai.com/en/latest/algorithms/ppo.html

## Troubleshooting

- `python: command not found`: use `python3` in commands.
- `ModuleNotFoundError: websockets`: activate venv and run `pip install -e .`.
- No updates in browser: ensure `run_viz` is running and refresh the page.

## Next steps

1. Add more maps in `src/car_rl/maps/`
2. Replace `ConstantActionAgent` with engineered policy
3. Add Gymnasium API compatibility and train PPO/SAC
4. Add checkpoints and dense progress rewards
