# CODEX Handoff Notes

This file is a continuity handoff for future Codex sessions.

## Project Purpose

`car-rl` is a reinforcement-learning playground around a headless car simulator (kinematic bicycle model) with web visualization and SB3 PPO training/evaluation tooling.

## Current State (Important)

- Python requirement: `>=3.11` (see `pyproject.toml`)
- Package layout: `src/`-based
- Main simulator supports:
  - bicycle dynamics
  - limits on accel/velocity/steering/steering-rate
  - collision with walls
  - finish/start-line event logic
- Collision model is oriented **car box vs wall segments** (not point/circle).
- Web UI supports:
  - play/pause/step/play-to-end
  - steering visualization
  - policy rollout visualization
- RL wrappers support:
  - raw state observations
  - boundary-only observations (ray distances + internal state)

## Main Commands

1. Train SB3 PPO

```bash
python3 -m car_rl.apps.train_sb3_ppo \
  --map straight_corridor \
  --observation-mode boundary \
  --timesteps 100000 \
  --eval-freq 5000 \
  --eval-episodes 10 \
  --device cuda
```

2. Evaluate model

```bash
python3 -m car_rl.apps.eval_policy \
  --model runs/<run_name>/best_model.zip \
  --map straight_corridor \
  --observation-mode boundary \
  --episodes 20 \
  --device cuda
```

3. Visualize trained policy in UI

```bash
python3 -m car_rl.apps.run_viz_policy \
  --model runs/<run_name>/best_model.zip \
  --map straight_corridor \
  --observation-mode boundary \
  --device cuda
```

4. UI infrastructure

```bash
python3 -m http.server 8080
```

Open: `http://127.0.0.1:8080/web/`

## Key Files

### Simulation / Env
- `src/car_rl/core/dynamics.py`
- `src/car_rl/core/simulator.py`
- `src/car_rl/core/geometry.py`
- `src/car_rl/core/map_data.py`
- `src/car_rl/env/environment.py`
- `src/car_rl/env/observation.py` (boundary ray features)
- `src/car_rl/env/features.py` (policy vector adapter)
- `src/car_rl/env/gym_env.py` (Gymnasium wrapper)

### Maps
- `src/car_rl/maps/straight_corridor.json`
- `src/car_rl/maps/easy_turn.json`
- `src/car_rl/maps/s_curve.json`
- `src/car_rl/maps/maze_small.json`
- `src/car_rl/maps/registry.py`

### Agents
- `src/car_rl/agents/constant.py`
- `src/car_rl/agents/engineered.py`

### Apps
- `src/car_rl/apps/run_headless.py`
- `src/car_rl/apps/run_viz.py`
- `src/car_rl/apps/run_viz_policy.py`
- `src/car_rl/apps/run_gym_smoke.py`
- `src/car_rl/apps/train_sb3_ppo.py`
- `src/car_rl/apps/eval_policy.py`
- `src/car_rl/apps/benchmark_agents.py`
- `src/car_rl/apps/inspect_policy_input.py`

### Web
- `web/index.html`

### Docs
- `README.md`

## Training/Eval Behavior Notes

- PPO can temporarily reach good policies and then regress later.
- Best practice: rely on `best_model.zip` (selected by eval callback), not `final_model.zip`.
- Example observed behavior:
  - intermediate eval reached success rate 1.0
  - final model regressed to timeout policy
  - `best_model.zip` still evaluated at 1.0 success on corridor

## Existing PPO Script Details

`train_sb3_ppo.py` currently supports:
- `--device` (`auto/cpu/cuda/...`)
- `--n-envs`
- eval every N timesteps
- checkpoints every N timesteps
- tie-break in best model selection: success rate then mean return
- TensorBoard log path (`--tensorboard-log`)
- PPO params (`lr`, `gamma`, `n_steps`, `batch_size`, `ent_coef`, `gae_lambda`, `clip_range`)

## Known Caveats

- SB3 warns that PPO + MLP often runs better on CPU than GPU. User prefers GPU and accepts this.
- `maze_small` is intentionally harder; engineered baseline currently fails there.

## Recommended Next Steps

1. Add reward shaping option (progress-to-finish) behind a flag.
2. Add curriculum training sequence: `straight_corridor -> easy_turn -> s_curve -> maze_small`.
3. Add cross-map evaluation script (generalization matrix).
4. Add custom PPO scaffold (`train_custom_ppo.py`) for educational implementation.
5. Add policy export + deterministic replay logger (state/action trace).

## Environment / Setup Notes

- User works in `.venv` and runs from repo root.
- `pip install -e .` used for dependencies.
- `.gitignore` includes runs/artifacts.

## If Starting Fresh Session

Ask user first:
- Continue SB3 workflow or begin custom PPO implementation?
- Keep boundary-only observations or add hybrid/state features?
- Optimize for training speed (CPU) or stick with CUDA preference?
