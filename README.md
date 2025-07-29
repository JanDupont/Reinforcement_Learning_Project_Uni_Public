# dependencies

see `requirements.txt`

# Training

## Baseline

```sh
python ray_training.py --experiment baseline --timesteps 153000
```

## RL Governance

```sh
python ray_training.py --experiment rl_governance --timesteps 153000
```

> alle 2500 steps (oder am Ende) wird ein checkpoint erstellt!

In public repo only last checkpoint (61) committed due to repo size.

# Run Trained Models

## Baseline

```sh
python test_trained.py results/pistonball_baseline/PPO_PistonBallRLlibEnvironment_2ee94_00000_0_2025-06-21_14-49-34/checkpoint_000061 --experiment baseline
```

## RL governance

```sh
python test_trained.py results/pistonball_rl_governance/PPO_PistonBallRLlibEnvironment_2d3f4_00000_0_2025-06-19_14-44-44/checkpoint_000061 --experiment rl_governance
```

# Evaluation

> Checkpoint files too large for GitHub!

## Baseline

```sh
python eval.py --experiment baseline --checkpoint results/pistonball_baseline/PPO_PistonBallRLlibEnvironment_2ee94_00000_0_2025-06-21_14-49-34/checkpoint_000061 --episodes 100
```

## RL Governance

```sh
python eval.py --experiment rl_governance --checkpoint results/pistonball_rl_governance/PPO_PistonBallRLlibEnvironment_2d3f4_00000_0_2025-06-19_14-44-44/checkpoint_000061 --episodes 100
```

# Tensorboard

```sh
tensorboard --logdir=tensorboard_backup
```
