# Expert Dataset Pipeline (Phase 2)

## Goal
Build a reproducible expert dataset for diffusion training:
1. Generate teacher transitions (`raw_transitions.npz`)
2. Filter to stronger transitions (`filtered_transitions.npz`)
3. Build horizon sequences for diffusion (`sequences_H{H}.npz`)

## Files
- `/Users/haochuanwang/Desktop/Research/tetris/diffusion/generate_teacher_rollouts.py`
- `/Users/haochuanwang/Desktop/Research/tetris/diffusion/filter_expert_dataset.py`
- `/Users/haochuanwang/Desktop/Research/tetris/diffusion/build_diffusion_sequences.py`
- `/Users/haochuanwang/Desktop/Research/tetris/experiments/make_dataset.py`
- `/Users/haochuanwang/Desktop/Research/tetris/diffusion/train_diffusion_updated.py` (dataset compatibility update)

## Transition Schema (`raw_transitions.npz`, `filtered_transitions.npz`)
- `episode_id` (N,)
- `t` (N,)
- `board` (N,H,W) uint8
- `curr_id`, `next_id` (N,)
- `action_id`, `action_rot`, `action_x` (N,)
- `reward`, `done` (N,)
- board health features:
  - `holes_before`, `holes_after`
  - `bumpiness_before`, `bumpiness_after`
  - `max_height_before`, `max_height_after`
  - `lines_before`, `lines_after`
  - `lines_cleared`
- DQN diagnostics:
  - `q_sa`, `q_max`, `advantage=q_sa-q_max`
  - `invalid_fallback` (if DQN argmax was invalid and replaced by best valid)
- `episode_return_so_far`

## Sequence Schema (`sequences_H{H}.npz`)
- `board` (M,1,H,W) uint8
- `curr_id`, `next_id` (M,)
- `rot_seq` (M,H)
- `x_seq` (M,H)
- `valid_mask` (M,H)
- `rtg` (M,) return-to-go at sequence start
- `episode_id`, `t_start` (M,)
- metadata: `width`, `height`, `horizon`, `stride`, and filter/build counts

Action tokenization matches `diffusion_model_updated.py`:
- `rot_token in [0..3]`
- `x_token in [0..W-1]`

## One-command pipeline
```bash
python -m experiments.make_dataset \
  --episodes 20000 \
  --out_dir datasets/expert_v1 \
  --teacher dqn \
  --dqn_ckpt dqn_updated.pt \
  --top_episode_pct 0.10 \
  --advantage_quantile 0.80 \
  --board_health \
  --horizon 8 \
  --stride 1
```

Expected outputs:
- `datasets/expert_v1/raw_transitions.npz`
- `datasets/expert_v1/filtered_transitions.npz`
- `datasets/expert_v1/sequences_H8.npz`

## Individual steps
```bash
python diffusion/generate_teacher_rollouts.py \
  --episodes 20000 \
  --teacher dqn \
  --dqn_ckpt dqn_updated.pt \
  --out_path datasets/expert_v1/raw_transitions.npz
```

```bash
python diffusion/filter_expert_dataset.py \
  --in_path datasets/expert_v1/raw_transitions.npz \
  --out_path datasets/expert_v1/filtered_transitions.npz \
  --top_episode_pct 0.10 \
  --advantage_quantile 0.80 \
  --board_health
```

```bash
python diffusion/build_diffusion_sequences.py \
  --in_path datasets/expert_v1/filtered_transitions.npz \
  --out_path datasets/expert_v1/sequences_H8.npz \
  --horizon 8 \
  --stride 1
```

## Training with new dataset
```bash
python diffusion/train_diffusion_updated.py \
  --dataset_path datasets/expert_v1/sequences_H8.npz \
  --horizon 8 \
  --epochs 10 \
  --device cuda
```

`train_diffusion_updated.py` now prints dataset sanity stats:
- `#episodes raw`
- `#episodes kept`
- `#transitions kept`
- `#sequences built`
- `RTG min/mean/max`

## Smoke run
```bash
python -m experiments.make_dataset \
  --episodes 20 \
  --max_steps_per_episode 100 \
  --out_dir datasets/expert_smoke \
  --teacher dqn \
  --top_episode_pct 0.50 \
  --advantage_quantile 0.50 \
  --horizon 8
```

