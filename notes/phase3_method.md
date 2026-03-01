# Phase 3 Method: Constraint-Aware Sampling + Self-Training

## 1) Constraint-aware diffusion sampling
- Added `sampling_constraints` in eval/planner config:
  - `none` (default; backward-compatible)
  - `mask_logits` (new)
- New legality utility:
  - `valid_placement_mask(board, curr_piece_id)` in `diffusion/diffusion_utils_updated.py`
  - returns boolean mask over flattened placement tokens of shape `(4 * W,)`.
- In `mask_logits` mode, planner samples a single flattened placement token each step:
  - logits for `(rot, x)` formed by `rot_logits + x_logits`,
  - invalid placements masked before softmax sampling,
  - token unflattened back to `(rot, x)`.
- Metrics logging now includes:
  - `invalid_rate` (per episode + global decision-level in summary),
  - `masked_fraction` (mean fraction masked during sampling).

## 2) Self-training loop
- New runnable stage:
  - `python -m experiments.pipeline selftrain --config configs/selftrain.yaml --output_dir runs/selftrain/exp001`
- Iteration `i`:
  1. Rollout current diffusion policy -> `iter_i/rollout/raw_transitions.npz`
  2. Filter transitions -> `iter_i/rollout/filtered_transitions.npz`
  3. Build sequence dataset -> `iter_i/dataset/sequences_H{H}.npz`
  4. Train next model (warm start optional) -> `iter_{i+1}/train/checkpoints/ckpt.pt`
  5. Evaluate diffusion -> `iter_{i+1}/eval/*`
  6. Evaluate beam baseline -> `iter_{i+1}/beam_eval/*`
  7. Append `progress.csv` row with dataset sizes + key metrics.

## 3) Diversity retention
- Optional (`diversity_enable=true`) in filtering.
- Applied after top-episode selection:
  - compute final episode `(max_height_after, holes_after)`,
  - bucket episodes into quantile bins (`diversity_buckets`),
  - keep at most `diversity_per_bucket_cap` episodes per bucket,
  - then apply advantage/board-health step filtering.

## 4) Lightning-style commands
```bash
# Self-training smoke
python -m experiments.pipeline selftrain \
  --config configs/selftrain.yaml \
  --output_dir runs/smoke_selftrain \
  --smoke
```

```bash
# Full self-training run
python -m experiments.pipeline selftrain \
  --config configs/selftrain.yaml \
  --output_dir runs/selftrain/exp001
```

```bash
# Build figures from one or many self-training runs
python -m experiments.make_figures \
  --runs_glob "runs/selftrain/*" \
  --output_dir runs/selftrain/figures
```

