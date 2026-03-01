# Diffusion Tetris Experiment Board

## Goal
Run diffusion-style planning with two low-risk, high-signal changes:
1. Value-network reranking (DQN critic).
2. Invalid-action handling during candidate planning.

## Metrics (per run)
- Mean score over `N` episodes.
- Median score over `N` episodes.
- Mean steps survived.
- `% episodes with score > 0`.
- Runtime per decision step (ms).

## Experiments
| ID | Hypothesis | Minimal code change | Expected outcome |
|---|---|---|---|
| Exp1 | Replacing heuristic reranker with learned DQN value reranker improves score quality | Add `--rerank_mode dqn --critic_ckpt dqn_updated.pt`; load DQNCNN in planner and score candidates with `V(s)=max_a Q(s,a)` | Higher mean/median score at moderate runtime cost |
| Exp2 | Explicit invalid-action handling reduces catastrophic bad candidates and improves consistency | Add `--invalid_handling {none,penalize,resample}` and apply penalty/repair during candidate simulation | Higher `%score>0`, better stability, possible slight latency increase |

## Required Ablations
- `K` candidates: e.g. `32,64`.
- Horizon `H`: e.g. `5`.
- Sampling steps `S`: e.g. `8`.
- Temperature `T`: e.g. `0.8,1.0`.

## Implemented Entry Points
- Single-run eval: `diffusion/run_diffusion_mpc_updated.py`
  - New flags: reranker mode, critic checkpoint, invalid handling, logging output.
- Reproducible batch run: `python -m experiments.run_all`
  - Runs Exp1 and Exp2 over `K/H/S/T` grid.
  - Writes `summary.csv`, per-episode CSVs, plots, and `report.md`.

## Example Commands
```bash
# Single experiment run
python diffusion/run_diffusion_mpc_updated.py \
  --ckpt checkpoints/plan_denoiser.pt \
  --episodes 100 \
  --rerank_mode dqn \
  --critic_ckpt dqn_updated.pt \
  --invalid_handling penalize \
  --num_candidates 64 \
  --horizon 5 \
  --sample_steps 8 \
  --temperature 1.0
```

```bash
# Full reproducible board (Exp1 + Exp2 + ablations)
python -m experiments.run_all \
  --ckpt checkpoints/plan_denoiser.pt \
  --critic_ckpt dqn_updated.pt \
  --episodes 100 \
  --k_values 32,64 \
  --h_values 5 \
  --s_values 8 \
  --temp_values 0.8,1.0
```

