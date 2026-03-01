# Idea Backlog (Prioritized)

## P0: Next 2 Runs (already implemented)
1. **Value-network reranker (Exp 1)**
   - Why first: highest signal, lowest code risk.
   - Change: use `--rerank_mode dqn` in `diffusion/run_diffusion_mpc_updated.py`.
   - Success metric: higher mean/median score than heuristic rerank at similar latency.

2. **Invalid-action handling in planning loop (Exp 2)**
   - Why second: legality is a hard constraint in Tetris.
   - Change: `--invalid_handling penalize|resample` in planner.
   - Success metric: higher `%score>0`, fewer catastrophic short episodes.

## P1: Dataset/Training Quality
3. **Teacher swap: heuristic -> DQN teacher**
   - Update `diffusion/collect_diffusion_dataset_updated.py` to choose teacher action via DQN greedy on valid-action mask.
   - Compare downstream planner trained on heuristic data vs DQN data.

4. **Return-conditioned denoiser**
   - Add scalar conditioning token (target score bucket / return bin) into `PlanDenoiser`.
   - Evaluate controllability (hit rate for higher-score bins).

## P2: Sampling-Time Guidance
5. **Q-guided token refinement**
   - During sampling, bias candidate tokens toward higher Q estimates of immediate/future states.
   - Compare post-hoc reranking only vs reranking + in-loop guidance.

6. **Validity-aware logits masking**
   - Build per-step legality mask over `(rot,x)` and apply before token sampling.
   - Compare with cheaper `penalize` and `resample` repairs.

## P3: Iterative Improvement Loop
7. **Self-evolving planner data loop**
   - Roll out current planner, retain top episodes, mix into offline dataset, retrain.
   - Track score improvement over loop iterations and diversity collapse risk.

8. **Compute-normalized planning**
   - Enforce fixed runtime budget and adapt `K/H/S` accordingly.
   - Report Pareto frontier: score vs decision latency.

## P4: Publication-Ready Robustness
9. **Seed robustness**
   - Evaluate across multiple random seeds for env and sampling.
   - Report mean +- std for all primary metrics.

10. **Generalization stress tests**
   - Perturb piece distribution or board initialization patterns.
   - Test if planner improvements persist out of nominal settings.

