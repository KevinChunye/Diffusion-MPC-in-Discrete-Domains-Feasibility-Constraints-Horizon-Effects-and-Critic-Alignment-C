# Lightning Runbook

## 0) Clone Repo via SSH in Studio

Use SSH clone (passwordless once key is configured):

```bash
bash scripts/lightning_clone_ssh.sh
```

Defaults:
- repo: `git@github.com:KevinChunye/Diffusion-Tetris.git`
- destination: `$HOME/Diffusion-Tetris`

You can override:
- `DIFFUSION_TETRIS_REPO_SSH`
- `DIFFUSION_TETRIS_REPO_DIR`

## 1) Set Persistent Artifact Root

Use a persistent teamspace path and export it once per session:

```bash
python -m experiments.lightning_env --print
export TETRIS_ARTIFACT_ROOT=/teamspace/studios/tetris_artifacts
```

Recommended layout under `TETRIS_ARTIFACT_ROOT`:

- `datasets/`
- `checkpoints/`
- `runs/`

## 2) Dataset / Train / Eval / Selftrain Commands

All commands below write to persistent storage via `TETRIS_ARTIFACT_ROOT`.

```bash
python -m experiments.pipeline dataset \
  --config configs/dataset_gen.yaml \
  --output_dir runs/dataset/ds_v1

python -m experiments.pipeline train \
  --config configs/diffusion_train.yaml \
  --output_dir runs/train/denoiser_v1 \
  --dataset_path datasets/expert_v1/sequences_H8.npz

python -m experiments.pipeline eval \
  --config configs/eval_run.yaml \
  --output_dir runs/eval/mask_logits_dqn_v1 \
  --ckpt runs/train/denoiser_v1/checkpoints/plan_denoiser.pt \
  --sampling_constraints mask_logits \
  --rerank_mode dqn

python -m experiments.pipeline selftrain \
  --config configs/selftrain.yaml \
  --output_dir runs/selftrain/exp001
```

Notes:

- Pipeline `--resume` defaults to `true` on Lightning (auto-detected).
- Re-running the same stage with completed artifacts will skip execution and still register/index.

## 3) Tuner -> Plan -> Lightning Submission

Generate a run plan:

```bash
python -m experiments.tuner \
  --index_path runs/index.csv \
  --method diffusion \
  --variant mask_logits_dqn \
  --budget 8 \
  --stage refine \
  --artifact_root "$TETRIS_ARTIFACT_ROOT"
```

Dry-run submissions:

```bash
python -m experiments.submit_lightning_jobs \
  --plan_path runs/run_plans/<timestamp>_plan.yaml \
  --dry_run
```

Submit jobs:

```bash
python -m experiments.submit_lightning_jobs \
  --plan_path runs/run_plans/<timestamp>_plan.yaml \
  --teamspace <your-teamspace> \
  --cluster <cluster-name> \
  --machine <machine-type> \
  --gpu <gpu-type>
```

Submission logs are written to `runs/submissions/<timestamp>.md`.

## 4) Analyze + Figures from Existing Runs

```bash
python -m experiments.analyze_run --run_dir runs/eval/mask_logits_dqn_v1

python -m experiments.make_figures \
  --index_path runs/index.csv \
  --runs_glob 'runs/*' \
  --output_dir runs/figures/paper
```

## 5) Daily Checklist

1. Confirm `TETRIS_ARTIFACT_ROOT` points to persistent teamspace storage.
2. Run `pipeline` stages with stable `output_dir` names.
3. Verify stage skips/resumes correctly on reruns.
4. Run tuner and create a new plan.
5. Submit plan with `submit_lightning_jobs`.
6. Check `runs/index.csv`, `manifest.json`, and submission logs.
7. Generate updated figures/tables from historical outputs.
