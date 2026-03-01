# Diffusion-Tetris

Diffusion-based Tetris planning with:
- offline teacher dataset generation
- MaskGIT-style denoiser training
- diffusion + MPC reranking evaluation
- Lightning-ready pipeline execution and job submission

## SSH-First GitHub Setup

Repository:
- SSH: `git@github.com:KevinChunye/Diffusion-Tetris.git`
- HTTPS: `https://github.com/KevinChunye/Diffusion-Tetris`

1. Generate an SSH key (if needed):
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. Add key to agent:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

3. Add key to GitHub:
```bash
cat ~/.ssh/id_ed25519.pub
```
Copy into GitHub -> Settings -> SSH and GPG keys.

4. Verify SSH auth:
```bash
ssh -T git@github.com
```

5. Clone with SSH:
```bash
git clone git@github.com:KevinChunye/Diffusion-Tetris.git
```

If your local clone currently uses HTTPS, switch it to SSH:
```bash
git remote set-url origin git@github.com:KevinChunye/Diffusion-Tetris.git
git remote -v
```

## Lightning Studio: Clone via SSH

Inside Lightning Studio:
```bash
bash scripts/lightning_clone_ssh.sh
```

Optional environment overrides:
- `DIFFUSION_TETRIS_REPO_SSH`
- `DIFFUSION_TETRIS_REPO_DIR`

Studio startup automation:
- `lightning_studio/on_start.sh` initializes known_hosts, clones/pulls repo, and creates artifact root.

## Lightning Artifacts and Pipeline

Set persistent artifact root (recommended):
```bash
export TETRIS_ARTIFACT_ROOT=/teamspace/studios/tetris_artifacts
mkdir -p "$TETRIS_ARTIFACT_ROOT"
```

Run stages:
```bash
python -m experiments.pipeline dataset --config configs/dataset_gen.yaml --output_dir runs/dataset/ds_v1 --resume true
python -m experiments.pipeline train   --config configs/diffusion_train.yaml --output_dir runs/train/tr_v1 --resume true
python -m experiments.pipeline eval    --config configs/eval_run.yaml --output_dir runs/eval/ev_v1 --resume true
```

Pipeline behavior:
- relative `output_dir` resolves under `TETRIS_ARTIFACT_ROOT` (when set)
- stage resume/skip is supported
- stage `manifest.json` is written

## Batch Job Submission

1. Create run plan:
```bash
python -m experiments.tuner \
  --index_path runs/index.csv \
  --method diffusion \
  --variant mask_logits_dqn \
  --budget 8 \
  --stage refine \
  --artifact_root "$TETRIS_ARTIFACT_ROOT"
```

2. Submit via Lightning CLI:
```bash
python -m experiments.submit_lightning_jobs \
  --plan_path runs/run_plans/<timestamp>_plan.yaml \
  --teamspace <teamspace> \
  --cluster <cluster> \
  --machine <machine> \
  --gpu <gpu>
```

Dry-run:
```bash
python -m experiments.submit_lightning_jobs --plan_path runs/run_plans/<timestamp>_plan.yaml --dry_run
```

See [Lightning runbook](notes/lightning_runbook.md) for full workflow.
