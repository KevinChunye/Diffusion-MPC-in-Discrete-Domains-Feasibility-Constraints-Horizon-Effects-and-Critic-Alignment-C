# Diffusion-Tetris, code for Diffusion-MPC in Discrete Domains: Feasibility Constraints, Horizon Effects, and Critic Alignment: Case study with Tetris


Diffusion-based planning for Tetris.

This repository implements a full pipeline for training and evaluating diffusion-style policies for sequential decision-making:

- Offline **teacher dataset generation**
- **MaskGIT-style denoiser** training
- **Diffusion + MPC reranking** evaluation
- **Lightning-ready** experiment pipeline and batch job execution

---

## Repository Setup

Clone using SSH:

```bash
git clone git@github.com:KevinChunye/Diffusion-Tetris.git
cd Diffusion-Tetris
```

If you do not yet have an SSH key:

```bash
ssh-keygen -t ed25519
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

Add the public key to GitHub:

```bash
cat ~/.ssh/id_ed25519.pub
```

Then verify authentication:

```bash
ssh -T git@github.com
```

---

## Lightning Studio Setup

Clone the repository inside Lightning Studio:

```bash
bash scripts/lightning_clone_ssh.sh
```

Optional environment variables:

```
DIFFUSION_TETRIS_REPO_SSH
DIFFUSION_TETRIS_REPO_DIR
```

Startup automation is handled by:

```
lightning_studio/on_start.sh
```

This script:

- initializes SSH known hosts  
- clones or updates the repo  
- prepares the artifact directory  

---

## Experiment Pipeline

Set a persistent artifact directory:

```bash
export TETRIS_ARTIFACT_ROOT=/teamspace/studios/tetris_artifacts
mkdir -p "$TETRIS_ARTIFACT_ROOT"
```

Run the pipeline stages:

```bash
python -m experiments.pipeline dataset \
  --config configs/dataset_gen.yaml \
  --output_dir runs/dataset/ds_v1 \
  --resume true
```

```bash
python -m experiments.pipeline train \
  --config configs/diffusion_train.yaml \
  --output_dir runs/train/tr_v1 \
  --resume true
```

```bash
python -m experiments.pipeline eval \
  --config configs/eval_run.yaml \
  --output_dir runs/eval/ev_v1 \
  --resume true
```

Pipeline features:

- automatic artifact management  
- resumable stages  
- stage manifests (`manifest.json`)  

---

## Batch Job Submission

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

Submit jobs via Lightning:

```bash
python -m experiments.submit_lightning_jobs \
  --plan_path runs/run_plans/<timestamp>_plan.yaml \
  --teamspace <teamspace> \
  --cluster <cluster> \
  --machine <machine> \
  --gpu <gpu>
```

Dry run:

```bash
python -m experiments.submit_lightning_jobs \
  --plan_path runs/run_plans/<timestamp>_plan.yaml \
  --dry_run
```

---

## Documentation

Additional workflow details are available in:

`notes/lightning_runbook.md`
