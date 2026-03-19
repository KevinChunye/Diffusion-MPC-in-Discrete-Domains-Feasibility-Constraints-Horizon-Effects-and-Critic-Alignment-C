# Diffusion-Tetris  
**Code for**  
**_Diffusion-MPC in Discrete Domains: Feasibility Constraints, Horizon Effects, and Critic Alignment_**  
Case Study: Tetris  

📄 Paper: https://arxiv.org/abs/2603.02348  

---

## Overview

This repository provides a **complete implementation of diffusion-based planning in discrete domains**, using Tetris as a case study.

It includes:

- Offline **dataset generation** (teacher policy)
- **MaskGIT-style diffusion policy** training
- **Diffusion + MPC reranking** for decision-time planning
- Scalable **experiment pipeline** (Lightning-ready)

---

## Quick Start

Clone the repository:

```bash
git clone git@github.com:KevinChunye/Diffusion-Tetris.git
cd Diffusion-Tetris
```

Run the full pipeline:

```bash
# 1. Generate dataset
python -m experiments.pipeline dataset   --config configs/dataset_gen.yaml   --output_dir runs/dataset/ds_v1

# 2. Train diffusion model
python -m experiments.pipeline train   --config configs/diffusion_train.yaml   --output_dir runs/train/tr_v1

# 3. Evaluate with MPC reranking
python -m experiments.pipeline eval   --config configs/eval_run.yaml   --output_dir runs/eval/ev_v1
```

---

## Pipeline

The framework is structured into three stages:

1. **Dataset Generation** — collect expert trajectories  
2. **Training** — learn diffusion-style policy (MaskGIT denoiser)  
3. **Evaluation** — MPC-style reranking with critic  

Key features:

- Resumable runs (`--resume`)  
- Structured outputs (`manifest.json`)  
- Unified artifact management  

---

## Scalable Experiments (Lightning)

To run large-scale jobs:

```bash
bash scripts/lightning_clone_ssh.sh
```

Submit batch experiments:

```bash
python -m experiments.submit_lightning_jobs   --plan_path runs/run_plans/<timestamp>_plan.yaml   --teamspace <teamspace>   --cluster <cluster>   --machine <machine>   --gpu <gpu>
```

---

## Project Structure

```
configs/        # Experiment configs
experiments/    # Pipeline + training/eval logic
scripts/        # Setup and automation
runs/           # Outputs (datasets, models, eval)
notes/          # Additional documentation
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026diffusion_tetris,
  title={Diffusion-MPC in Discrete Domains: Feasibility Constraints, Horizon Effects, and Critic Alignment},
  author={Wang, Haochuan},
  year={2026},
  eprint={2603.02348},
  archivePrefix={arXiv}
}
```

---

## Notes

Additional details:  
`notes/lightning_runbook.md`
