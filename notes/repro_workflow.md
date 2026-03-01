# Repro Workflow (Lightning-ready)

## 1) Dataset Build
```bash
python -m experiments.pipeline dataset \
  --config configs/dataset_gen.yaml \
  --output_dir runs/pipeline_dataset
```

## 2) Diffusion Training
```bash
python -m experiments.pipeline train \
  --config configs/diffusion_train.yaml \
  --output_dir runs/pipeline_train
```

## 3) Diffusion Eval
```bash
python -m experiments.pipeline eval \
  --config configs/eval_run.yaml \
  --output_dir runs/pipeline_eval
```

## 4) Full Pipeline (dataset -> train -> eval)
```bash
python -m experiments.pipeline full \
  --dataset_config configs/dataset_gen.yaml \
  --train_config configs/diffusion_train.yaml \
  --eval_config configs/eval_run.yaml \
  --output_dir runs/pipeline_full
```

## 5) Beam Search Baseline Eval
```bash
python baselines/run_beam_search.py \
  --config configs/beam_eval.yaml \
  --output_dir runs/beam_eval
```

## Smoke mode (fast checks)
```bash
python -m experiments.pipeline dataset --config configs/dataset_gen.yaml --output_dir runs/smoke_dataset --smoke
python -m experiments.pipeline train --config configs/diffusion_train.yaml --output_dir runs/smoke_train --smoke
python -m experiments.pipeline eval --config configs/eval_run.yaml --output_dir runs/smoke_eval --smoke
python -m experiments.pipeline full --output_dir runs/smoke_full --smoke
python -m experiments.pipeline selftrain --config configs/selftrain.yaml --output_dir runs/smoke_selftrain --smoke
```

## Artifacts written per run directory
- `config.yaml` (resolved config)
- `git_commit.txt` (best effort)
- `pip_freeze.txt` (best effort)
- `metrics.csv`
- `summary.csv`
- `checkpoints/`
