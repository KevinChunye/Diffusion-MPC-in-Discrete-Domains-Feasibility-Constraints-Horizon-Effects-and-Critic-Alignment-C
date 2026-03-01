"""train_diffusion_updated.py

Train the MaskGIT plan denoiser on an offline dataset.

Typical workflow:
  1) Collect data:
     python collect_diffusion_dataset_updated.py --out datasets/tetris_h5.npz --episodes 500 --horizon 5
  2) Train denoiser:
     python train_diffusion_updated.py --data datasets/tetris_h5.npz --epochs 10 --device cuda
  3) Evaluate with MPC:
     python run_diffusion_mpc_updated.py --ckpt checkpoints/plan_denoiser.pt --episodes 100
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

# Allow running as `python diffusion/train_diffusion_updated.py` from repo root.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (THIS_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diffusion_model_updated import PlanDenoiser, masked_ce_loss
from experiments.config_utils import parse_with_config
from experiments.hashing import sha256_file, write_json
from experiments.repro import prepare_run_artifacts


def _mask_batch(rot: torch.Tensor, xs: torch.Tensor, valid_mask: torch.Tensor, rot_mask_id: int, x_mask_id: int, p_mask: float):
    """Randomly mask a subset of tokens.

    Returns:
        rot_in, xs_in: masked inputs
        rot_masked, xs_masked: boolean masks indicating which tokens were masked
    """
    device = rot.device
    B, H = rot.shape

    # only mask valid positions (not padding)
    bern_rot = (torch.rand((B, H), device=device) < p_mask) & valid_mask
    bern_x = (torch.rand((B, H), device=device) < p_mask) & valid_mask

    rot_in = rot.clone()
    xs_in = xs.clone()
    rot_in[bern_rot] = rot_mask_id
    xs_in[bern_x] = x_mask_id

    return rot_in, xs_in, bern_rot, bern_x


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--data", type=str, default="", help="Legacy dataset flag (npz).")
    p.add_argument("--dataset_path", type=str, default="", help="Preferred dataset path (npz).")
    p.add_argument("--horizon", type=int, default=0, help="Optional override/check for sequence horizon.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mask_prob", type=float, default=0.5)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="runs/diffusion_train")
    p.add_argument("--save", type=str, default="")
    p.add_argument("--init_ckpt", type=str, default="", help="Optional warm-start checkpoint.")
    p.add_argument("--log_token_acc", type=int, default=0, help="If 1, log small held-out token accuracy each epoch.")
    p.add_argument("--seed", type=int, default=0)
    return p


def run_train(args: argparse.Namespace) -> dict:
    if not args.save:
        args.save = os.path.join(args.output_dir, "checkpoints", "plan_denoiser.pt")
    artifacts = prepare_run_artifacts(args.output_dir, vars(args).copy())

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = args.dataset_path or args.data
    if not dataset_path:
        raise ValueError("Provide --dataset_path (or legacy --data).")

    data = np.load(dataset_path)
    board = torch.tensor(data["board"], dtype=torch.float32)  # (N,1,H,W)
    curr_id = torch.tensor(data["curr_id"], dtype=torch.long)
    next_id = torch.tensor(data["next_id"], dtype=torch.long)
    rot_seq = torch.tensor(data["rot_seq"], dtype=torch.long)
    x_seq = torch.tensor(data["x_seq"], dtype=torch.long)
    valid_mask = torch.tensor(data["valid_mask"], dtype=torch.bool)

    N = board.shape[0]
    height = int(data["height"][0])
    width = int(data["width"][0])
    horizon = int(data["horizon"][0]) if "horizon" in data else int(rot_seq.shape[1])
    if args.horizon > 0 and args.horizon != horizon:
        raise ValueError(f"--horizon={args.horizon} does not match dataset horizon={horizon}")

    raw_eps = int(data["meta_raw_num_episodes"][0]) if "meta_raw_num_episodes" in data else -1
    if "meta_kept_num_episodes" in data:
        kept_eps = int(data["meta_kept_num_episodes"][0])
    elif "episode_id" in data:
        kept_eps = int(np.unique(data["episode_id"]).shape[0])
    else:
        kept_eps = -1
    kept_trans = int(data["meta_kept_num_transitions"][0]) if "meta_kept_num_transitions" in data else N
    built_seq = int(data["meta_sequences_built"][0]) if "meta_sequences_built" in data else N
    if "rtg" in data:
        rtg = data["rtg"].astype(np.float32)
        rtg_msg = f"RTG stats min/mean/max = {float(np.min(rtg)):.3f}/{float(np.mean(rtg)):.3f}/{float(np.max(rtg)):.3f}"
    else:
        rtg_msg = "RTG stats: not available in dataset"

    print(f"Loaded dataset: {dataset_path}")
    print(f"#episodes raw={raw_eps}, #episodes kept={kept_eps}, #transitions kept={kept_trans}, #sequences built={built_seq}")
    print(rtg_msg)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = PlanDenoiser(
        board_h=height,
        board_w=width,
        horizon=horizon,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
    ).to(device)
    if args.init_ckpt:
        init_ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        init_sd = init_ckpt.get("state_dict", init_ckpt)
        missing, unexpected = model.load_state_dict(init_sd, strict=False)
        print(f"Warm-start from {args.init_ckpt}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    idx = np.arange(N)
    epoch_rows = []
    np.random.shuffle(idx)
    val_size = min(max(1, N // 10), 256) if N > 1 else 1
    val_idx = idx[:val_size]
    train_idx = idx[val_size:] if val_size < N else idx

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(train_idx)
        losses = []
        epoch_t0 = time.perf_counter()

        pbar = tqdm(range(0, train_idx.shape[0], args.batch_size), desc=f"epoch {epoch}/{args.epochs}")
        model.train()

        for s in pbar:
            b = train_idx[s : s + args.batch_size]
            b_board = board[b].to(device)
            b_curr = curr_id[b].to(device)
            b_next = next_id[b].to(device)
            b_rot = rot_seq[b].to(device)
            b_x = x_seq[b].to(device)
            b_valid = valid_mask[b].to(device)

            rot_in, x_in, rot_masked, x_masked = _mask_batch(
                b_rot, b_x, b_valid,
                rot_mask_id=model.tokens.rot_mask,
                x_mask_id=model.tokens.x_mask,
                p_mask=args.mask_prob,
            )

            rot_logits, x_logits = model(b_board, b_curr, b_next, rot_in, x_in)
            loss = masked_ce_loss(rot_logits, x_logits, b_rot, b_x, rot_masked, x_masked)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(losses[-50:]))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        rot_acc = ""
        x_acc = ""
        if bool(int(args.log_token_acc)) and val_idx.size > 0:
            model.eval()
            with torch.no_grad():
                vb = val_idx[: min(int(args.batch_size), val_idx.size)]
                v_board = board[vb].to(device)
                v_curr = curr_id[vb].to(device)
                v_next = next_id[vb].to(device)
                v_rot = rot_seq[vb].to(device)
                v_x = x_seq[vb].to(device)
                v_valid = valid_mask[vb].to(device)

                rot_in = v_rot.clone()
                x_in = v_x.clone()
                rot_in[v_valid] = model.tokens.rot_mask
                x_in[v_valid] = model.tokens.x_mask
                rot_logits, x_logits = model(v_board, v_curr, v_next, rot_in, x_in)
                rot_pred = torch.argmax(rot_logits, dim=-1)
                x_pred = torch.argmax(x_logits, dim=-1)
                if v_valid.any():
                    rot_acc = float((rot_pred[v_valid] == v_rot[v_valid]).float().mean().item())
                    x_acc = float((x_pred[v_valid] == v_x[v_valid]).float().mean().item())
                else:
                    rot_acc = 0.0
                    x_acc = 0.0

        epoch_rows.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "rot_acc": rot_acc,
                "x_acc": x_acc,
                "lr": float(opt.param_groups[0]["lr"]),
                "walltime_sec": float(time.perf_counter() - epoch_t0),
            }
        )
        print(f"Epoch {epoch} mean loss: {mean_loss:.4f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "width": width,
            "height": height,
            "horizon": horizon,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
        },
        args.save,
    )
    canonical_ckpt = os.path.join(args.output_dir, "checkpoints", "ckpt.pt")
    if os.path.abspath(args.save) != os.path.abspath(canonical_ckpt):
        os.makedirs(os.path.dirname(canonical_ckpt), exist_ok=True)
        shutil.copyfile(args.save, canonical_ckpt)
    print(f"Saved checkpoint to {args.save}")

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    train_metrics_path = os.path.join(args.output_dir, "train_metrics.csv")
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "rot_acc", "x_acc", "lr", "walltime_sec"])
        writer.writeheader()
        writer.writerows(epoch_rows)
    with open(train_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "rot_acc", "x_acc", "lr", "walltime_sec"])
        writer.writeheader()
        writer.writerows(epoch_rows)

    dataset_hash = sha256_file(dataset_path)
    model_hash = sha256_file(args.save)
    param_count = int(sum(p.numel() for p in model.parameters()))
    ckpt_meta = {
        "model_hash": model_hash,
        "dataset_hash_used": dataset_hash,
        "config_hash_used": artifacts.get("config_hash", ""),
        "param_count": param_count,
        "best_loss": float(min([r["loss"] for r in epoch_rows])) if epoch_rows else 0.0,
        "checkpoint_path": args.save,
    }
    write_json(os.path.join(os.path.dirname(args.save), "ckpt_meta.json"), ckpt_meta)

    summary = {
        "run_name": "diffusion_train",
        "dataset_path": dataset_path,
        "dataset_hash": dataset_hash,
        "epochs": int(args.epochs),
        "final_mean_loss": float(epoch_rows[-1]["loss"]) if epoch_rows else 0.0,
        "num_sequences": int(N),
        "horizon": int(horizon),
        "checkpoint_path": args.save,
        "model_hash": model_hash,
        "init_ckpt": args.init_ckpt,
    }
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"metrics.csv: {metrics_path}")
    print(f"train_metrics.csv: {train_metrics_path}")
    print(f"summary.csv: {summary_path}")
    return summary


def main():
    parser = build_parser()
    args, _ = parse_with_config(parser)
    run_train(args)


if __name__ == "__main__":
    main()
