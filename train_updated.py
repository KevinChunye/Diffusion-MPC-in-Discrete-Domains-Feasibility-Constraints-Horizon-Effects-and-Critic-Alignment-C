"""train_updated.py

Updated training/evaluation runner for the UPDATED CNN-DQN stack:
  - TetrisGame_updated.py
  - TetrisGym_updated.py
  - agent/cnn_dqn_updated.py (board + piece-id embeddings + shaped reward + loss logging)

Creates a run directory under --runs_dir and saves:
  - config.json
  - train.csv + plots (reward/score/len/epsilon/loss if available)
  - eval.csv + plots
  - summary.json
  - checkpoint.pt
  - optional gameplay GIF

Usage example:
  python train_updated.py --episodes 2000 --max_steps 2000 --eval_episodes 100 --device cuda --save_gif
"""

from __future__ import annotations

import argparse
import json
import os
import random
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from TetrisGym_updated import TetrisGym
from agent.cnn_dqn_updated import CNNAgent


# -----------------------------
# Small utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _moving_avg(x: list[float], window: int = 100) -> list[float]:
    if len(x) == 0:
        return []
    out = []
    for i in range(len(x)):
        j0 = max(0, i - window + 1)
        out.append(float(np.mean(x[j0 : i + 1])))
    return out


def plot_train_curves(
    run_dir: str,
    rewards: list[float],
    scores: list[int],
    lengths: list[int],
    eps: list[float] | None,
    losses: list[float] | None,
    ma_window: int = 100,
) -> None:
    """Writes PNGs to run_dir."""

    # rewards
    plt.figure()
    plt.plot(rewards, label="reward")
    plt.plot(_moving_avg(rewards, ma_window), label=f"reward_ma{ma_window}")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_reward.png"))
    plt.close()

    # scores
    plt.figure()
    plt.plot(scores, label="score")
    plt.plot(_moving_avg([float(s) for s in scores], ma_window), label=f"score_ma{ma_window}")
    plt.xlabel("episode")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_score.png"))
    plt.close()

    # lengths
    plt.figure()
    plt.plot(lengths, label="episode_len")
    plt.plot(_moving_avg([float(l) for l in lengths], ma_window), label=f"len_ma{ma_window}")
    plt.xlabel("episode")
    plt.ylabel("steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_len.png"))
    plt.close()

    # epsilon
    if eps is not None:
        plt.figure()
        plt.plot(eps, label="epsilon")
        plt.xlabel("episode")
        plt.ylabel("epsilon")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "train_epsilon.png"))
        plt.close()

    # loss (episode mean loss, if agent provides it)
    if losses is not None and len(losses) == len(rewards):
        plt.figure()
        plt.plot(losses, label="episode_loss")
        plt.plot(_moving_avg(losses, ma_window), label=f"loss_ma{ma_window}")
        plt.xlabel("episode")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "train_loss.png"))
        plt.close()


def plot_eval_hist(run_dir: str, scores: list[int], lengths: list[int], rewards: list[float]) -> None:
    """Writes PNG histograms to run_dir."""

    plt.figure()
    plt.hist(scores, bins=30)
    plt.xlabel("eval score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "eval_score_hist.png"))
    plt.close()

    plt.figure()
    plt.hist(lengths, bins=30)
    plt.xlabel("eval episode length")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "eval_len_hist.png"))
    plt.close()

    plt.figure()
    plt.hist(rewards, bins=30)
    plt.xlabel("eval reward")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "eval_reward_hist.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--save_gif", action="store_true")

    # Agent hyperparams
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.01)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--memory_size", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--target_sync", type=int, default=64)

    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    set_seed(args.seed, device)

    # Run directory name: model + episodes + max_steps + seed (plus a few key hparams)
    agent_name = "cnn_updated"
    run_name = (
        f"{agent_name}_ep{args.episodes}_ms{args.max_steps}_seed{args.seed}"
        f"_lr{args.lr:g}_g{args.gamma:g}"
        f"_bs{args.batch_size}_ts{args.target_sync}"
    )
    run_dir = os.path.join(args.runs_dir, run_name)
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    env = TetrisGym(
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        render_mode="skip",
        seed=args.seed,
    )

    agent = CNNAgent(
        board_width=args.width,
        board_height=args.height,
        alpha=args.lr,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        target_sync=args.target_sync,
        device=torch.device(device),
    )

    print("--- Training cnn_updated ---")
    agent.train(env, episodes=args.episodes, max_steps=args.max_steps)

    # Extract train metrics
    train_rewards = getattr(agent, "rewards", [])
    train_scores = getattr(agent, "scores", [])
    train_lengths = getattr(agent, "episode_lengths", [])
    train_eps = getattr(agent, "eps_history", None)
    train_loss = getattr(agent, "episode_loss", None)

    # Save training CSV
    header = ["episode", "reward", "score", "episode_len"]
    if train_eps is not None:
        header.append("epsilon")
    if train_loss is not None and len(train_loss) == len(train_rewards):
        header.append("loss")

    rows = []
    for i in range(len(train_rewards)):
        row = [
            i + 1,
            float(train_rewards[i]),
            int(train_scores[i]) if i < len(train_scores) else 0,
            int(train_lengths[i]) if i < len(train_lengths) else 0,
        ]
        if train_eps is not None:
            row.append(float(train_eps[i]))
        if "loss" in header:
            row.append(float(train_loss[i]))
        rows.append(row)

    save_csv(os.path.join(run_dir, "train.csv"), header, rows)

    # Save plots
    plot_train_curves(
        run_dir,
        train_rewards,
        train_scores if len(train_scores) == len(train_rewards) else [0] * len(train_rewards),
        train_lengths if len(train_lengths) == len(train_rewards) else [0] * len(train_rewards),
        train_eps,
        train_loss if ("loss" in header) else None,
    )

    # Save checkpoint
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    agent.save_agent(ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Evaluation
    print("--- Evaluating cnn_updated ---")
    eval_rewards, eval_scores, eval_lengths = agent.evaluate_agent(env, num_episodes=args.eval_episodes)

    # Save eval CSV
    eval_rows = []
    for i in range(len(eval_rewards)):
        eval_rows.append([i + 1, float(eval_rewards[i]), int(eval_scores[i]), int(eval_lengths[i])])
    save_csv(
        os.path.join(run_dir, "eval.csv"),
        ["eval_episode", "reward", "score", "episode_len"],
        eval_rows,
    )

    # Save eval plots
    plot_eval_hist(run_dir, eval_scores, eval_lengths, eval_rewards)

    # Summary
    summary = {
        "eval_avg_score": float(np.mean(eval_scores)) if eval_scores else 0.0,
        "eval_avg_len": float(np.mean(eval_lengths)) if eval_lengths else 0.0,
        "eval_avg_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
        "eval_max_score": int(np.max(eval_scores)) if eval_scores else 0,
        "eval_pct_score_gt0": float(np.mean(np.array(eval_scores) > 0.0)) if eval_scores else 0.0,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Eval avg score: {summary['eval_avg_score']:.2f} | "
        f"avg len: {summary['eval_avg_len']:.2f} | "
        f"avg reward: {summary['eval_avg_reward']:.2f} | "
        f"max score: {summary['eval_max_score']} | "
        f"%score>0: {summary['eval_pct_score_gt0']:.2%}"
    )

    # Save GIF
    if args.save_gif:
        gif_file = os.path.join(run_dir, f"gameplay_{run_name}.gif")
        agent.save_gif(gif_file, max_steps=args.max_steps)

    print(f"Done. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
