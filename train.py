import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from TetrisGym import TetrisGymEnv

from agent.cnn_dqn import CNNAgent
from agent.mlp_dqn import MLPAgent
from agent.tabular_q import TabularQAgent
from agent.value_dqn import ValueDQNAgent


def set_seed(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_agent(agent_name: str, board_w: int, board_h: int, device: str):
    dev = torch.device(device)

    if agent_name == "cnn":
        return CNNAgent(board_width=board_w, board_height=board_h, device=dev)
    if agent_name == "mlp":
        return MLPAgent(board_width=board_w, board_height=board_h, device=dev)
    if agent_name == "tabular":
        return TabularQAgent(board_width=board_w, board_height=board_h)
    if agent_name == "value":
        return ValueDQNAgent(board_width=board_w, board_height=board_h, device=dev)

    raise ValueError(f"Unknown agent: {agent_name}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, header: list[str], rows: list[list]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def rolling_mean(x: np.ndarray, window: int):
    if len(x) < window:
        return None
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_train_curves(run_dir: str, rewards, scores, lengths, epsilons):
    rewards = np.asarray(rewards, dtype=float)
    scores = np.asarray(scores, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    epsilons = np.asarray(epsilons, dtype=float) if epsilons is not None else None

    # reward curve
    plt.figure()
    plt.plot(rewards, label="reward")
    rm = rolling_mean(rewards, 100)
    if rm is not None:
        plt.plot(np.arange(len(rm)) + 99, rm, label="reward_ma100")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_reward.png"))
    plt.close()

    # score curve
    plt.figure()
    plt.plot(scores, label="score")
    sm = rolling_mean(scores, 100)
    if sm is not None:
        plt.plot(np.arange(len(sm)) + 99, sm, label="score_ma100")
    plt.xlabel("episode")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_score.png"))
    plt.close()

    # length curve
    plt.figure()
    plt.plot(lengths, label="episode_len")
    lm = rolling_mean(lengths, 100)
    if lm is not None:
        plt.plot(np.arange(len(lm)) + 99, lm, label="len_ma100")
    plt.xlabel("episode")
    plt.ylabel("steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_len.png"))
    plt.close()

    # epsilon curve (if available)
    if epsilons is not None:
        plt.figure()
        plt.plot(epsilons, label="epsilon")
        plt.xlabel("episode")
        plt.ylabel("epsilon")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "train_epsilon.png"))
        plt.close()


def plot_eval_hist(run_dir: str, eval_scores, eval_lengths, eval_rewards):
    eval_scores = np.asarray(eval_scores, dtype=float)
    eval_lengths = np.asarray(eval_lengths, dtype=float)
    eval_rewards = np.asarray(eval_rewards, dtype=float)

    plt.figure()
    plt.hist(eval_scores, bins=30)
    plt.xlabel("eval score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "eval_score_hist.png"))
    plt.close()

    plt.figure()
    plt.hist(eval_lengths, bins=30)
    plt.xlabel("eval episode length")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "eval_len_hist.png"))
    plt.close()

    plt.figure()
    plt.hist(eval_rewards, bins=30)
    plt.xlabel("eval reward")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "eval_reward_hist.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["cnn", "mlp", "tabular", "value"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--save_gif", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    set_seed(args.seed, device)

    # Create run directory name
    run_name = f"{args.agent}_ep{args.episodes}_ms{args.max_steps}_eval{args.eval_episodes}_seed{args.seed}"
    run_dir = os.path.join(args.runs_dir, run_name)
    ensure_dir(run_dir)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Environment
    env = TetrisGymEnv(width=args.width, height=args.height, max_steps=args.max_steps, render_mode="skip", seed=args.seed)

    # Agent
    agent = build_agent(args.agent, args.width, args.height, device)

    print(f"--- Training {args.agent} ---")
    agent.train(env, episodes=args.episodes, max_steps=args.max_steps)

    # Extract train metrics (we’ll add episode_lengths in Step C)
    train_rewards = getattr(agent, "rewards", [])
    train_scores = getattr(agent, "scores", [])
    train_lengths = getattr(agent, "episode_lengths", [])
    train_eps = getattr(agent, "eps_history", None)  # we’ll add this too

    # Save training CSV
    header = ["episode", "reward", "score", "episode_len"]
    if train_eps is not None:
        header.append("epsilon")

    rows = []
    for i in range(len(train_rewards)):
        row = [i + 1, float(train_rewards[i]), int(train_scores[i]) if i < len(train_scores) else 0,
               int(train_lengths[i]) if i < len(train_lengths) else 0]
        if train_eps is not None:
            row.append(float(train_eps[i]))
        rows.append(row)

    save_csv(os.path.join(run_dir, "train.csv"), header, rows)

    # Save plots
    plot_train_curves(
        run_dir,
        train_rewards,
        train_scores if len(train_scores) == len(train_rewards) else [0] * len(train_rewards),
        train_lengths if len(train_lengths) == len(train_rewards) else [0] * len(train_rewards),
        train_eps
    )

    # Save checkpoint
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    agent.save_agent(ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    print(f"--- Evaluating {args.agent} ---")
    eval_rewards, eval_scores, eval_lengths = agent.evaluate_agent(env, num_episodes=args.eval_episodes)

    # Save eval CSV
    eval_rows = []
    for i in range(len(eval_rewards)):
        eval_rows.append([i + 1, float(eval_rewards[i]), int(eval_scores[i]), int(eval_lengths[i])])
    save_csv(os.path.join(run_dir, "eval.csv"), ["eval_episode", "reward", "score", "episode_len"], eval_rows)

    # Save eval plots
    plot_eval_hist(run_dir, eval_scores, eval_lengths, eval_rewards)

    # Summary
    summary = {
        "eval_avg_score": float(np.mean(eval_scores)),
        "eval_avg_len": float(np.mean(eval_lengths)),
        "eval_avg_reward": float(np.mean(eval_rewards)),
        "eval_max_score": int(np.max(eval_scores)),
        "eval_pct_score_gt0": float(np.mean(np.array(eval_scores) > 0.0)),
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

    # Save GIF (separate folder + informative filename)
    if args.save_gif:
        gif_file = os.path.join(run_dir, f"gameplay_{run_name}.gif")
        agent.save_gif(gif_file, max_steps=args.max_steps)

    print(f"Done. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
