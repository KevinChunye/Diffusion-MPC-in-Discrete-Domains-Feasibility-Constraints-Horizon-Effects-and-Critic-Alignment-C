"""generate_teacher_rollouts.py

Generate per-step teacher transitions for expert dataset construction.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Support both direct script execution and module imports.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (THIS_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from TetrisGym_updated import TetrisGym
from diffusion_utils_updated import board_feature_summary, heuristic_score_board
from experiments.config_utils import parse_with_config


Action = Tuple[int, int]  # (rot_idx, x)


class DQNTeacher:
    def __init__(self, ckpt_path: str, board_h: int, board_w: int, device: torch.device):
        from agent.cnn_dqn_updated import DQNCNN

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        n_actions = int(ckpt.get("num_actions", 4 * board_w))
        self.device = device
        self.model = DQNCNN(n_actions=n_actions, board_h=board_h, board_w=board_w).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def q_values(self, board: np.ndarray, curr_id: int, next_id: int) -> torch.Tensor:
        board_t = torch.from_numpy(board.astype(np.float32)[None, None, ...]).to(self.device)
        curr_t = torch.tensor([int(curr_id)], dtype=torch.long, device=self.device)
        next_t = torch.tensor([int(next_id)], dtype=torch.long, device=self.device)
        return self.model((board_t, curr_t, next_t)).squeeze(0).detach().cpu()


def _enumerate_successor_boards(env: TetrisGym) -> Dict[int, np.ndarray]:
    """Map valid action_id to successor board after locking current piece."""
    successors: Dict[int, np.ndarray] = {}
    game = env.game
    _, rotations = game.current_piece
    for (rot_idx, x) in env.valid_actions:
        piece = rotations[rot_idx]
        h, w = piece.shape
        y = game._find_drop_height(piece, x)
        if y is None:
            continue
        board = game.board.copy()
        sub = board[y:y + h, x:x + w]
        board[y:y + h, x:x + w] = sub + piece
        action_id = env.action_to_id[(rot_idx, x)]
        successors[action_id] = board
    return successors


def _select_heuristic_action(env: TetrisGym) -> Optional[int]:
    successors = _enumerate_successor_boards(env)
    if not successors:
        return None
    best_id = None
    best_score = -1e18
    for aid, board in successors.items():
        score = heuristic_score_board(board)
        if score > best_score:
            best_score = score
            best_id = aid
    return best_id


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_teacher_rollouts(
    episodes: int,
    seed: int,
    out_path: str,
    teacher: str,
    dqn_ckpt: str,
    max_steps_per_episode: int,
    width: int,
    height: int,
    device: str,
) -> str:
    _set_seed(seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    env = TetrisGym(width=width, height=height, max_steps=max_steps_per_episode, render_mode="skip", seed=seed)
    dqn_teacher = None
    if teacher == "dqn":
        dqn_teacher = DQNTeacher(dqn_ckpt, board_h=height, board_w=width, device=torch_device)

    rows: List[dict] = []

    for ep in tqdm(range(episodes), desc=f"Rollouts[{teacher}]"):
        obs = env.reset()
        done = False
        t = 0
        ep_return = 0.0
        prev_score = 0.0

        while not done and t < max_steps_per_episode:
            valid_ids = env.get_valid_action_ids()
            if not valid_ids:
                break

            board = obs.board.copy()
            curr_id = int(obs.curr_id)
            next_id = int(obs.next_id)
            before = board_feature_summary(board)

            q_sa = np.nan
            q_max = np.nan
            advantage = np.nan
            invalid_fallback = False

            if teacher == "dqn":
                assert dqn_teacher is not None
                q_vals = dqn_teacher.q_values(board, curr_id=curr_id, next_id=next_id)
                proposed_id = int(torch.argmax(q_vals).item())
                valid_set = set(valid_ids)
                if proposed_id not in valid_set:
                    invalid_fallback = True
                    valid_q = q_vals[valid_ids]
                    aid = int(valid_ids[int(torch.argmax(valid_q).item())])
                else:
                    aid = proposed_id
                q_sa = float(q_vals[aid].item())
                q_max = float(q_vals[valid_ids].max().item())
                advantage = q_sa - q_max
            else:
                aid = _select_heuristic_action(env)
                if aid is None:
                    break

            (rot_idx, xpos) = env.id_to_action[aid]
            next_obs, _, done, info = env.step(aid)
            score_now = float(info.get("score", prev_score))
            reward = score_now - prev_score
            prev_score = score_now
            ep_return += reward

            after = board_feature_summary(next_obs.board)
            lines_cleared = float(info.get("lines_cleared", 0))

            rows.append(
                {
                    "episode_id": int(ep),
                    "t": int(t),
                    "board": board.astype(np.uint8),
                    "curr_id": curr_id,
                    "next_id": next_id,
                    "action_id": int(aid),
                    "action_rot": int(rot_idx),
                    "action_x": int(xpos),
                    "reward": float(reward),
                    "done": bool(done),
                    "holes_before": float(before["holes"]),
                    "holes_after": float(after["holes"]),
                    "bumpiness_before": float(before["bumpiness"]),
                    "bumpiness_after": float(after["bumpiness"]),
                    "max_height_before": float(before["max_height"]),
                    "max_height_after": float(after["max_height"]),
                    "lines_before": float(before["lines"]),
                    "lines_after": float(after["lines"]),
                    "lines_cleared": lines_cleared,
                    "q_sa": float(q_sa),
                    "q_max": float(q_max),
                    "advantage": float(advantage),
                    "invalid_fallback": bool(invalid_fallback),
                    "episode_return_so_far": float(ep_return),
                }
            )

            obs = next_obs
            t += 1

    if not rows:
        raise RuntimeError("No transitions collected. Check env/teacher settings.")

    data = {
        "episode_id": np.array([r["episode_id"] for r in rows], dtype=np.int64),
        "t": np.array([r["t"] for r in rows], dtype=np.int64),
        "board": np.stack([r["board"] for r in rows]).astype(np.uint8),  # (N,H,W)
        "curr_id": np.array([r["curr_id"] for r in rows], dtype=np.int64),
        "next_id": np.array([r["next_id"] for r in rows], dtype=np.int64),
        "action_id": np.array([r["action_id"] for r in rows], dtype=np.int64),
        "action_rot": np.array([r["action_rot"] for r in rows], dtype=np.int64),
        "action_x": np.array([r["action_x"] for r in rows], dtype=np.int64),
        "reward": np.array([r["reward"] for r in rows], dtype=np.float32),
        "done": np.array([r["done"] for r in rows], dtype=bool),
        "holes_before": np.array([r["holes_before"] for r in rows], dtype=np.float32),
        "holes_after": np.array([r["holes_after"] for r in rows], dtype=np.float32),
        "bumpiness_before": np.array([r["bumpiness_before"] for r in rows], dtype=np.float32),
        "bumpiness_after": np.array([r["bumpiness_after"] for r in rows], dtype=np.float32),
        "max_height_before": np.array([r["max_height_before"] for r in rows], dtype=np.float32),
        "max_height_after": np.array([r["max_height_after"] for r in rows], dtype=np.float32),
        "lines_before": np.array([r["lines_before"] for r in rows], dtype=np.float32),
        "lines_after": np.array([r["lines_after"] for r in rows], dtype=np.float32),
        "lines_cleared": np.array([r["lines_cleared"] for r in rows], dtype=np.float32),
        "q_sa": np.array([r["q_sa"] for r in rows], dtype=np.float32),
        "q_max": np.array([r["q_max"] for r in rows], dtype=np.float32),
        "advantage": np.array([r["advantage"] for r in rows], dtype=np.float32),
        "invalid_fallback": np.array([r["invalid_fallback"] for r in rows], dtype=bool),
        "episode_return_so_far": np.array([r["episode_return_so_far"] for r in rows], dtype=np.float32),
        "teacher": np.array([teacher]),
        "width": np.array([width], dtype=np.int64),
        "height": np.array([height], dtype=np.int64),
        "episodes": np.array([episodes], dtype=np.int64),
        "max_steps_per_episode": np.array([max_steps_per_episode], dtype=np.int64),
    }
    np.savez_compressed(out_path, **data)

    n_eps = int(np.unique(data["episode_id"]).size)
    print(f"Saved raw transitions: {out_path}")
    print(f"Transitions: {data['board'].shape[0]} | Episodes: {n_eps}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_path", type=str, default="datasets/expert_v1/raw_transitions.npz")
    parser.add_argument("--teacher", type=str, default="dqn", choices=["dqn", "heuristic"])
    parser.add_argument("--dqn_ckpt", type=str, default="dqn_updated.pt")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args, _ = parse_with_config(parser)

    generate_teacher_rollouts(
        episodes=args.episodes,
        seed=args.seed,
        out_path=args.out_path,
        teacher=args.teacher,
        dqn_ckpt=args.dqn_ckpt,
        max_steps_per_episode=args.max_steps_per_episode,
        width=args.width,
        height=args.height,
        device=args.device,
    )


if __name__ == "__main__":
    main()
