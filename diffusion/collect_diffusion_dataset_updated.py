"""collect_diffusion_dataset_updated.py

Collect offline training data for the plan denoiser.

Each example:
- board: (1,H,W) uint8
- curr_id,next_id: int64
- rot_seq,x_seq: (horizon,) int64
- valid_mask: (horizon,) bool

Teacher policy (no training required): greedy one-step lookahead using a fast
heuristic on successor boards. For sequence data, we roll this greedy teacher
forward H steps in a deepcopy of the current state.

Usage:
  python collect_diffusion_dataset_updated.py --out datasets/tetris_h5.npz --episodes 500 --horizon 5
"""

from __future__ import annotations

import argparse
import copy
from typing import List, Tuple, Dict

import numpy as np

from TetrisGym_updated import TetrisGym
from diffusion_utils_updated import action_seq_to_tokens, heuristic_score_board


Action = Tuple[int, int]


def _enumerate_successors(sim_env: TetrisGym) -> Dict[Action, np.ndarray]:
    """Return mapping (rot,x) -> successor board (after locking piece, before spawning next)."""
    game = sim_env.game
    piece_type, rotations = game.current_piece
    succ: Dict[Action, np.ndarray] = {}

    # valid actions already respects piece-specific rotations
    for (rot_idx, x) in sim_env.valid_actions:
        piece = rotations[rot_idx]
        h, w = piece.shape
        y = game._find_drop_height(piece, x)
        if y is None:
            continue
        board = game.board.copy()
        sub = board[y:y+h, x:x+w]
        board[y:y+h, x:x+w] = sub + piece
        succ[(rot_idx, x)] = board
    return succ


def _greedy_action(sim_env: TetrisGym) -> Action | None:
    succ = _enumerate_successors(sim_env)
    if not succ:
        return None
    best_a, best_s = None, -1e18
    for a, board in succ.items():
        s = heuristic_score_board(board)
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def _rollout_teacher_sequence(env: TetrisGym, horizon: int) -> List[Action]:
    """From current state, generate a greedy length<=H sequence."""
    actions: List[Action] = []
    sim_env = copy.deepcopy(env)

    for _ in range(horizon):
        if sim_env.game.game_over:
            break
        a = _greedy_action(sim_env)
        if a is None:
            break
        actions.append(a)
        aid = sim_env.action_to_id[a]
        _, _, done, _ = sim_env.step(aid)
        if done:
            break
    return actions


def collect_dataset(out_path: str, episodes: int, max_steps: int, horizon: int, width: int, height: int, seed: int):
    env = TetrisGym(width=width, height=height, max_steps=max_steps, render_mode="skip", seed=seed)

    boards = []
    curr_ids = []
    next_ids = []
    rot_seqs = []
    x_seqs = []
    masks = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            board, curr_id, next_id = obs

            seq = _rollout_teacher_sequence(env, horizon=horizon)
            rot, xs, mask = action_seq_to_tokens(seq, horizon=horizon, width=width)

            boards.append(board.astype(np.uint8)[None, ...])
            curr_ids.append(int(curr_id))
            next_ids.append(int(next_id))
            rot_seqs.append(rot)
            x_seqs.append(xs)
            masks.append(mask)

            if len(seq) == 0:
                done = True
                break

            first_action = seq[0]
            aid = env.action_to_id[first_action]
            obs, _, done, _ = env.step(aid)
            steps += 1

    data = {
        "board": np.stack(boards),
        "curr_id": np.array(curr_ids, dtype=np.int64),
        "next_id": np.array(next_ids, dtype=np.int64),
        "rot_seq": np.stack(rot_seqs).astype(np.int64),
        "x_seq": np.stack(x_seqs).astype(np.int64),
        "valid_mask": np.stack(masks).astype(bool),
        "width": np.array([width], dtype=np.int64),
        "height": np.array([height], dtype=np.int64),
        "horizon": np.array([horizon], dtype=np.int64),
    }

    np.savez_compressed(out_path, **data)
    print(f"Saved dataset to {out_path} with N={data['board'].shape[0]} examples")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="datasets/tetris_plan_h5.npz")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    collect_dataset(args.out, args.episodes, args.max_steps, args.horizon, args.width, args.height, args.seed)


if __name__ == "__main__":
    main()
