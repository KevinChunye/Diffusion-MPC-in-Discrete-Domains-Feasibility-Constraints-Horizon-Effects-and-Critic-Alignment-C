"""diffusion_utils_updated.py

Utilities for diffusion dataset collection and diffusion MPC planning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    from agent.value_dqn import board_props
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from agent.value_dqn import board_props

try:
    from TetrisGame_updated import TetrisGame
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from TetrisGame_updated import TetrisGame

Action = Tuple[int, int]  # (rot_idx, x)


@dataclass
class SimResult:
    done: bool
    steps: int
    infos: List[Dict]


def clone_env(env):
    """Deep-copy env (and its game) so we can simulate plans safely."""
    return copy.deepcopy(env)


def simulate_action_sequence(env, actions: List[Action], stop_on_done: bool = True) -> Tuple[np.ndarray, SimResult]:
    """Roll out a candidate action sequence in a cloned env.

    Args:
        env: a TetrisGym_updated.TetrisGym instance (will be mutated).
        actions: list of (rot_idx, x) actions to apply.
        stop_on_done: if True, stops once game is done.

    Returns:
        final_board: numpy array (H,W) of the final board.
        result: SimResult containing done flag and collected infos.
    """
    infos: List[Dict] = []
    steps = 0
    done = False
    for (rot_idx, x) in actions:
        # env.step expects action_id; map (rot,x) -> id
        action_id = env.action_to_id.get((rot_idx, x), None)
        if action_id is None:
            done = True
            break
        _, _, done, info = env.step(action_id)
        infos.append(info)
        steps += 1
        if stop_on_done and done:
            break
    return env.game.board.copy(), SimResult(done=done, steps=steps, infos=infos)


def heuristic_score_board(board: np.ndarray) -> float:
    """Fast board score for ranking diffusion candidates.

    Uses board_props from value_dqn:
        [lines, max_h, min_h, total_h, max_bump, total_bump, holes]

    We want:
      - more cleared lines (good)
      - fewer holes (very good)
      - lower total height and bumpiness (good)

    Returns a scalar score (higher is better).
    """
    feats = board_props(board.astype(np.uint8))
    lines = float(feats[0])
    max_h = float(feats[1])
    total_h = float(feats[3])
    total_bump = float(feats[5])
    holes = float(feats[6])

    # weights tuned to be reasonable defaults; you can tune later
    return (
        5.0 * lines
        - 0.8 * holes
        - 0.05 * total_h
        - 0.03 * total_bump
        - 0.02 * max_h
    )


def board_feature_summary(board: np.ndarray) -> Dict[str, float]:
    """Return compact board-health features used in dataset logging."""
    feats = board_props(board.astype(np.uint8))
    return {
        "lines": float(feats[0]),
        "max_height": float(feats[1]),
        "bumpiness": float(feats[5]),
        "holes": float(feats[6]),
    }


PIECE2IDX = {"I": 0, "J": 1, "L": 2, "O": 3, "S": 4, "Z": 5, "T": 6}
IDX2PIECE = {v: k for k, v in PIECE2IDX.items()}


def valid_placement_mask(board: np.ndarray, curr_piece_id: int) -> np.ndarray:
    """Return legality mask over flattened placement actions (rot * W + x).

    Shape: (4 * width,), where width = board.shape[1].
    """
    board = np.asarray(board)
    if board.ndim != 2:
        raise ValueError(f"board must be 2D (H,W), got shape={board.shape}")
    h, w = board.shape
    piece = IDX2PIECE.get(int(curr_piece_id), None)
    if piece is None:
        raise ValueError(f"Unknown curr_piece_id={curr_piece_id}")

    game = TetrisGame(width=w, height=h)
    game.board = board.astype(int).copy()
    rotations = TetrisGame.TETROMINOES[piece]

    mask = np.zeros((4 * w,), dtype=bool)
    for rot_idx in range(4):
        if rot_idx >= len(rotations):
            continue
        p = rotations[rot_idx]
        _, pw = p.shape
        for x in range(w):
            flat = rot_idx * w + x
            if x + pw > w:
                continue
            y = game._find_drop_height(p, x)
            if y is None:
                continue
            if game._valid_position(p, (y, x)):
                mask[flat] = True
    return mask


def action_seq_to_tokens(actions: List[Action], horizon: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert list of (rot,x) to (rot_tokens, x_tokens, valid_mask).

    rot_tokens: int64 (H,) in {0..3}
    x_tokens:   int64 (H,) in {0..width-1}
    valid_mask: bool  (H,) True for valid entries, False for padding
    """
    rot = np.zeros((horizon,), dtype=np.int64)
    xs = np.zeros((horizon,), dtype=np.int64)
    mask = np.zeros((horizon,), dtype=bool)
    for i in range(min(horizon, len(actions))):
        r, x = actions[i]
        rot[i] = int(r)
        xs[i] = int(x)
        mask[i] = True
    return rot, xs, mask


def tokens_to_action_seq(rot_tokens: np.ndarray, x_tokens: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> List[Action]:
    """Convert tokens back to list of (rot,x)."""
    horizon = int(rot_tokens.shape[0])
    actions: List[Action] = []
    for i in range(horizon):
        if valid_mask is not None and not bool(valid_mask[i]):
            break
        actions.append((int(rot_tokens[i]), int(x_tokens[i])))
    return actions
