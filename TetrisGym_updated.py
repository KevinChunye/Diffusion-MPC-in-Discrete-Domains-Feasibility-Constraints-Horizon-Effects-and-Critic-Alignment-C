
from __future__ import annotations

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from TetrisGame_updated import TetrisGame


Observation = namedtuple("Observation", ["board", "curr_id", "next_id"])
PIECE2IDX = {'I': 0, 'J': 1, 'L': 2, 'O': 3, 'S': 4, 'Z': 5, 'T': 6}


class TetrisGym:
    """Gym-like wrapper around `TetrisGame`.

    This environment uses **placement actions**. Each discrete action id maps to
    a pair `(rotation_index, x_position)`. The mapping is built once as the union
    of all (rot_idx, x) possibilities across all tetromino types.

    Notes:
      - `step()` returns reward=0.0 by design; agents should compute shaped
        rewards from `info` (e.g., lines cleared, top-out).
      - Valid actions are piece-dependent and are provided via
        `get_valid_action_ids()` for action masking.
    """

    def __init__(self, width=10, height=20, max_steps=None, render_mode='skip', seed=None):
        self.game = TetrisGame(width, height)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_count = 0
        self.valid_actions: list[tuple[int, int]] = []  # cached per step
        self.frames = []  # cached matplotlib figs for gif

        # Precompute the full action space: (rotation_idx, x_position)
        self.full_action_space = self._build_action_space()
        self.action_to_id = {a: i for i, a in enumerate(self.full_action_space)}
        self.id_to_action = {i: a for i, a in enumerate(self.full_action_space)}

        if seed is not None:
            # Keep RNG behavior reproducible across numpy + python random.
            np.random.seed(seed)
            try:
                import random as _random
                _random.seed(seed)
            except Exception:
                pass

        self.reset()

    @property
    def action_size(self) -> int:
        return len(self.full_action_space)

    def _build_action_space(self) -> list[tuple[int, int]]:
        """Fixed grid action space: 4 rotations x all x positions.

        Some (rot_idx, x) pairs will be invalid for a given piece and are
        excluded by `get_valid_action_ids()` / masking.
        """
        return [(rot_idx, x) for rot_idx in range(4) for x in range(self.game.width)]

    def _obs(self) -> Observation:
        board = self.game.board.astype(np.uint8)
        curr_id, _ = self.game.current_piece
        next_id, _ = self.game.next_piece
        return Observation(board, PIECE2IDX[curr_id], PIECE2IDX[next_id])

    def reset(self) -> Observation:
        self.game.reset_board()
        # `reset_board()` seeds `next_piece` but leaves `current_piece=None`.
        # Push the queue forward so `current_piece` is populated.
        self.game.spawn_new_piece()
        self.valid_actions = self.game.get_valid_actions()
        self.step_count = 0
        self.frames = []
        return self._obs()

    def step(self, action_id: int):
        if self.game.game_over:
            raise RuntimeError("Cannot step in a finished episode. Call reset().")

        rot_idx, x = self.id_to_action[action_id]
        info = self.game.update_board(rot_idx, x)
        self.step_count += 1

        # episode termination
        self.game.check_game_over()
        done = self.game.game_over or (
            self.max_steps is not None and self.step_count >= self.max_steps
        )

        # advance to next piece if not done
        if not done:
            self.game.spawn_new_piece()
            self.valid_actions = self.game.get_valid_actions()
            if not self.valid_actions:
                self.game.game_over = True
                done = True

        obs = self._obs()

        if self.render_mode == 'render':
            self.render(info)
        elif self.render_mode == 'capture':
            self.capture(info)

        # reward intentionally 0.0; agents compute shaped reward from `info`
        return obs, 0.0, done, info

    def get_valid_action_ids(self) -> list[int]:
        """Return discrete action ids corresponding to valid actions."""
        return [self.action_to_id[a] for a in self.valid_actions]

    def get_valid_actions_mask(self) -> np.ndarray:
        """Boolean mask of length `action_size` indicating which action ids are valid."""
        mask = np.zeros(self.action_size, dtype=bool)
        for aid in self.get_valid_action_ids():
            mask[aid] = True
        return mask

    def _fig_to_rgb(self, fig) -> np.ndarray:
        canvas = fig.canvas
        canvas.draw()
        if hasattr(canvas, "buffer_rgba"):
            arr = np.asarray(canvas.buffer_rgba())
            return arr[..., :3].copy()
        if hasattr(canvas, "tostring_argb"):
            w, h = canvas.get_width_height()
            buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
            rgba = buf[:, :, [1, 2, 3, 0]]
            return rgba[..., :3].copy()
        raise AttributeError("Canvas does not support buffer_rgba/tostring_argb")

    def render(self, info=None, mode=None):
        placement_mask = info.get("placement_mask") if info else None
        pre_clear_board = info.get("pre_clear_board") if info else None
        if mode == "rgb_array":
            fig = self.game.render(
                valid_actions=self.valid_actions,
                placement_mask=placement_mask,
                pre_clear_board=pre_clear_board,
                return_fig=True,
            )
            img = self._fig_to_rgb(fig)
            plt.close(fig)
            return img
        self.game.render(
            valid_actions=self.valid_actions,
            placement_mask=placement_mask,
            pre_clear_board=pre_clear_board,
        )

    def capture(self, info=None):
        placement_mask = info.get("placement_mask") if info else None
        pre_clear_board = info.get("pre_clear_board") if info else None
        fig = self.game.render(
            valid_actions=self.valid_actions,
            placement_mask=placement_mask,
            pre_clear_board=pre_clear_board,
            return_fig=True,
        )
        self.frames.append(fig)
        plt.close(fig)


    def save_gif(self, filename: str, fps: int = 2):
        """Save captured matplotlib figures to a GIF.

        Robust across Matplotlib backends/versions:
        - prefers canvas.buffer_rgba() when available
        - falls back to tostring_argb() with channel reorder
        """
        if not self.frames:
            print("No frames to save. Set render_mode='capture'.")
            return

        images = []
        for fig in self.frames:
            canvas = fig.canvas
            canvas.draw()

            # Prefer buffer_rgba() when available
            if hasattr(canvas, "buffer_rgba"):
                arr = np.asarray(canvas.buffer_rgba())  # (H,W,4) RGBA uint8
                image = arr[..., :3].copy()  # RGB
            else:
                # Fallback: ARGB bytes -> RGB
                if not hasattr(canvas, "tostring_argb"):
                    raise AttributeError(
                        "Canvas does not support buffer_rgba or tostring_argb; cannot export GIF frames."
                    )
                w, h = canvas.get_width_height()
                buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
                # ARGB -> RGBA
                rgba = buf[:, :, [1, 2, 3, 0]]
                image = rgba[..., :3].copy()

            images.append(image)
            plt.close(fig)

        imageio.mimsave(filename, images, fps=fps)

# Backwards-compatible name used by your agents
TetrisGymEnv = TetrisGym
