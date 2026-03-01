from __future__ import annotations

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from TetrisGame import TetrisGame


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
        """Union of all (rotation_index, x) pairs across all tetromino types."""
        seen = set()
        actions: list[tuple[int, int]] = []
        for _, rotations in self.game.TETROMINOES.items():
            for rot_idx, piece in enumerate(rotations):
                _, piece_width = piece.shape
                for x in range(self.game.width - piece_width + 1):
                    key = (rot_idx, x)
                    if key not in seen:
                        seen.add(key)
                        actions.append(key)
        return actions

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

    def render(self, info=None):
        placement_mask = info.get("placement_mask") if info else None
        pre_clear_board = info.get("pre_clear_board") if info else None
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
        """
        Save captured frames to a GIF.

        Fix: Matplotlib in Lightning AI may not support `tostring_rgb()` on Agg canvas.
        We use `buffer_rgba()` (works on modern Matplotlib) and drop alpha channel.
        """
        if not self.frames:
            print("No frames to save. Set render_mode='capture'.")
            return

        images = []
        for fig in self.frames:
            fig.canvas.draw()

            # Robust: Agg canvas provides RGBA buffer
            w, h = fig.canvas.get_width_height()
            rgba = np.asarray(fig.canvas.buffer_rgba())  # shape (h, w, 4), uint8
            rgb = rgba[:, :, :3].copy()                  # drop alpha
            images.append(rgb)

            plt.close(fig)

        imageio.mimsave(filename, images, fps=fps)


# Backwards-compatible name used by your agents
TetrisGymEnv = TetrisGym


# from __future__ import annotations

# from collections import namedtuple

# import numpy as np
# import matplotlib.pyplot as plt
# import imageio.v2 as imageio

# from TetrisGame import TetrisGame


# Observation = namedtuple("Observation", ["board", "curr_id", "next_id"])
# PIECE2IDX = {'I': 0, 'J': 1, 'L': 2, 'O': 3, 'S': 4, 'Z': 5, 'T': 6}


# class TetrisGym:
#     """Gym-like wrapper around `TetrisGame`.

#     This environment uses **placement actions**. Each discrete action id maps to
#     a pair `(rotation_index, x_position)`. The mapping is built once as the union
#     of all (rot_idx, x) possibilities across all tetromino types.

#     Notes:
#       - `step()` returns reward=0.0 by design; agents should compute shaped
#         rewards from `info` (e.g., lines cleared, top-out).
#       - Valid actions are piece-dependent and are provided via
#         `get_valid_action_ids()` for action masking.
#     """

#     def __init__(self, width=10, height=20, max_steps=None, render_mode='skip', seed=None):
#         self.game = TetrisGame(width, height)
#         self.max_steps = max_steps
#         self.render_mode = render_mode
#         self.step_count = 0
#         self.valid_actions: list[tuple[int, int]] = []  # cached per step
#         self.frames = []  # cached matplotlib figs for gif

#         # Precompute the full action space: (rotation_idx, x_position)
#         self.full_action_space = self._build_action_space()
#         self.action_to_id = {a: i for i, a in enumerate(self.full_action_space)}
#         self.id_to_action = {i: a for i, a in enumerate(self.full_action_space)}

#         if seed is not None:
#             # Keep RNG behavior reproducible across numpy + python random.
#             np.random.seed(seed)
#             try:
#                 import random as _random
#                 _random.seed(seed)
#             except Exception:
#                 pass

#         self.reset()

#     @property
#     def action_size(self) -> int:
#         return len(self.full_action_space)

#     def _build_action_space(self) -> list[tuple[int, int]]:
#         """Union of all (rotation_index, x) pairs across all tetromino types."""
#         seen = set()
#         actions: list[tuple[int, int]] = []
#         for _, rotations in self.game.TETROMINOES.items():
#             for rot_idx, piece in enumerate(rotations):
#                 _, piece_width = piece.shape
#                 for x in range(self.game.width - piece_width + 1):
#                     key = (rot_idx, x)
#                     if key not in seen:
#                         seen.add(key)
#                         actions.append(key)
#         return actions

#     def _obs(self) -> Observation:
#         board = self.game.board.astype(np.uint8)
#         curr_id, _ = self.game.current_piece
#         next_id, _ = self.game.next_piece
#         return Observation(board, PIECE2IDX[curr_id], PIECE2IDX[next_id])

#     def reset(self) -> Observation:
#         self.game.reset_board()
#         # `reset_board()` seeds `next_piece` but leaves `current_piece=None`.
#         # Push the queue forward so `current_piece` is populated.
#         self.game.spawn_new_piece()
#         self.valid_actions = self.game.get_valid_actions()
#         self.step_count = 0
#         self.frames = []
#         return self._obs()

#     def step(self, action_id: int):
#         if self.game.game_over:
#             raise RuntimeError("Cannot step in a finished episode. Call reset().")

#         rot_idx, x = self.id_to_action[action_id]
#         info = self.game.update_board(rot_idx, x)
#         self.step_count += 1

#         # episode termination
#         self.game.check_game_over()
#         done = self.game.game_over or (
#             self.max_steps is not None and self.step_count >= self.max_steps
#         )

#         # advance to next piece if not done
#         if not done:
#             self.game.spawn_new_piece()
#             self.valid_actions = self.game.get_valid_actions()
#             if not self.valid_actions:
#                 self.game.game_over = True
#                 done = True

#         obs = self._obs()

#         if self.render_mode == 'render':
#             self.render(info)
#         elif self.render_mode == 'capture':
#             self.capture(info)

#         # reward intentionally 0.0; agents compute shaped reward from `info`
#         return obs, 0.0, done, info

#     def get_valid_action_ids(self) -> list[int]:
#         """Return discrete action ids corresponding to valid actions."""
#         return [self.action_to_id[a] for a in self.valid_actions]

#     def get_valid_actions_mask(self) -> np.ndarray:
#         """Boolean mask of length `action_size` indicating which action ids are valid."""
#         mask = np.zeros(self.action_size, dtype=bool)
#         for aid in self.get_valid_action_ids():
#             mask[aid] = True
#         return mask

#     def render(self, info=None):
#         placement_mask = info.get("placement_mask") if info else None
#         pre_clear_board = info.get("pre_clear_board") if info else None
#         self.game.render(
#             valid_actions=self.valid_actions,
#             placement_mask=placement_mask,
#             pre_clear_board=pre_clear_board,
#         )

#     def capture(self, info=None):
#         placement_mask = info.get("placement_mask") if info else None
#         pre_clear_board = info.get("pre_clear_board") if info else None
#         fig = self.game.render(
#             valid_actions=self.valid_actions,
#             placement_mask=placement_mask,
#             pre_clear_board=pre_clear_board,
#             return_fig=True,
#         )
#         self.frames.append(fig)
#         plt.close(fig)

#     def save_gif(self, filename: str, fps: int = 2):
#         if not self.frames:
#             print("No frames to save. Set render_mode='capture'.")
#             return

#         images = []
#         for fig in self.frames:
#             fig.canvas.draw()
#             # Matplotlib >=3.8: FigureCanvasAgg no longer exposes tostring_rgb().
#             # Use buffer_rgba() (RGBA) and drop alpha for GIF frames.
#             buf = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4) uint8
#             image = buf[..., :3].copy()  # (H, W, 3)
#             images.append(image)
#             plt.close(fig)

#         imageio.mimsave(filename, images, fps=fps)


# # Backwards-compatible name used by your agents
# TetrisGymEnv = TetrisGym
