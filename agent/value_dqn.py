import random
from collections import deque
import copy

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# --- tiny MLP that outputs a single V(s) ---
class ValueNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)  # scalar, V(s)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)  # (B,)


# --- helper functions to evaluate a board ---
def _clear_lines(board):
    """Return (#cleared, new_board) exactly as in TetrisGame"""
    _, w = board.shape

    full_mask = np.all(board == 1, axis=1)  # shape: (height,)
    num_cleared = np.count_nonzero(full_mask)
    new_board = board[~full_mask]  # Keep only the rows that are not full
    new_rows = np.zeros((num_cleared, w), dtype=board.dtype)  # stack empty rows on top
    new_board = np.vstack([new_rows, new_board])

    return num_cleared, new_board


def board_props(board):
    """[lines_cleared, holes, total_bumpiness, sum_height]"""
    lines, board = _clear_lines(board)

    h, _ = board.shape
    filled_mask = (board != 0)  # mask for filled cells
    filled_rows = np.argmax(filled_mask, axis=0)  # first filled row per column, i.e. max height of each col
    empty_cols = ~np.any(filled_mask, axis=0)
    filled_rows[empty_cols] = h  # if column empty, treat as full height from bottom

    col_heights = h - filled_rows  # vector of shape (w,)
    abs_height_diff = np.abs(np.diff(col_heights))

    max_height = float(np.max(col_heights))
    min_height = float(np.min(col_heights))
    total_height = float(np.sum(col_heights))
    max_bumpiness = float(np.max(abs_height_diff))
    total_bumpiness = float(np.sum(abs_height_diff))

    # hole feature: a hole is any empty cell with at least one filled cell above it
    accum_filled = np.maximum.accumulate(filled_mask, axis=0)  # cumulative of 'filled' down each column
    holes = (~filled_mask) & accum_filled  # empty cells that have a block above
    total_holes = float(np.sum(holes))

    features = np.array([
        lines,
        max_height, min_height, total_height,
        max_bumpiness, total_bumpiness,
        total_holes
    ], dtype=np.float32)

    return features



class ValueDQNAgent:
    """
    - Learns V(s) (expected future score)
    - Pick action to max V(s') over all next possible states s' (enumerated on-the-fly)
    """
    def __init__(self, board_width, board_height,
                 alpha=0.001, gamma=0.95,
                 eps_start=1.0, eps_min=0.1, eps_decay=0.995,
                 memory_size=10_000, batch_size=64,
                 device=None):

        self.board_width, self.board_height = board_width, board_height
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.memory_size, self.batch_size = memory_size, batch_size

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        feature_dim = board_props(np.zeros((self.board_height, self.board_width), dtype=np.uint8)).shape[0]  # use fake board to get input_dim
        self.model = ValueNet(feature_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.memory = deque(maxlen=memory_size)

        self.rewards, self.scores = [], []  # for logging


    # --- reward plug ---
    def compute_reward(self, info, done):
        reward = (10 / self.board_height) + info["lines_cleared"]**2 * self.board_width
        if done and info.get("game_over", True):
            reward -= 10
        return reward


    # --- core loop helpers ---
    def _enumerate_successors(self, env):
        """
        Return dict{action_id: state (aka feature vector)} for current piece
        Uses env.game's private helpers
        """
        game = env.game
        piece_type, rotations = game.current_piece
        successors = {}
        for rot_idx, piece in enumerate(rotations):
            piece_h, piece_w = piece.shape
            for x in range(game.width - piece_w + 1):
                y = game._find_drop_height(piece, x)  # a read-only method, is safe

                if y is None:
                    continue

                board = game.board.copy()
                sub_board = board[y:y+piece_h, x:x+piece_w]
                board[y:y+piece_h, x:x+piece_w] = sub_board + piece
                # board[y:y+piece_h, x:x+piece_w] |= piece  # |= is bit-wise OR, used to lock in a piece
                state = board_props(board)

                action = (rot_idx, x)
                successors[env.action_to_id[action]] = state
        return successors

    def _select_action(self, successors):
        """epsilon-greedy over successor values"""
        if not successors:  # no possible next states
            return None, None

        if random.random() < self.eps:
            action_id, state = random.choice(list(successors.items()))
        else:
            with torch.no_grad():
                states = torch.tensor(
                    np.stack(list(successors.values())), dtype=torch.float32, device=self.device
                    )
                values = self.model(states)  # (N,)
                best_idx = torch.argmax(values).item()  # find best action's index position
            action_id = list(successors.keys())[best_idx]
            state = successors[action_id]

        return action_id, state

    def _memorize(self, state, reward, next_state, done):
        self.memory.append((state.copy().astype(np.float32), reward, next_state.copy().astype(np.float32), done))

    # --- training ---
    def _replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states, rewards, next_states, dones = zip(*minibatch)
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)  # [B, F]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # [B]
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)  # [B, F]
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)  # [B]

        with torch.no_grad():
            target = rewards + self.gamma * self.model(next_states) * (1 - dones)

        pred = self.model(states)

        loss = F.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    # --- training loop ---
    def train(self, env, episodes=1_000, max_steps=1_000):
        pbar = tqdm(range(1, episodes + 1), desc='Value-DQN')
        for ep in pbar:
            env.reset()
            state = board_props(env.game.board)   # state before placing the first piece
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps:
                successors = self._enumerate_successors(env)
                action_id, next_state = self._select_action(successors)

                if action_id is None:  # no legal moves
                    break

                _, _, done, info = env.step(action_id)  # env.step returns (next_state, reward, done, info)
                reward = self.compute_reward(info, done)
                total_reward += reward

                self._memorize(state, reward, next_state, done)
                self._replay()

                state = next_state
                steps += 1

            self.rewards.append(total_reward)
            self.scores.append(env.game.score)

            # epsilon decay
            if self.eps > self.eps_min:
                self.eps *= self.eps_decay

            if len(self.rewards) >= 100:
                pbar.set_description(f"Value-DQN | mu_100={np.mean(self.rewards[-100:]):.1f} | epsilon={self.eps:.2f}")

# --- save / load ---
    def save_agent(self, save_file):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.eps,
                "rewards": self.rewards,
                "scores": self.scores,
                "memory": list(self.memory), # replay buffer
                "input_dim": self.model.fc1.in_features,
            },
            save_file)

    def load_agent(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if self.model is None:
            in_dim = checkpoint["input_dim"]
            self.model = ValueNet(in_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.memory = deque(checkpoint["memory"], maxlen=self.memory_size)

        self.eps = checkpoint["epsilon"]
        self.rewards = checkpoint["rewards"]
        self.scores = checkpoint["scores"]

    def save_gif(self, save_path, max_steps=1000):
        """
        Play one greedy game (epsilon=0) and save a GIF.
        """
        from TetrisGym import TetrisGym

        env = TetrisGym(width=self.board_width, height=self.board_height, max_steps=max_steps, render_mode="capture")

        # greedy play
        env.reset()
        done = False
        while not done:
            successors = self._enumerate_successors(env)
            if not successors:
                break

            # Pick best successor purely greedily
            with torch.no_grad():
                states = torch.tensor(np.stack(list(successors.values())), dtype=torch.float32, device=self.device)
                vals = self.model(states)
                best_id = list(successors.keys())[torch.argmax(vals).item()]

            _, _, done, _ = env.step(best_id)

        env.save_gif(save_path)
        print(f"Saved gameplay GIF to: {save_path}")


    # --- evaluation ---
    def evaluate_agent(self, env, num_episodes=1000):
        """
        Evaluate the ValueDQNAgent using a greedy policy (epsilon=0).

        Returns:
            tuple: (rewards, scores, survival_lengths)
        """
        rewards = []
        scores = []
        survival_lengths = []

        original_eps = self.eps
        self.eps = 0.0  # disable exploration

        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            obs = env.reset()
            state = board_props(env.game.board)

            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                successors = self._enumerate_successors(env)
                if not successors:
                    break

                with torch.no_grad():
                    states = torch.tensor(np.stack(list(successors.values())), dtype=torch.float32, device=self.device)
                    values = self.model(states)
                    best_id = list(successors.keys())[torch.argmax(values).item()]

                _, _, done, info = env.step(best_id)
                reward = self.compute_reward(info, done)
                total_reward += reward

                state = successors[best_id]
                steps += 1

            rewards.append(total_reward)
            scores.append(env.game.score)
            survival_lengths.append(steps)

        self.eps = original_eps  # restore original epsilon
        return rewards, scores, survival_lengths
