import random
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PIECE2IDX = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}


def flatten_obs(obs):
    """
    Raw (board, curr_id, next_id) -> flattened board and one-hot curr and next piece
    """
    board, curr, next_ = obs
    flat_board = board.flatten().astype(np.float32)  # H*W
    one = np.eye(7, dtype=np.float32)
    return np.concatenate([flat_board, one[curr], one[next_]])  # H*W + 2*7 dimensional



class DQNMLP(nn.Module):
    """Simple 2-hidden-layer MLP"""
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, n_actions)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # (B, n_actions)


class MLPAgent:
    def __init__(self, board_width, board_height,
                 alpha=0.001, gamma=0.9,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.995,
                 memory_size=10_000, batch_size=128, target_sync=64,
                 device=None):

        self.board_width, self.board_height = board_width, board_height
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.memory_size, self.batch_size, self.target_sync= memory_size, batch_size, target_sync

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-initialised after env is supplied
        self.model = self.target_model = self.optimizer = None
        self.memory = deque(maxlen=memory_size)

        self.rewards, self.scores = [], []  # for logging

    # --- observation / reward plugs ---
    def parse_obs(self, obs):
        board = obs[0]
        flat_obs = flatten_obs(obs)  # (H*W+14, )

        # height features
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
            max_height, min_height, total_height,
            max_bumpiness, total_bumpiness,
            total_holes
        ], dtype=np.float32)

        return torch.from_numpy(np.concatenate([flat_obs, features]))


    def compute_reward(self, info, done):
        reward = 1 + info["lines_cleared"]**2 * self.board_width
        if done and info.get("game_over", True):
            reward -= 10
        return reward

    # --- internal helpers ---
    def _memorize(self, state, action, reward, next_state, done, next_valid_mask):
        """Store a transition along with valid-action mask for the next state."""
        self.memory.append((state.detach().cpu(), action, reward, next_state.detach().cpu(), done, next_valid_mask))

    def _act(self, state, valid_action_ids):
        """Epsilon-greedy"""
        if random.random() < self.eps:
            return random.choice(valid_action_ids)

        self.model.eval()
        with torch.no_grad():
            q_vals = self.model(state.unsqueeze(0).to(self.device)).squeeze(0)
        valid_q  = q_vals[valid_action_ids]
        return valid_action_ids[int(torch.argmax(valid_q).item())]

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        # organize batch into tensors
        states = torch.stack([s.to(self.device) for (s, _, _, _, _, _) in minibatch])
        actions = torch.tensor([a for (_, a, _, _, _, _) in minibatch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([r for (_, _, r, _, _, _) in minibatch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([ns.to(self.device) for (_, _, _, ns, _, _) in minibatch])
        dones = torch.tensor([d for (_, _, _, _, d, _) in minibatch], dtype=torch.float32, device=self.device)
        next_valid_mask = torch.tensor(
            np.stack([m for (_, _, _, _, _, m) in minibatch]),
            dtype=torch.bool,
            device=self.device,
        )

        # Predicted Q(s, a), from main model
        q_pred = self.model(states)  # [B, num_actions], compute Q-values for all actions in each state
        q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B], .gather picks Q-value for actual action taken

        with torch.no_grad():
            # Single DQN with valid-action masking
            q_next_all = self.target_model(next_states)
            q_next_all = q_next_all.masked_fill(~next_valid_mask, -1e9)
            q_next = q_next_all.max(dim=1)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # rescale gradient
        self.optimizer.step()



    # --- training loop ---
    def train(self, env, episodes=10_000, max_steps=1_000):
        input_dim  = self.parse_obs((np.zeros((self.board_height, self.board_width), dtype=np.uint8),0,0)).numel() # get an example observation, find number of elements
        n_actions  = len(env.full_action_space)

        if self.model is None:
            self.model = DQNMLP(input_dim, n_actions).to(self.device)
            self.target_model = DQNMLP(input_dim, n_actions).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        pbar = tqdm(range(1, episodes+1), desc="MLP")
        for ep in pbar:
            obs  = env.reset()
            state = self.parse_obs(obs).to(self.device)
            total_reward = 0.0
            done = False
            step = 0

            while not done and step < max_steps:
                valid_action_ids = env.get_valid_action_ids()
                if not valid_action_ids:
                    break

                action = self._act(state, valid_action_ids)
                next_raw_obs, _, done, info = env.step(action)
                reward = self.compute_reward(info, done)
                total_reward += reward

                next_state = self.parse_obs(next_raw_obs).to(self.device)

                next_valid_mask = env.get_valid_actions_mask()

                self._memorize(state, action, reward, next_state, done, next_valid_mask)
                self._replay()

                state = next_state
                step += 1

            if ep % self.target_sync == 0:  # sync frozen network
                self.target_model.load_state_dict(self.model.state_dict())

            self.rewards.append(total_reward)
            self.scores.append(env.game.score)

            if self.eps > self.eps_min:
                self.eps *= self.eps_decay

            if len(self.rewards) >= 100:
                pbar.set_description(f"MLP | mu_100={np.mean(self.rewards[-100:]):.1f} | epsilon={self.eps:.2f}")

    # --- save / load ---
    def save_agent(self, save_file):
        torch.save(dict(model=self.model.state_dict(),
                        target=self.target_model.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        epsilon=self.eps,
                        rewards=self.rewards,
                        scores=self.scores,
                        input_dim = self.parse_obs(
                            (np.zeros((self.board_height, self.board_width), dtype=np.uint8), 0, 0)
                            ).numel(),
                        num_actions=self.model.out.out_features,
                        memory=list(self.memory)),
                   save_file)

    def load_agent(self, load_file):
        checkpoint = torch.load(load_file, map_location=self.device)
        if self.model is None:
            input_dim = checkpoint["input_dim"]
            n_act = checkpoint["num_actions"]
            self.model = DQNMLP(input_dim, n_act).to(self.device)
            self.target_model = DQNMLP(input_dim, n_act).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.memory = deque(checkpoint["memory"], maxlen=self.memory_size)

        self.eps = checkpoint["epsilon"]
        self.rewards = checkpoint["rewards"]
        self.scores = checkpoint["scores"]


    def save_gif(self, save_path, max_steps=1000):
        from TetrisGym import TetrisGym

        env = TetrisGym(width=self.board_width, height=self.board_height, max_steps=max_steps, render_mode="capture")

        # greedy play
        obs = env.reset()
        state = self.parse_obs(obs).to(self.device)
        done = False
        while not done:
            valid_action_ids = env.get_valid_action_ids()
            if not valid_action_ids:
                break

            # pick best action, always greedy
            self.model.eval()
            with torch.no_grad():
                q_vals = self.model(state.unsqueeze(0).to(self.device)).squeeze(0)
            best_action = max(valid_action_ids, key=lambda a: q_vals[a].item())

            obs, _, done, _ = env.step(best_action)
            state = self.parse_obs(obs).to(self.device)

        # Save GIF
        env.save_gif(save_path)
        print(f"Saved gameplay GIF to: {save_path}")

    def evaluate_agent(self, env, num_episodes=1000):
        """
        Evaluate the MLPAgent using a greedy policy (epsilon=0).

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
            state = self.parse_obs(obs).to(self.device)

            total_reward = 0.0
            done = False
            steps = 0

            while not done:
                valid_action_ids = env.get_valid_action_ids()
                if not valid_action_ids:
                    break

                self.model.eval()
                with torch.no_grad():
                    q_vals = self.model(state.unsqueeze(0).to(self.device)).squeeze(0)
                action = max(valid_action_ids, key=lambda a: q_vals[a].item())

                next_obs, _, done, info = env.step(action)
                reward = self.compute_reward(info, done)
                total_reward += reward

                state = self.parse_obs(next_obs).to(self.device)
                steps += 1

            rewards.append(total_reward)
            scores.append(env.game.score)
            survival_lengths.append(steps)

        self.eps = original_eps  # restore epsilon
        return rewards, scores, survival_lengths
