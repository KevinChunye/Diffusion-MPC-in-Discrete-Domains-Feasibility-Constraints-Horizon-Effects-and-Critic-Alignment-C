import random
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PIECE2IDX = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}


# -----------------------------
# Board feature helpers (reward shaping)
# -----------------------------
def _column_heights(board: np.ndarray) -> np.ndarray:
    """Return per-column heights (W,) where height counts filled cells from bottom to highest filled cell."""
    H, W = board.shape
    heights = np.zeros(W, dtype=np.int32)
    for x in range(W):
        filled = np.where(board[:, x] > 0)[0]
        heights[x] = 0 if filled.size == 0 else (H - int(filled[0]))
    return heights


def _count_holes(board: np.ndarray) -> int:
    """Holes = empty cells that have at least one filled cell above them in same column."""
    holes = 0
    H, W = board.shape
    for x in range(W):
        col = board[:, x]
        filled = np.where(col > 0)[0]
        if filled.size == 0:
            continue
        top = int(filled[0])
        holes += int(np.sum(col[top:] == 0))
    return int(holes)


def _bumpiness(heights: np.ndarray) -> int:
    return int(np.sum(np.abs(np.diff(heights))))


def _aggregate_height(heights: np.ndarray) -> int:
    return int(np.sum(heights))


def tensorize_obs(obs):
    """
    Upgrade A encoding:

    Raw (board, curr_id, next_id) -> (board_tensor, curr_id_tensor, next_id_tensor)
      - board_tensor: float32, shape (1,H,W)
      - curr_id_tensor: int64 scalar
      - next_id_tensor: int64 scalar
    """
    board, curr_piece, next_piece = obs
    board = torch.from_numpy(board.astype(np.float32)[None, ...])  # (1,H,W)
    curr_id = torch.tensor(int(curr_piece), dtype=torch.long)
    next_id = torch.tensor(int(next_piece), dtype=torch.long)
    return board, curr_id, next_id


class DQNCNN(nn.Module):
    """CNN Q-network with piece embeddings (Upgrade A).

    Input is a tuple (board, curr_id, next_id):
      - board: float tensor, (B,1,H,W)
      - curr_id: long tensor, (B,)
      - next_id: long tensor, (B,)
    """

    def __init__(self, n_actions, board_h, board_w, piece_emb_dim: int = 16):
        super().__init__()

        self.piece_emb = nn.Embedding(7, piece_emb_dim)

        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 20x10 -> 10x5
        )

        # compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, board_h, board_w)
            flat_dim = self.feat(dummy).view(1, -1).size(1)

        flat_dim += 2 * piece_emb_dim  # curr + next embeddings

        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, n_actions)

    def forward(self, x):
        # Support either a pre-packed tuple (board, curr_id, next_id)
        # or a dict with those keys.
        if isinstance(x, (tuple, list)):
            board, curr_id, next_id = x
        elif isinstance(x, dict):
            board, curr_id, next_id = x["board"], x["curr_id"], x["next_id"]
        else:
            raise TypeError(
                "DQNCNN expects input as (board, curr_id, next_id) or a dict with those keys"
            )

        board = board.float()
        curr_id = curr_id.long()
        next_id = next_id.long()

        conv = self.feat(board).view(board.size(0), -1)
        piece_feat = torch.cat([self.piece_emb(curr_id), self.piece_emb(next_id)], dim=1)

        x = torch.cat([conv, piece_feat], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class CNNAgent:
    def __init__(self, board_width, board_height,
                 alpha=0.001, gamma=0.9,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.995,
                 memory_size=10_000, batch_size=128, target_sync=64,
                 device=None):
        self.board_width, self.board_height = board_width, board_height

        # Hyperparameters
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.memory_size, self.batch_size, self.target_sync = memory_size, batch_size, target_sync

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.target_model = self.optimizer = None  # Filled after seeing the environment
        self.memory = deque(maxlen=memory_size)

        # for logging
        self.rewards, self.scores = [], []
        self.episode_lengths = []
        self.eps_history = []


        # --- reward shaping weights (tune later) ---
        self.w_lines = 1.0
        self.w_holes = 2.0
        self.w_agg_height = 0.05
        self.w_bump = 0.02
        self.topout_penalty = 10.0

        # --- training diagnostics ---
        self.loss_history = []  # per gradient step (from replay)
        self.episode_loss = []  # per episode mean loss


    # --- observation and reward plugs ---
    def parse_obs(self, obs):
        # (board, curr_id, next_id)
        return tensorize_obs(obs)

    def compute_reward(self, info, done):
        """
        Tetris-aware shaped reward.

        Requires:
          - info['pre_clear_board'] : board before lock+clear
          - info['post_clear_board']: board after lock+clear (add this in TetrisGame_updated.py)

        Components:
          + line clears (nonlinear table)
          - increase in holes
          - increase in aggregate height
          - increase in bumpiness
          - top-out penalty
        """
        lines = int(info.get("lines_cleared", 0))

        # Line-clear reward (less spiky than raw score; tune later)
        line_reward_table = {0: 0.0, 1: 2.0, 2: 6.0, 3: 15.0, 4: 40.0}
        r_lines = float(line_reward_table.get(lines, 10.0 * lines))

        pre = info.get("pre_clear_board", None)
        post = info.get("post_clear_board", None)

        if pre is None or post is None:
            reward = r_lines
        else:
            pre = np.asarray(pre, dtype=np.int8)
            post = np.asarray(post, dtype=np.int8)

            pre_h = _column_heights(pre)
            post_h = _column_heights(post)

            d_holes = _count_holes(post) - _count_holes(pre)
            d_agg = _aggregate_height(post_h) - _aggregate_height(pre_h)
            d_bump = _bumpiness(post_h) - _bumpiness(pre_h)

            reward = (
                self.w_lines * r_lines
                - self.w_holes * float(d_holes)
                - self.w_agg_height * float(d_agg)
                - self.w_bump * float(d_bump)
            )

        if done and info.get("game_over", True):
            reward -= float(self.topout_penalty)

        return float(reward)


    # --- internal helpers ---
    def _memorize(self, state, action, reward, next_state, done, next_valid_mask):
        """Store a transition.

        We also store a boolean mask of which action ids are valid in the *next*
        state, so TD targets can be computed with valid-action masking.
        """
        board, curr_id, next_id = state
        n_board, n_curr_id, n_next_id = next_state
        self.memory.append(
            (
                (board.detach().cpu(), curr_id.detach().cpu(), next_id.detach().cpu()),
                action,
                reward,
                (n_board.detach().cpu(), n_curr_id.detach().cpu(), n_next_id.detach().cpu()),
                done,
                next_valid_mask,
            )
        )


    def _act(self, state, valid_action_ids):
        """Epsilon-greedy action selection using only valid actions"""
        if random.random() < self.eps:
            return random.choice(valid_action_ids)

        self.model.eval()
        with torch.no_grad():
            board, curr_id, next_id = state
            q_vals = self.model(
                (board.unsqueeze(0).to(self.device),
                 curr_id.unsqueeze(0).to(self.device),
                 next_id.unsqueeze(0).to(self.device))
            ).squeeze(0)
        valid_q  = q_vals[valid_action_ids]
        return valid_action_ids[int(torch.argmax(valid_q).item())]


    def _replay(self):
        """
        Samples mini-batch of past experience from memory
        Use them to train NN by minimizing the squared error between predicted Q-values and target Q-values
        """
        if len(self.memory) < self.batch_size:
            return  # not enough samples yet

        minibatch = random.sample(self.memory, self.batch_size)

        state_boards = torch.stack([s[0].to(self.device) for (s, _, _, _, _, _) in minibatch])  # [B,1,H,W]
        state_curr = torch.stack([s[1].to(self.device) for (s, _, _, _, _, _) in minibatch])    # [B]
        state_next = torch.stack([s[2].to(self.device) for (s, _, _, _, _, _) in minibatch])    # [B]

        actions = torch.tensor([a for (_, a, _, _, _, _) in minibatch], dtype=torch.long, device=self.device)  # [B]
        rewards = torch.tensor([r for (_, _, r, _, _, _) in minibatch], dtype=torch.float32, device=self.device)  # [B]

        next_boards = torch.stack([ns[0].to(self.device) for (_, _, _, ns, _, _) in minibatch])  # [B,1,H,W]
        next_curr = torch.stack([ns[1].to(self.device) for (_, _, _, ns, _, _) in minibatch])    # [B]
        next_next = torch.stack([ns[2].to(self.device) for (_, _, _, ns, _, _) in minibatch])    # [B]

        dones = torch.tensor([d for (_, _, _, _, d, _) in minibatch], dtype=torch.float32, device=self.device)   # [B]
        next_valid_mask = torch.tensor(
            np.stack([m for (_, _, _, _, _, m) in minibatch]),
            dtype=torch.bool,
            device=self.device,
        )  # [B, num_actions]

        # Predicted Q(s, a), from main model
        q_pred = self.model((state_boards, state_curr, state_next))  # [B, num_actions]
        q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        # Target Q-values, from frozen target model
        with torch.no_grad():
            # Double DQN with valid-action masking
            next_q_online = self.model((next_boards, next_curr, next_next))
            next_q_online = next_q_online.masked_fill(~next_valid_mask, -1e9)
            next_actions = next_q_online.argmax(1, keepdim=True)

            next_q_target = self.target_model((next_boards, next_curr, next_next))
            next_q_target = next_q_target.masked_fill(~next_valid_mask, -1e9)
            q_next_max = next_q_target.gather(1, next_actions).squeeze(1)
            q_target = rewards + self.gamma * q_next_max * (1 - dones)

        loss = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.loss_history.append(float(loss.item()))
        return float(loss.item())


    # --- train loop ---
    def train(self, env, episodes=10_000, max_steps=1_000):
        n_actions = len(env.full_action_space)

        if self.model is None:
            self.model = DQNCNN(n_actions, board_h=self.board_height, board_w=self.board_width).to(self.device)
            self.target_model = DQNCNN(n_actions, board_h=self.board_height, board_w=self.board_width).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        pbar = tqdm(range(1, episodes+1), desc="DQN")
        for ep in pbar:
            raw_obs = env.reset()
            state = tuple(t.to(self.device) for t in self.parse_obs(raw_obs))
            total_reward = 0.0
            done = False
            step = 0

            loss_sum = 0.0
            loss_n = 0

            while not done and step < max_steps:
                valid_action_ids = env.get_valid_action_ids()
                if not valid_action_ids:
                    break

                action = self._act(state, valid_action_ids)
                next_raw_obs, _, done, info = env.step(action)
                # Attach post-move board so reward shaping can compute deltas.
                # (Env does not need to return it explicitly.)
                if isinstance(info, dict) and "post_clear_board" not in info:
                    info["post_clear_board"] = env.game.board.copy()
                reward = self.compute_reward(info, done)
                total_reward += reward

                next_state = tuple(t.to(self.device) for t in self.parse_obs(next_raw_obs))
                next_valid_mask = env.get_valid_actions_mask()

                self._memorize(state, action, reward, next_state, done, next_valid_mask)
                loss_val = self._replay()
                if loss_val is not None:
                    loss_sum += float(loss_val)
                    loss_n += 1

                state = next_state
                step += 1

            # --- episode-level logs ---
            self.episode_lengths.append(step)
            self.episode_loss.append(float(loss_sum / loss_n) if loss_n > 0 else 0.0)
            self.eps_history.append(float(self.eps))

            if ep % self.target_sync == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.rewards.append(total_reward)
            self.scores.append(env.game.score)

            if self.eps > self.eps_min:
                self.eps *= self.eps_decay

            if len(self.rewards) >= 100:
                pbar.set_description(f"DQN | mu_100={np.mean(self.rewards[-100:]):.1f} | epsilon={self.eps:.2f}")


    # --- save / load ---
    def save_agent(self, save_file):
        board_shape = (1, self.board_height, self.board_width)
        torch.save(
            dict(model=self.model.state_dict(),
                 target=self.target_model.state_dict(),
                 optimizer=self.optimizer.state_dict(),
                 epsilon=self.eps,
                 rewards=self.rewards,
                 scores=self.scores,
                 episode_lengths=self.episode_lengths,
                 eps_history=self.eps_history,
                 episode_loss=self.episode_loss,
                 loss_history=self.loss_history,
                 board_shape=board_shape,
                 num_actions=self.model.out.out_features,
                 memory=list(self.memory)),
            save_file
        )

    def load_agent(self, load_file):
        checkpoint = torch.load(load_file, map_location=self.device)

        if self.model is None:
            n_act = checkpoint["num_actions"]
            self.model = DQNCNN(n_act, board_h=self.board_height, board_w=self.board_width).to(self.device)
            self.target_model = DQNCNN(n_act, board_h=self.board_height, board_w=self.board_width).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.memory = deque(checkpoint["memory"], maxlen=self.memory_size)

        self.eps = checkpoint["epsilon"]
        self.rewards = checkpoint["rewards"]
        self.scores = checkpoint["scores"]
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        self.eps_history = checkpoint.get("eps_history", [])
        self.episode_loss = checkpoint.get("episode_loss", [])
        self.loss_history = checkpoint.get("loss_history", [])


    def save_gif(self, save_path, max_steps=1000):
        from TetrisGym_updated import TetrisGym  # updated env

        env = TetrisGym(width=self.board_width, height=self.board_height, max_steps=max_steps, render_mode="capture")

        obs = env.reset()
        state = tuple(t.to(self.device) for t in self.parse_obs(obs))
        done = False
        while not done:
            valid_action_ids = env.get_valid_action_ids()
            if not valid_action_ids:
                break

            self.model.eval()
            with torch.no_grad():
                board, curr_id, next_id = state
                q_vals = self.model(
                    (board.unsqueeze(0), curr_id.unsqueeze(0), next_id.unsqueeze(0))
                ).squeeze(0)
            best_action = max(valid_action_ids, key=lambda a: q_vals[a].item())

            obs, _, done, _ = env.step(best_action)
            state = tuple(t.to(self.device) for t in self.parse_obs(obs))

        env.save_gif(save_path)
        print(f"Saved gameplay GIF to: {save_path}")


    # --- evaluation ---
    def evaluate_agent(self, env, num_episodes=1000):
        rewards = []
        scores = []
        survival_lengths = []

        original_eps = self.eps
        self.eps = 0.0

        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            obs = env.reset()
            state = tuple(t.to(self.device) for t in self.parse_obs(obs))

            total_reward = 0.0
            done = False
            steps = 0

            while not done:
                valid_actions = env.get_valid_action_ids()
                if not valid_actions:
                    break

                self.model.eval()
                with torch.no_grad():
                    board, curr_id, next_id = state
                    q_vals = self.model(
                        (board.unsqueeze(0), curr_id.unsqueeze(0), next_id.unsqueeze(0))
                    ).squeeze(0)
                action = max(valid_actions, key=lambda a: q_vals[a].item())

                next_obs, _, done, info = env.step(action)
                if isinstance(info, dict) and "post_clear_board" not in info:
                    info["post_clear_board"] = env.game.board.copy()
                reward = self.compute_reward(info, done)
                total_reward += reward

                state = tuple(t.to(self.device) for t in self.parse_obs(next_obs))
                steps += 1

            rewards.append(total_reward)
            scores.append(env.game.score)
            survival_lengths.append(steps)

        self.eps = original_eps
        return rewards, scores, survival_lengths
