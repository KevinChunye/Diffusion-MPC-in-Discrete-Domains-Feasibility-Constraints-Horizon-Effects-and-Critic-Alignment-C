import random
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PIECE2IDX = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}


def tensorize_obs(obs):
    """
    Raw (board, curr_id, next_id) -> 15-channel float32 tensor (c,h,w)
    C0: board;
    C1-7: one-hot for current piece;
    C8-14: one-hot for next piece
    """
    board, curr_piece, next_piece = obs
    board_h, board_w = board.shape

    board = board.astype(np.float32)[None, ...]             # (1,H,W)

    eye = np.eye(7, dtype=np.float32)
    curr_planes = np.repeat(eye[curr_piece][:, None, None], board_h, 1)
    curr_planes = np.repeat(curr_planes, board_w, 2)   # (7,H,W)

    next_planes = np.repeat(eye[next_piece][:, None, None], board_h, 1)
    next_planes = np.repeat(next_planes, board_w, 2)   # (7,H,W)

    return torch.from_numpy(np.concatenate([board, curr_planes, next_planes]))  # (15,H,W)

class DQNCNN(nn.Module):
    def __init__(self, n_actions, board_h, board_w):
        super().__init__()

        # Convolutional layers 
        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # padding to keep size
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)                   # 20×10 → 10×5
        )
        # compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, board_h, board_w)
            flat_dim = self.feat(dummy).view(1, -1).size(1)
        flat_dim += 14  # one-hot encoding of current and next pieces
        # fully connected head
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, n_actions)


    def forward(self, x):
        x = x.float()  # (B,15,H,W)
        board = x[:, :1]  # (B,1,H,W)
        piece = x[:, 1:, 0, 0]  # (B,14)
        conv = self.feat(board).view(x.size(0), -1)
        x = torch.cat([conv, piece], dim=1)
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

        self.rewards, self.scores = [], []  # for logging


    # --- observation and reward plugs ---
    def parse_obs(self, obs):
        return tensorize_obs(obs)  # torch tensor (15,H,W)

    def compute_reward(self, info, done):
        reward = 1 + info["lines_cleared"]**2 * self.board_width
        if done and info.get("game_over", True):
            reward -= 10
        return reward


    # --- internal helpers ---
    def _memorize(self, state, action, reward, next_state, done, next_valid_mask):
        """Store a transition.

        We also store a boolean mask of which action ids are valid in the *next*
        state, so TD targets can be computed with valid-action masking.
        """
        self.memory.append(
            (state.detach().cpu(), action, reward, next_state.detach().cpu(), done, next_valid_mask)
        )


    def _act(self, state, valid_action_ids):
        """Epsilon-greedy action selection using only valid actions"""
        if random.random() < self.eps:
            return random.choice(valid_action_ids)

        self.model.eval()  # put to evaluation mode for inference
        with torch.no_grad():  # no gradient computation to save time
            q_vals = self.model(state.unsqueeze(0).to(self.device)).squeeze(0)
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
        # organize batch into tensors. B: batch size, C: channel size, H: board height, W: board width
        states = torch.stack([s.to(self.device) for (s, _, _, _, _, _) in minibatch]) # [B, C, H, W]
        actions = torch.tensor([a for (_, a, _, _, _, _) in minibatch], dtype=torch.long, device=self.device)  # [B]
        rewards = torch.tensor([r for (_, _, r, _, _, _) in minibatch], dtype=torch.float32, device=self.device)  # [B]
        next_states = torch.stack([ns.to(self.device) for (_, _, _, ns, _, _) in minibatch])    # [B, C, H, W]
        dones = torch.tensor([d for (_, _, _, _, d, _) in minibatch], dtype=torch.float32, device=self.device)   # [B]
        next_valid_mask = torch.tensor(
            np.stack([m for (_, _, _, _, _, m) in minibatch]),
            dtype=torch.bool,
            device=self.device,
        )  # [B, num_actions]

        # Predicted Q(s, a), from main model
        q_pred = self.model(states)  # [B, num_actions], compute Q-values for all actions in each state
        q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B], .gather picks Q-value for actual action taken

        # Target Q-values, from frozen target model
        with torch.no_grad():
            # Single DQN
            # q_next = self.target_model(next_states)  # [B, num_actions], find all future q-values
            # q_next_max = q_next.max(dim=1)[0]  # [B], max Q-value for next state
            # q_target = rewards + self.gamma * q_next_max * (1 - dones)  # [B], if done target = rewards
            # Double DQN
            # Double DQN with valid-action masking
            next_q_online = self.model(next_states)
            next_q_online = next_q_online.masked_fill(~next_valid_mask, -1e9)
            next_actions = next_q_online.argmax(1, keepdim=True)

            next_q_target = self.target_model(next_states)
            next_q_target = next_q_target.masked_fill(~next_valid_mask, -1e9)
            q_next_max = next_q_target.gather(1, next_actions).squeeze(1)
            q_target = rewards + self.gamma * q_next_max * (1 - dones)


        #  Loss and Backprop
        loss = F.mse_loss(q_pred, q_target)

        # Updates the main model's weights 
        self.optimizer.zero_grad()
        loss.backward()  # back propagation
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # rescales gradients s.t. global l2-norm<=max_norm 
        self.optimizer.step()


    # --- train loop ---
    def train(self, env, episodes=10_000, max_steps=1_000):
        # one-time init that needs the env
        state_shape = (15, self.board_height, self.board_width)
        n_actions = len(env.full_action_space)

        if self.model is None:
            self.model = DQNCNN(n_actions, board_h=self.board_height, board_w=self.board_width).to(self.device)
            self.target_model = DQNCNN(n_actions, board_h=self.board_height, board_w=self.board_width).to(self.device)  # forzen model
            self.target_model.load_state_dict(self.model.state_dict())  # sync model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        pbar = tqdm(range(1, episodes+1), desc="DQN")  # pbar means progress bar
        for ep in pbar:
            raw_obs = env.reset()
            state = self.parse_obs(raw_obs).to(self.device)
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

                # mask of valid actions in next state for TD target masking
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
                pbar.set_description(f"DQN | mu_100={np.mean(self.rewards[-100:]):.1f} | epsilon={self.eps:.2f}")


    # --- save / load ---
    def save_agent(self, save_file):
        torch.save(
            dict(model=self.model.state_dict(),
                 target=self.target_model.state_dict(),
                 optimizer=self.optimizer.state_dict(),
                 epsilon=self.eps,
                 rewards=self.rewards,
                 scores=self.scores,
                 state_shape=tuple(self.parse_obs(
                     (np.zeros((self.board_height, self.board_width), dtype=np.uint8),0,0)  # fake state
                     ).shape),
                 num_actions=self.model.out.out_features,
                 memory=list(self.memory)), 
            save_file
        )

    def load_agent(self, load_file):
        checkpoint = torch.load(load_file, map_location=self.device)

        # rebuild nets if this object was freshly constructed
        if self.model is None:
            c, _, _ = checkpoint["state_shape"]
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


    def save_gif(self, save_path, max_steps=1000):
        """
        Run the agent in greedy mode (epsilon=0), save gameplay as GIF.
        Args:
            save_path: full path ending in .gif
            board_width, board_height: if None, use self.board_{width,height}
            max_steps: episode length cap
        """
        from TetrisGym import TetrisGym  # only import here to avoid circular issues

        # Capture mode
        env = TetrisGym(width=self.board_width, height=self.board_height, max_steps=max_steps, render_mode="capture")

        # Greedy play
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


    # --- evaluation ---
    def evaluate_agent(self, env, num_episodes=1000):
        """
        Evaluate the CNNAgent using a greedy policy (epsilon=0).

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
                valid_actions = env.get_valid_action_ids()
                if not valid_actions:
                    break

                # Greedy action selection
                self.model.eval()
                with torch.no_grad():
                    q_vals = self.model(state.unsqueeze(0).to(self.device)).squeeze(0)
                action = max(valid_actions, key=lambda a: q_vals[a].item())

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
