import numpy as np, random, pickle
from collections import defaultdict, deque
from tqdm import tqdm

PIECE2IDX = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}

# --- helpers ---
def flatten_obs(obs):
    """Parse the raw observation"""
    board, curr, next_ = obs
    v_board = board.flatten().astype(np.float32)
    one = np.eye(7, dtype=np.float32)
    return np.concatenate([v_board, one[curr], one[next_]])



class TabularQAgent:
    def __init__(self, board_width, board_height,
                 alpha=.05, gamma=.5, 
                 eps_start=1.0, eps_min=.01, eps_decay=.99995):
        self.board_width, self.board_height = board_width, board_height
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.Q = None  # lazy init when env known
        self.rewards = []
        self.scores = []

    # --- obs and reward plugs ---
    def parse_obs(self, obs):
        """Parse the raw observation"""
        return flatten_obs(obs)

    def compute_reward(self, info, done):
        """Reward function"""
        reward = (10 / self.board_height) + info["lines_cleared"]**2 * self.board_width
        if done and info.get("game_over", True):
            reward -= 10
        return reward

    # --- helper functions ---
    @staticmethod
    def _key(vec):
        """Parse vector into look-up-able keys"""
        return vec.tobytes()

    def _act(self, key, valid_ids):
        if random.random() < self.eps:          # explore
            return random.choice(valid_ids)
        q = self.Q[key]
        return max(valid_ids, key=lambda a: q[a])

    def _learn(self, k, a, r, k1, next_valid, done):  # k1, a1 means next state keys and next action
        best_next = 0 if done else max(self.Q[k1][a1] for a1 in next_valid)
        self.Q[k][a] += self.alpha * (r + self.gamma*best_next - self.Q[k][a])

    # --- train loop ---
    def train(self, env, episodes=100_000, max_steps=100):
        if self.Q is None:
            self.Q = defaultdict(lambda: np.zeros(len(env.full_action_space)))

        pbar = tqdm(range(episodes), desc="Tab-Q")  # pbar means progress bar
        for _ in pbar:
            obs = env.reset()
            s = self.parse_obs(obs)
            key = self._key(s)
            total_reward = 0.0

            for _ in range(max_steps):
                valid_actions = env.get_valid_action_ids()
                if not valid_actions:
                    break
                a = self._act(key, valid_actions)

                next_obs, _, done, info = env.step(a)
                reward = self.compute_reward(info, done)
                total_reward += reward

                next_obs = self.parse_obs(next_obs)
                k1 = self._key(next_obs)
                self._learn(key, a, reward, k1, env.get_valid_action_ids(), done)

                if done: 
                    break
                key = k1

            self.rewards.append(total_reward)
            self.scores.append(env.game.score)
            if self.eps > self.eps_min:
                self.eps *= self.eps_decay
            if len(self.rewards) >= 100:
                pbar.set_description(f"Tab-Q | mu_100={np.mean(self.rewards[-100:]):.1f} | epsilon={self.eps:.2f}")


    # --- save and load agent ---
    def save_agent(self, save_file):
        with open(save_file,"wb") as file:
            pickle.dump(dict(Q=dict(self.Q), num_actions = len(next(iter(self.Q.values()))), epsilon=self.eps,
                             rewards=self.rewards, scores=self.scores), file)

    def load_agent(self, load_file):
        with open(load_file, "rb") as file:
            data = pickle.load(file)
        self.Q = defaultdict(lambda: np.zeros(data["num_actions"]), data["Q"])  # defaultdict for agent access
        self.eps = data["epsilon"]
        self.rewards = data["rewards"]
        self.score = data['scores']


    # --- evaluation ---
    def evaluate_agent(self, env, num_episodes=1000):
        """Evaluate the trained agent with greedy policy (no exploration)."""
        rewards = []
        scores = []
        survival_lengths = []

        original_eps = self.eps  # Save current epsilon
        self.eps = 0.0  # Disable exploration for evaluation

        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            obs = env.reset()
            state = self.parse_obs(obs)
            key = self._key(state)

            total_reward = 0.0
            step_count = 0
            done = False

            while not done:
                valid_actions = env.get_valid_action_ids()
                if not valid_actions:
                    break
                action = max(valid_actions, key=lambda a: self.Q[key][a])  # greedy action

                next_obs, _, done, info = env.step(action)
                reward = self.compute_reward(info, done)
                total_reward += reward

                next_state = self.parse_obs(next_obs)
                key = self._key(next_state)
                step_count += 1

            rewards.append(total_reward)
            scores.append(env.game.score)
            survival_lengths.append(step_count)

        self.eps = original_eps  # Restore original epsilon
        return rewards, scores, survival_lengths

