"""diffusion_planner_updated.py

Diffusion-style MPC planner for Tetris.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_model_updated import PlanDenoiser, maskgit_sample
from diffusion_utils_updated import heuristic_score_board, tokens_to_action_seq, valid_placement_mask


@dataclass
class PlannerCfg:
    horizon: int = 5
    num_candidates: int = 64
    sample_steps: int = 8
    temperature: float = 1.0
    sampling_constraints: str = "none"  # none | mask_logits
    rerank_mode: str = "heuristic"      # heuristic | dqn | hybrid
    critic_weight: float = 0.0          # used when rerank_mode == "hybrid"
    invalid_handling: str = "none"      # none | penalize | resample
    invalid_penalty: float = 1e6
    resample_retries: int = 3


class DQNCritic:
    """Load a trained DQN checkpoint and expose Q/V scorers."""

    def __init__(self, ckpt_path: str, board_h: int, board_w: int, device: torch.device):
        try:
            from agent.cnn_dqn_updated import DQNCNN
        except ModuleNotFoundError:
            import sys
            from pathlib import Path

            repo_root = Path(__file__).resolve().parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from agent.cnn_dqn_updated import DQNCNN

        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        n_actions = int(ckpt.get("num_actions", 4 * board_w))
        self.model = DQNCNN(n_actions=n_actions, board_h=board_h, board_w=board_w).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def q_values(self, board: np.ndarray, curr_id: int, next_id: int) -> torch.Tensor:
        board_t = torch.from_numpy(board.astype(np.float32)[None, None, ...]).to(self.device)
        curr_t = torch.tensor([int(curr_id)], dtype=torch.long, device=self.device)
        next_t = torch.tensor([int(next_id)], dtype=torch.long, device=self.device)
        return self.model((board_t, curr_t, next_t)).squeeze(0)

    @torch.no_grad()
    def value(self, board: np.ndarray, curr_id: int, next_id: int, valid_action_ids: List[int]) -> float:
        if not valid_action_ids:
            return -1e18
        q = self.q_values(board, curr_id, next_id)
        valid_idx = torch.tensor(valid_action_ids, dtype=torch.long, device=self.device)
        return float(q[valid_idx].max().item())


class DiffusionMPCPlanner:
    def __init__(
        self,
        model: PlanDenoiser,
        cfg: PlannerCfg,
        device: torch.device,
        critic: Optional[DQNCritic] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.critic = critic
        self.last_plan_stats = {
            "masked_fraction": 0.0,
            "selected_candidate_invalid_count": 0,
            "best_candidate_rollout_score": 0.0,
            "chosen_rollout_score": 0.0,
            "regret": 0.0,
        }

    def _fallback_action(self, env) -> int:
        valid = env.get_valid_action_ids()
        return valid[0] if valid else 0

    def _score_board(self, sim_env) -> float:
        if self.cfg.rerank_mode == "heuristic":
            return heuristic_score_board(sim_env.game.board)

        if self.cfg.rerank_mode in {"dqn", "hybrid"}:
            if self.critic is None:
                raise ValueError("rerank_mode='dqn'/'hybrid' requires a loaded critic")
            obs = sim_env._obs()
            valid = sim_env.get_valid_action_ids()
            return self.critic.value(obs.board, obs.curr_id, obs.next_id, valid)

        raise ValueError(f"Unknown rerank_mode: {self.cfg.rerank_mode}")

    def _repair_action(self, sim_env, attempts: int) -> Optional[Tuple[int, Tuple[int, int]]]:
        for _ in range(max(1, attempts)):
            valid = sim_env.get_valid_action_ids()
            if not valid:
                return None
            aid = int(np.random.choice(valid))
            return aid, sim_env.id_to_action[aid]
        return None

    def _score_candidate(self, env, seq: List[Tuple[int, int]]) -> Tuple[float, float, List[Tuple[int, int]], int, int]:
        sim_env = copy.deepcopy(env)
        rollout: List[Tuple[int, int]] = []
        invalid_count = 0

        for (r, x) in seq:
            action = (int(r), int(x))
            aid = sim_env.action_to_id.get(action)
            valid_now = set(sim_env.get_valid_action_ids())
            is_invalid = (aid is None) or (aid not in valid_now)

            if is_invalid:
                invalid_count += 1
                if self.cfg.invalid_handling == "resample":
                    repaired = self._repair_action(sim_env, attempts=self.cfg.resample_retries)
                    if repaired is None:
                        break
                    aid, action = repaired
                else:
                    break

            rollout.append(action)
            _, _, done, _ = sim_env.step(aid)
            if done:
                break

        if invalid_count > 0 and self.cfg.invalid_handling == "penalize":
            score_rerank = -abs(float(self.cfg.invalid_penalty)) - invalid_count
        else:
            score_rerank = self._score_board(sim_env)
        score_rollout = heuristic_score_board(sim_env.game.board)

        if rollout and rollout[0] in env.action_to_id:
            first_aid = env.action_to_id[rollout[0]]
        else:
            first_aid = self._fallback_action(env)
        return float(score_rerank), float(score_rollout), rollout, int(first_aid), int(invalid_count)

    def _sample_one_mask_logits(self, env) -> Tuple[List[Tuple[int, int]], float]:
        """Sample one candidate with logits masked over flattened (rot,x) tokens."""
        sim_env = copy.deepcopy(env)
        H = int(self.cfg.horizon)
        W = int(sim_env.game.width)

        rot_tokens = torch.full((1, H), self.model.tokens.rot_mask, dtype=torch.long, device=self.device)
        x_tokens = torch.full((1, H), self.model.tokens.x_mask, dtype=torch.long, device=self.device)
        seq: List[Tuple[int, int]] = []
        masked_fracs: List[float] = []

        for h in range(H):
            if sim_env.game.game_over:
                break

            obs = sim_env._obs()
            board_t = torch.from_numpy(obs.board.astype(np.float32)[None, None, ...]).to(self.device)
            curr_t = torch.tensor([int(obs.curr_id)], dtype=torch.long, device=self.device)
            next_t = torch.tensor([int(obs.next_id)], dtype=torch.long, device=self.device)

            rot_logits, x_logits = self.model(board_t, curr_t, next_t, rot_tokens, x_tokens)
            flat_logits = (rot_logits[0, h][:, None] + x_logits[0, h][None, :]).reshape(4 * W)

            valid_np = valid_placement_mask(obs.board, obs.curr_id)
            valid = torch.from_numpy(valid_np.astype(bool)).to(self.device)
            masked_fracs.append(float((~valid_np).mean()))
            if not bool(valid.any().item()):
                break

            masked_logits = flat_logits.clone()
            masked_logits[~valid] = -1e9
            probs = F.softmax(masked_logits / max(float(self.cfg.temperature), 1e-6), dim=0)
            flat = int(torch.multinomial(probs, 1).item())

            rot_idx = int(flat // W)
            xpos = int(flat % W)
            rot_tokens[0, h] = rot_idx
            x_tokens[0, h] = xpos
            seq.append((rot_idx, xpos))

            aid = sim_env.action_to_id.get((rot_idx, xpos))
            valid_ids = set(sim_env.get_valid_action_ids())
            if aid is None or aid not in valid_ids:
                break
            _, _, done, _ = sim_env.step(aid)
            if done:
                break

        mean_masked = float(np.mean(masked_fracs)) if masked_fracs else 0.0
        return seq, mean_masked

    def _sample_candidates(self, env, board: np.ndarray, curr_id: int, next_id: int) -> Tuple[List[List[Tuple[int, int]]], float]:
        if self.cfg.sampling_constraints == "none":
            board_t = torch.from_numpy(board.astype(np.float32)[None, None, ...]).to(self.device)
            curr_t = torch.tensor([int(curr_id)], dtype=torch.long, device=self.device)
            next_t = torch.tensor([int(next_id)], dtype=torch.long, device=self.device)
            B = int(self.cfg.num_candidates)
            board_t = board_t.repeat(B, 1, 1, 1)
            curr_t = curr_t.repeat(B)
            next_t = next_t.repeat(B)

            rot_tok, x_tok = maskgit_sample(
                self.model,
                board_t,
                curr_t,
                next_t,
                steps=self.cfg.sample_steps,
                temperature=self.cfg.temperature,
            )
            rot_np = rot_tok.detach().cpu().numpy()
            x_np = x_tok.detach().cpu().numpy()
            seqs = [tokens_to_action_seq(rot_np[i], x_np[i]) for i in range(B)]
            return seqs, 0.0

        if self.cfg.sampling_constraints == "mask_logits":
            seqs: List[List[Tuple[int, int]]] = []
            masked_vals: List[float] = []
            for _ in range(int(self.cfg.num_candidates)):
                seq, masked_frac = self._sample_one_mask_logits(env)
                seqs.append(seq)
                masked_vals.append(masked_frac)
            return seqs, (float(np.mean(masked_vals)) if masked_vals else 0.0)

        raise ValueError(f"Unknown sampling_constraints={self.cfg.sampling_constraints}")

    @torch.no_grad()
    def plan(self, env, obs) -> Tuple[int, List[Tuple[int, int]], float]:
        board, curr_id, next_id = obs
        candidates, mean_masked_fraction = self._sample_candidates(env, board, int(curr_id), int(next_id))

        # Track best possible rollout among candidates (for regret)
        best_candidate_rollout_score = -1e18

        # Default chosen
        best_seq: List[Tuple[int, int]] = []
        best_action_id = self._fallback_action(env)
        best_invalid = 0
        chosen_rollout_score = -1e18
        best_score = -1e18  # returned score (meaning depends on mode)

        if self.cfg.rerank_mode != "hybrid":
            # Existing behavior: choose candidate with max "score" (heuristic or dqn score_rerank)
            for seq in candidates:
                score, rollout_score, rollout, first_aid, invalid_count = self._score_candidate(env, seq)

                if rollout_score > best_candidate_rollout_score:
                    best_candidate_rollout_score = rollout_score

                if score > best_score:
                    best_score = score
                    best_seq = rollout
                    best_action_id = first_aid
                    best_invalid = invalid_count
                    chosen_rollout_score = rollout_score

        else:
            # HYBRID: choose by rollout_score + alpha * zscore(dqn_value)
            if self.critic is None:
                raise ValueError("rerank_mode='hybrid' requires a loaded critic")

            alpha = float(self.cfg.critic_weight)

            cand_rollout_scores: List[float] = []
            cand_dqn_scores: List[float] = []
            cand_rollouts: List[List[Tuple[int, int]]] = []
            cand_first_aids: List[int] = []
            cand_invalids: List[int] = []

            for seq in candidates:
                dqn_score, rollout_score, rollout, first_aid, invalid_count = self._score_candidate(env, seq)

                cand_rollout_scores.append(float(rollout_score))
                cand_dqn_scores.append(float(dqn_score))
                cand_rollouts.append(rollout)
                cand_first_aids.append(int(first_aid))
                cand_invalids.append(int(invalid_count))

                if rollout_score > best_candidate_rollout_score:
                    best_candidate_rollout_score = rollout_score

            dqn_arr = np.asarray(cand_dqn_scores, dtype=np.float32)
            if dqn_arr.size:
                mu = float(dqn_arr.mean())
                sd = float(dqn_arr.std())
            else:
                mu, sd = 0.0, 0.0

            if sd > 1e-6:
                dqn_z = (dqn_arr - mu) / sd
            else:
                dqn_z = np.zeros_like(dqn_arr)

            roll_arr = np.asarray(cand_rollout_scores, dtype=np.float32)
            hybrid_scores = roll_arr + float(alpha) * dqn_z

            chosen_idx = int(hybrid_scores.argmax()) if hybrid_scores.size else 0

            best_seq = cand_rollouts[chosen_idx]
            best_action_id = cand_first_aids[chosen_idx]
            best_invalid = cand_invalids[chosen_idx]
            chosen_rollout_score = float(cand_rollout_scores[chosen_idx])
            best_score = float(hybrid_scores[chosen_idx])

        if best_candidate_rollout_score <= -1e17:
            best_candidate_rollout_score = 0.0
        if chosen_rollout_score <= -1e17:
            chosen_rollout_score = 0.0

        regret = float(best_candidate_rollout_score - chosen_rollout_score)

        self.last_plan_stats = {
            "masked_fraction": float(mean_masked_fraction),
            "selected_candidate_invalid_count": int(best_invalid),
            "best_candidate_rollout_score": float(best_candidate_rollout_score),
            "chosen_rollout_score": float(chosen_rollout_score),
            "regret": float(regret),
        }
        return int(best_action_id), best_seq, float(best_score)