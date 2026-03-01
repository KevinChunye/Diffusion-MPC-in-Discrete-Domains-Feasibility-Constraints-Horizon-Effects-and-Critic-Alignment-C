"""beam_search_planner.py

Non-neural beam-search planner for Tetris placement actions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple

from diffusion.diffusion_utils_updated import heuristic_score_board


@dataclass
class BeamCfg:
    horizon: int = 3
    beam_width: int = 16


@dataclass
class BeamNode:
    env: object
    first_action_id: int
    score: float


class BeamSearchPlanner:
    def __init__(self, cfg: BeamCfg):
        self.cfg = cfg

    def plan(self, env) -> Tuple[int, float]:
        valid_root = env.get_valid_action_ids()
        if not valid_root:
            return 0, -1e18

        beam: List[BeamNode] = [BeamNode(env=copy.deepcopy(env), first_action_id=valid_root[0], score=-1e18)]

        for depth in range(max(1, self.cfg.horizon)):
            expanded: List[BeamNode] = []
            for node in beam:
                if bool(node.env.game.game_over):
                    continue
                valid_ids = node.env.get_valid_action_ids()
                if not valid_ids:
                    continue
                for aid in valid_ids:
                    sim = copy.deepcopy(node.env)
                    if bool(sim.game.game_over):
                        continue
                    try:
                        _, _, done, _ = sim.step(aid)
                    except RuntimeError:
                        continue
                    score = heuristic_score_board(sim.game.board)
                    first_aid = aid if depth == 0 else node.first_action_id
                    expanded.append(BeamNode(env=sim, first_action_id=first_aid, score=score))
                    if done:
                        continue

            if not expanded:
                break
            expanded.sort(key=lambda n: n.score, reverse=True)
            beam = expanded[: max(1, int(self.cfg.beam_width))]

        best = max(beam, key=lambda n: n.score)
        return int(best.first_action_id), float(best.score)
