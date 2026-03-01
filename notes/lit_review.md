# Diffusion Planning for Tetris: Literature-to-Implementation Mapping

## Paper -> Implementation Mapping
| Core idea | How it maps to our repo | What ablation to run |
|---|---|---|
| Diffuser: plan by denoising trajectories under state conditioning | `diffusion/diffusion_model_updated.py` already denoises discrete `(rot, x)` sequences conditioned on board + piece IDs; `diffusion/diffusion_planner_updated.py` turns samples into action candidates | Compare `sample_steps` and horizon `H` to test whether stronger denoising improves score or just latency |
| MaskGIT: iterative masked-token refinement with confidence-based unmasking | `maskgit_sample(...)` in `diffusion_model_updated.py` follows iterative refinement with progressive locking | Swap confidence schedule (linear vs cosine lock rate) and measure score/runtime |
| Diffusion MPC (D-MPC): diffusion proposals + MPC re-ranking | `DiffusionMPCPlanner.plan(...)` samples `K` sequences then simulates + reranks | Reranker comparison: heuristic vs DQN critic, then invalid handling none vs penalize vs resample |
| AdaptDiffuser-style adaptive/self-improving generation | Current pipeline supports offline data generation + retraining (`collect_diffusion_dataset_updated.py`, `train_diffusion_updated.py`) | Add loop: evaluate planner -> keep top trajectories -> finetune denoiser; measure iteration gain |
| Diffusion-for-RL survey taxonomy: planning, policy generation, value guidance, conditioning | We already sit in "planning via generated candidates + downstream critic" bucket | Add conditioning (return bucket / board difficulty) and compare control over score tails |

## Diffuser (Planning with Diffusion)
- Treats planning as conditional generation of action/state trajectories.
- Denoising process can act as an implicit trajectory optimizer.
- Strong fit for offline RL where behavior data exists but online exploration is expensive.
- Naturally supports multi-modal action plans, useful when many placements are viable.
- In our repo, trajectory tokens are discrete `(rotation, x)` pairs rather than continuous controls.
- Conditioning signals already exist: board image-like tensor and current/next piece IDs.
- Sampling quality/compute tradeoff is controlled by `sample_steps` and temperature.
- MPC wrapper lets us reuse environment dynamics for candidate validation.
- Key engineering risk is invalid token proposals under piece-dependent legality.
- Practical ablation: hold model fixed and vary `K`/`H`/`S` to locate quality-latency knee point.
- Practical ablation: compare first-action only execution versus short receding-horizon execution.
- Expected takeaway: diffusion helps proposal diversity more than direct one-shot policy quality.

## MaskGIT (Iterative Masked Refinement)
- Generates sequences by repeatedly predicting masked tokens.
- Locks higher-confidence positions first, then fills uncertain positions later.
- Works well for discrete structured outputs, matching `(rot, x)` tokenization.
- Allows variable "computation at inference" by changing refinement iterations.
- In our code, all positions start masked and are progressively resolved.
- Confidence combines max probability of `rot` and `x` heads.
- Temperature acts as a diversity knob; high temperature broadens candidate set.
- Lock schedule currently grows linearly with iteration index.
- Failure mode: early overconfident but invalid choices can contaminate candidate quality.
- Improvement path: inject action-validity awareness during refinement or repair.
- Ablation: linear lock schedule vs slower early-lock (preserve optionality).
- Expected takeaway: iterative refinement is useful, but legality constraints dominate gains in Tetris.

## Diffusion MPC (D-MPC Pattern)
- Uses diffusion as candidate generator, not final controller.
- Evaluates candidates with an explicit model/critic before execution.
- Receding-horizon execution reduces compounding model error.
- Naturally supports plug-and-play rerankers.
- Our `DiffusionMPCPlanner` already simulates each candidate in a cloned env.
- Previous scoring used `heuristic_score_board`; new mode adds DQN value reranking.
- Candidate count `K` controls search breadth; horizon `H` controls lookahead depth.
- Runtime cost scales strongly with `K * H * sample_steps`.
- Invalid placements can waste candidate budget and bias search.
- New `invalid_handling` flag enables `none|penalize|resample`.
- Ablation: fixed compute budget (e.g., lower `K` when increasing `S`) to compare fairly.
- Expected takeaway: critic reranking should improve mean score with modest latency overhead.

## AdaptDiffuser (Adaptive / Self-Evolving Diffusion Planning)
- Emphasizes planner improvement beyond static offline training.
- Uses generated behavior and filtering to refine policy/planner iteratively.
- Bridges offline initialization with targeted online adaptation.
- Relevant because Tetris dynamics are deterministic but piece sequence introduces stochasticity.
- Our dataset collector already supports teacher rollout generation.
- Easy extension: replace heuristic teacher with DQN teacher for better target plans.
- Easy extension: collect planner rollouts, keep top percentile, retrain denoiser.
- Potentially reduces mismatch between training trajectories and planner execution distribution.
- Main risk: feedback loop can collapse diversity if only top episodes are kept.
- Mitigation: keep a replay mixture ratio (e.g., 70% old, 30% new top rollouts).
- Ablation: different keep ratios and retrain intervals.
- Expected takeaway: modest iterative data refresh should improve robustness and tail performance.

## Survey (Diffusion Models in RL / Decision Making)
- Organizes methods into behavior generation, planning, policy improvement, and guidance.
- Highlights tradeoff between sample quality and iterative compute.
- Shows value-guided generation as a recurring successful pattern.
- Distinguishes post-hoc reranking from in-loop guidance (different complexity/benefit).
- Notes conditioning mechanisms (goal, return, constraint) as key controllability tool.
- Legality/constraint handling is a first-order issue for discrete action domains.
- Emphasizes strong baselines and ablations to separate model gain from search gain.
- Recommends reporting both task score and wall-clock/runtime metrics.
- Encourages fixed-compute comparisons when changing sampling steps/candidate count.
- In our setup, this maps directly to `K/H/S/T` and reranker/constraint toggles.
- Related-work framing: our method is "offline diffusion proposal + online critic-constrained MPC".
- Expected takeaway: publishable contribution likely comes from robust constrained planning protocol, not raw architecture novelty.

