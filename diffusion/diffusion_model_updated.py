"""diffusion_model_updated.py

MaskGIT-style discrete denoising model for Tetris planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaskTokens:
    rot_mask: int
    rot_pad: int
    x_mask: int
    x_pad: int


class PlanDenoiser(nn.Module):
    """Conditional Transformer that predicts masked (rot,x) tokens."""

    def __init__(
        self,
        board_h: int,
        board_w: int,
        horizon: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        piece_emb_dim: int = 16,
    ):
        super().__init__()
        self.board_h = board_h
        self.board_w = board_w
        self.horizon = horizon

        self.rot_vocab = 6  # 0-3 + MASK(4) + PAD(5)
        self.x_vocab = board_w + 2  # 0..W-1 + MASK(W) + PAD(W+1)
        self.tokens = MaskTokens(rot_mask=4, rot_pad=5, x_mask=board_w, x_pad=board_w + 1)

        # condition encoder: board CNN + 2 piece embeddings
        self.piece_emb = nn.Embedding(7, piece_emb_dim)
        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, board_h, board_w)
            flat_dim = self.board_cnn(dummy).view(1, -1).size(1)
        cond_dim = flat_dim + 2 * piece_emb_dim
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # token embeddings
        self.rot_emb = nn.Embedding(self.rot_vocab, d_model)
        self.x_emb = nn.Embedding(self.x_vocab, d_model)
        self.pos_emb = nn.Embedding(horizon + 1, d_model)  # +1 for cond token

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.rot_head = nn.Linear(d_model, 4)
        self.x_head = nn.Linear(d_model, board_w)

    def encode_condition(self, board: torch.Tensor, curr_id: torch.Tensor, next_id: torch.Tensor) -> torch.Tensor:
        board = board.float()
        feat = self.board_cnn(board).view(board.size(0), -1)
        pieces = torch.cat([self.piece_emb(curr_id.long()), self.piece_emb(next_id.long())], dim=1)
        cond = torch.cat([feat, pieces], dim=1)
        return self.cond_proj(cond)  # (B,d)

    def forward(
        self,
        board: torch.Tensor,
        curr_id: torch.Tensor,
        next_id: torch.Tensor,
        rot_tokens: torch.Tensor,
        x_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits for rot and x.

        rot_logits: (B,H,4)
        x_logits:   (B,H,W)
        """
        B = board.size(0)
        cond = self.encode_condition(board, curr_id, next_id).unsqueeze(1)  # (B,1,d)
        tok = self.rot_emb(rot_tokens.long()) + self.x_emb(x_tokens.long())  # (B,H,d)
        tok = torch.cat([cond, tok], dim=1)  # (B,1+H,d)

        pos = self.pos_emb(torch.arange(self.horizon + 1, device=board.device)).unsqueeze(0)
        h = self.transformer(tok + pos, src_key_padding_mask=src_key_padding_mask)
        h_steps = h[:, 1:, :]
        return self.rot_head(h_steps), self.x_head(h_steps)


def masked_ce_loss(
    rot_logits: torch.Tensor,
    x_logits: torch.Tensor,
    rot_true: torch.Tensor,
    x_true: torch.Tensor,
    rot_masked: torch.Tensor,
    x_masked: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy on masked positions only.

    rot_masked/x_masked are boolean tensors of shape (B,H) indicating
    which positions were masked and should contribute to loss.
    """
    device = rot_logits.device
    B, H, _ = rot_logits.shape

    # rotation loss
    rot_loss = F.cross_entropy(
        rot_logits.reshape(B * H, 4),
        rot_true.reshape(B * H),
        reduction="none",
    ).reshape(B, H)
    rot_loss = rot_loss.masked_select(rot_masked).mean() if rot_masked.any() else torch.tensor(0.0, device=device)

    # x loss
    W = x_logits.shape[-1]
    x_loss = F.cross_entropy(
        x_logits.reshape(B * H, W),
        x_true.reshape(B * H),
        reduction="none",
    ).reshape(B, H)
    x_loss = x_loss.masked_select(x_masked).mean() if x_masked.any() else torch.tensor(0.0, device=device)

    return rot_loss + x_loss


@torch.no_grad()
def maskgit_sample(
    model: PlanDenoiser,
    board: torch.Tensor,
    curr_id: torch.Tensor,
    next_id: torch.Tensor,
    steps: int = 8,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterative refinement sampler.

    Starts from all MASK tokens, repeatedly predicts logits, samples tokens,
    and progressively locks in the most confident positions.

    Returns:
        rot_tokens: (B,H) in 0..3
        x_tokens:   (B,H) in 0..W-1
    """
    device = board.device
    B = board.size(0)
    H = model.horizon
    W = model.board_w

    rot = torch.full((B, H), model.tokens.rot_mask, dtype=torch.long, device=device)
    xs = torch.full((B, H), model.tokens.x_mask, dtype=torch.long, device=device)

    # which positions are still unknown
    unknown = torch.ones((B, H), dtype=torch.bool, device=device)

    for s in range(steps):
        rot_logits, x_logits = model(board, curr_id, next_id, rot, xs)

        rot_probs = F.softmax(rot_logits / max(temperature, 1e-6), dim=-1)  # (B,H,4)
        x_probs = F.softmax(x_logits / max(temperature, 1e-6), dim=-1)      # (B,H,W)

        # sample proposals for all positions
        rot_sample = torch.multinomial(rot_probs.reshape(B * H, 4), 1).reshape(B, H)
        x_sample = torch.multinomial(x_probs.reshape(B * H, W), 1).reshape(B, H)

        # confidence = max prob (rot) * max prob (x)
        rot_conf, _ = rot_probs.max(dim=-1)
        x_conf, _ = x_probs.max(dim=-1)
        conf = rot_conf * x_conf  # (B,H)

        # update unknown positions with samples
        rot = torch.where(unknown, rot_sample, rot)
        xs = torch.where(unknown, x_sample, xs)

        # decide how many positions to lock this round
        frac = (s + 1) / steps
        k = max(1, int(round(frac * H)))

        # lock top-k confident positions per batch item
        new_unknown = unknown.clone()
        for b in range(B):
            idx = torch.argsort(conf[b], descending=True)
            lock_idx = idx[:k]
            new_unknown[b, lock_idx] = False
        unknown = new_unknown

        if not unknown.any():
            break

    # ensure no MASK tokens remain
    rot = rot.clamp(0, 3)
    xs = xs.clamp(0, W - 1)
    return rot, xs
