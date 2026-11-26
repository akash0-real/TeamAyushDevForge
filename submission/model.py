"""Custom spatial-memory neural architecture for DevForge Track 1 (DOOM).

Components:
  1. CircularRayEncoder: 360-degree circular 1D-CNN over 64 ego-centric rays.
  2. EntityAttentionEncoder: Masked multi-head self-attention + learned query
     pooling over up to 32 variable-length visible entities (safe zero-entity fallback).
  3. VitalsEncoder: MLP over normalized vitals, key flags, velocity & prev_action.
  4. GRU Recurrent Core: Persistent cross-room spatial memory (reset per map).
  5. Factored Policy Head: Mutually exclusive categorical heads mapped to MultiBinary(9)
     so conflicting buttons (e.g. forward+backward, turn_left+turn_right) never cancel out.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from preprocess import FACTORED_DIMS, MAP_CHANNELS, NUM_ENTITY_TYPES, NUM_RAY_TYPES


def _ortho_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CircularRayEncoder(nn.Module):
    def __init__(self, in_channels: int = 1 + NUM_RAY_TYPES, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 8, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )
        self.apply(_ortho_init)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        # rays: (B, 7, 64)
        x = self.conv(rays)
        return self.proj(x.flatten(1))


class EntityAttentionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, num_queries: int = 2, out_dim: int = 128):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 16)
        self.in_mlp = nn.Sequential(
            nn.Linear(5 + 16, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(num_queries * embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )
        self.apply(_ortho_init)

    def forward(self, ent_feat: torch.Tensor, ent_type: torch.Tensor, ent_mask: torch.Tensor) -> torch.Tensor:
        # ent_feat: (B, 32, 5), ent_type: (B, 32), ent_mask: (B, 32) bool
        bsz = ent_feat.shape[0]
        has_any = ent_mask.any(dim=1)  # (B,)

        x = torch.cat([ent_feat, self.type_emb(ent_type)], dim=-1)
        x = self.in_mlp(x)

        # Avoid all-True key_padding_mask (which yields NaN softmax) by unmasking slot 0 on empty rows
        safe_mask = ent_mask.clone()
        empty = ~has_any
        if empty.any():
            safe_mask[empty, 0] = True
        kpm = ~safe_mask  # True = ignored by PyTorch MHA

        sa_out, _ = self.self_attn(x, x, x, key_padding_mask=kpm, need_weights=False)
        x = self.norm1(x + sa_out)

        q = self.queries.expand(bsz, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=kpm, need_weights=False)
        out = self.out_proj(pooled.flatten(1))
        # Zero out rows that had 0 visible entities
        return out * has_any.unsqueeze(-1).to(out.dtype)


class EgoMapEncoder(nn.Module):
    """Tiny CNN over the EgoMap crops (2 x 6 channels, 15 x 15, heading-up) -> 64-dim."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * MAP_CHANNELS, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),  # 15 -> 7
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        return self.net(crops)


class DoomSpatialPolicy(nn.Module):
    """Recurrent spatial-attention actor-critic for ViZDoom."""

    def __init__(self, vitals_dim: int = 30, hidden_dim: int = 256, use_map: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_map = use_map
        self.map_enc = EgoMapEncoder(out_dim=64) if use_map else None
        self.ray_enc = CircularRayEncoder(out_dim=128)
        self.ent_enc = EntityAttentionEncoder(out_dim=128)
        self.vit_enc = nn.Sequential(
            nn.Linear(vitals_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.vit_enc.apply(_ortho_init)

        self.gru = nn.GRU(128 + 128 + 64 + (64 if use_map else 0), hidden_dim, num_layers=1)
        self.post_norm = nn.LayerNorm(hidden_dim)

        self.actor = nn.Linear(hidden_dim, sum(FACTORED_DIMS))
        self.critic = nn.Linear(hidden_dim, 1)
        _ortho_init(self.actor, gain=0.01)
        _ortho_init(self.critic, gain=1.0)
        with torch.no_grad():  # prior: start out mostly walking forward instead of jittering in place
            self.actor.bias[0:3] = torch.tensor([0.0, 1.0, -1.0])  # move: none / fwd / back

    def initial_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def encode(self, obs_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        r = self.ray_enc(obs_batch["rays"])
        e = self.ent_enc(obs_batch["ent_feat"], obs_batch["ent_type"], obs_batch["ent_mask"])
        v = self.vit_enc(obs_batch["vitals"])
        if self.map_enc is not None:
            return torch.cat([r, e, v, self.map_enc(obs_batch["map"])], dim=-1)
        return torch.cat([r, e, v], dim=-1)

    def forward_step(
        self, obs_batch: dict[str, torch.Tensor], h: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Single-step forward: obs fields have shape (B, ...), h has shape (1, B, H)."""
        feat = self.encode(obs_batch).unsqueeze(0)  # (1, B, 320)
        out, next_h = self.gru(feat, h)
        z = self.post_norm(out.squeeze(0))
        logits = list(torch.split(self.actor(z), FACTORED_DIMS, dim=-1))
        value = self.critic(z).squeeze(-1)
        return logits, value, next_h

    def forward_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],
        h0: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Truncated BPTT forward over (T, B, ...) resetting hidden state at done boundaries."""
        T, B = dones.shape
        flat_obs = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_seq.items()}
        feats = self.encode(flat_obs).reshape(T, B, -1)

        # dones[t] == 1 means an episode ended BEFORE step t (reset h before processing step t). Run one cuDNN GRU
        # call per stretch between such steps instead of one call per step: identical math, far fewer launches.
        cuts = (torch.nonzero(dones[1:].any(dim=1)).flatten() + 1).tolist()
        outputs = []
        h = h0
        for a, b in zip([0] + cuts, cuts + [T]):
            h = h * (1.0 - dones[a]).view(1, B, 1)
            out_ab, h = self.gru(feats[a:b], h)
            outputs.append(out_ab)
        out = self.post_norm(torch.cat(outputs, dim=0))  # (T, B, H)
        logits = list(torch.split(self.actor(out), FACTORED_DIMS, dim=-1))
        values = self.critic(out).squeeze(-1)  # (T, B)
        return logits, values, h

    @staticmethod
    def sample_and_logprob(
        logits: list[torch.Tensor], actions: torch.Tensor | None = None, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dists = [Categorical(logits=lg) for lg in logits]
        if actions is None:
            if deterministic:
                acts = [lg.argmax(dim=-1) for lg in logits]
            else:
                acts = [d.sample() for d in dists]
            actions = torch.stack(acts, dim=-1)
        log_prob = sum(d.log_prob(actions[..., i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return actions, log_prob, entropy


def load_policy(path, device) -> "DoomSpatialPolicy":
    """Build the right model for a checkpoint (with/without EgoMap) and load its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = DoomSpatialPolicy(
        use_map=any(k.startswith("map_enc.") for k in state),
        hidden_dim=int(state["gru.weight_hh_l0"].shape[1]),  # GRU size comes from the checkpoint (256 or larger)
    ).to(device)
    model.load_state_dict(state)
    return model


def warm_start_map_model(model: "DoomSpatialPolicy", base_state: dict) -> None:
    """Load a no-map checkpoint into a map model: shared layers copied, the GRU's extra map inputs start at zero."""
    own = model.state_dict()
    for k, v in base_state.items():
        if k == "gru.weight_ih_l0" and own[k].shape != v.shape:
            w = torch.zeros_like(own[k])
            w[:, : v.shape[1]] = v
            own[k] = w
        elif k in own and own[k].shape == v.shape:
            own[k] = v
    model.load_state_dict(own)
