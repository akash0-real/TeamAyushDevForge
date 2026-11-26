"""Observation preprocessing for DoomAgent and training pipelines.

Converts raw `doom_env` observations (docs/CONTRACT.md) into normalized,
fixed-shape numpy/torch arrays strictly from ego-centric sensory inputs.
"""

from __future__ import annotations

import numpy as np

NUM_RAYS = 64
NUM_RAY_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12
NUM_VITALS = 17
NUM_ACTIONS = 9

# Factored action dimensions:
# [move_fb(3), strafe_lr(3), turn_lr(3), attack(2), use(2), next_weapon(2)]
FACTORED_DIMS = (3, 3, 3, 2, 2, 2)


def factored_to_multibinary(factored: np.ndarray) -> np.ndarray:
    """Convert factored discrete indices (..., 6) to MultiBinary(9) int8 array."""
    f = np.asarray(factored, dtype=np.int64)
    out = np.zeros((*f.shape[:-1], NUM_ACTIONS), dtype=np.int8)
    out[..., 0] = (f[..., 0] == 1)  # MOVE_FORWARD
    out[..., 1] = (f[..., 0] == 2)  # MOVE_BACKWARD
    out[..., 2] = (f[..., 1] == 1)  # MOVE_LEFT
    out[..., 3] = (f[..., 1] == 2)  # MOVE_RIGHT
    out[..., 4] = (f[..., 2] == 1)  # TURN_LEFT
    out[..., 5] = (f[..., 2] == 2)  # TURN_RIGHT
    out[..., 6] = (f[..., 3] == 1)  # ATTACK
    out[..., 7] = (f[..., 4] == 1)  # USE
    out[..., 8] = (f[..., 5] == 1)  # SELECT_NEXT_WEAPON
    return out


def preprocess_obs(obs: dict, prev_action: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Encode a single `doom_env` observation dict into clean float32/int64 arrays."""
    # 1. Rays: (7, 64) -> [distance, one-hot hit_type(6)]
    ray_dist = np.asarray(obs["rays"]["distance"], dtype=np.float32)
    ray_type = np.asarray(obs["rays"]["hit_type"], dtype=np.int64)
    ray_type = np.clip(ray_type, 0, NUM_RAY_TYPES - 1)
    rays = np.zeros((1 + NUM_RAY_TYPES, NUM_RAYS), dtype=np.float32)
    rays[0] = np.clip(ray_dist, 0.0, 1.0)
    rays[1 + ray_type, np.arange(NUM_RAYS)] = 1.0

    # 2. Visible entities: features (32, 5), types (32,), mask (32,)
    ent_raw = np.asarray(obs["visible_entities"]["features"], dtype=np.float32)
    ent_feat = np.empty((MAX_ENTITIES, 5), dtype=np.float32)
    ent_feat[:, 0] = np.clip(ent_raw[:, 0] / 4.0, 0.0, 1.0)  # distance / 4096
    ent_feat[:, 1] = np.clip(ent_raw[:, 1], -1.0, 1.0)       # sin(rel_angle)
    ent_feat[:, 2] = np.clip(ent_raw[:, 2], -1.0, 1.0)       # cos(rel_angle)
    ent_feat[:, 3] = np.tanh(ent_raw[:, 3] / 15.0)           # rel_vel_forward
    ent_feat[:, 4] = np.tanh(ent_raw[:, 4] / 15.0)           # rel_vel_right
    ent_type = np.clip(np.asarray(obs["visible_entities"]["entity_type"], dtype=np.int64), 0, NUM_ENTITY_TYPES - 1)
    ent_mask = (np.asarray(obs["visible_entities"]["mask"], dtype=np.int8) > 0)

    # 3. Vitals (17) + prev_action (9) + 4 ego summary scalars = (30,)
    v = np.asarray(obs["vitals"], dtype=np.float32)
    v_norm = np.zeros(NUM_VITALS, dtype=np.float32)
    v_norm[0] = np.clip(v[0] / 100.0, 0.0, 2.0)              # health
    v_norm[1] = np.clip(v[1] / 100.0, 0.0, 2.0)              # armor
    v_norm[2] = np.clip(v[2] / 7.0, 0.0, 1.0)                # selected_weapon
    v_norm[3:8] = np.log1p(np.maximum(v[3:8], 0.0)) / 5.0    # ammo counts
    v_norm[8:14] = np.clip(v[8:14], 0.0, 1.0)                # 6 key flags
    v_norm[14:17] = np.tanh(v[14:17] / 15.0)                 # vel_fwd, vel_right, vel_up

    if prev_action is None:
        pa = np.zeros(NUM_ACTIONS, dtype=np.float32)
    else:
        pa = np.asarray(prev_action, dtype=np.float32)

    # Front-cone (indices 61..63, 0..3 -> +/- ~17 deg) summary features
    front_idx = np.array([61, 62, 63, 0, 1, 2, 3], dtype=np.int64)
    front_dist = ray_dist[front_idx]
    front_type = ray_type[front_idx]
    summary = np.array(
        [
            float(np.min(front_dist)),
            float(np.any((front_type == 2) & (front_dist < 0.15))),  # door right ahead
            float(np.any((front_type == 3) & (front_dist < 0.15))),  # locked door right ahead
            float(np.any(ent_mask & ((ent_type >= 1) & (ent_type <= 3)))),  # enemy visible
        ],
        dtype=np.float32,
    )

    vitals_vec = np.concatenate([v_norm, pa, summary], axis=0)

    return {
        "rays": rays,
        "ent_feat": ent_feat,
        "ent_type": ent_type,
        "ent_mask": ent_mask,
        "vitals": vitals_vec,
    }
