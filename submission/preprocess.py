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


# ---------------------------------------------------------------------------------------------------------------
# EgoMap: spawn-anchored memory grid built only from the observation + our own past actions (no engine coords).
# It is an INPUT to the network (allowed: "preprocessing/encoding observations, learned memory"); it never picks
# actions. Pose = dead reckoning: heading from our TURN presses, position from the observed player-frame velocity.
# ---------------------------------------------------------------------------------------------------------------
MAP_CELLS = 128          # 128 x 128 cells
MAP_CELL_UNITS = 64.0    # 64 map units per cell -> 8192 x 8192 units around spawn
MAP_CHANNELS = 6         # visited, free-seen, wall, door, locked door, key
MAP_CROP = 15            # network sees two 15x15 ego-rotated crops: fine (1 cell) + coarse (8 cells)
MAP_COARSE = 8
# Doom turning, measured per tic: while any turn button is held a counter runs (switching left<->right keeps it,
# releasing resets it); the first 5 held tics turn 320/65536 of a circle, later ones 640/65536. Input during the
# first 7 tics of an episode is ignored but still advances the counter. Reproduces every measured step exactly.
TURN_SLOW_DEG = 320 / 65536 * 360
TURN_FAST_DEG = 640 / 65536 * 360
TURN_SLOW_TICS = 5
TURN_DEAD_TICS = 7
TICS_PER_STEP = 4
_RAY_ANG = np.arange(NUM_RAYS) * (2 * np.pi / NUM_RAYS)            # index 0 ahead, counter-clockwise
_RAY_SAMPLES = np.arange(0.5, 32.0) / 32.0                         # free-space samples along each ray
_CROP_OFF = np.arange(MAP_CROP) - MAP_CROP // 2


class EgoMap:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grid = np.zeros((MAP_CHANNELS, MAP_CELLS, MAP_CELLS), dtype=np.float32)
        self.x = self.y = self.theta = 0.0
        self.turn_held = 0          # consecutive tics with a turn button down
        self.tic = 0                # game tics since reset
        self.ox = self.oy = 0       # grid origin shift in cells (the grid scrolls to keep the agent inside)
        self.prev_vel = np.zeros(2, dtype=np.float32)

    def _cells(self, wx: np.ndarray, wy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cx = np.floor(wx / MAP_CELL_UNITS).astype(np.int64) + MAP_CELLS // 2 - self.ox
        cy = np.floor(wy / MAP_CELL_UNITS).astype(np.int64) + MAP_CELLS // 2 - self.oy
        return np.clip(cx, 0, MAP_CELLS - 1), np.clip(cy, 0, MAP_CELLS - 1)

    def _recenter(self) -> None:
        """Scroll the grid when the agent gets within 1/4 of its edge (big maps): keep everything near the agent,
        drop what falls off the far side. Shifts are multiples of MAP_COARSE so the coarse view stays aligned."""
        n, k = MAP_CELLS, MAP_COARSE
        cx = int(np.floor(self.x / MAP_CELL_UNITS)) + n // 2 - self.ox
        cy = int(np.floor(self.y / MAP_CELL_UNITS)) + n // 2 - self.oy
        if n // 4 <= cx < n - n // 4 and n // 4 <= cy < n - n // 4:
            return
        sx, sy = (cx - n // 2) // k * k, (cy - n // 2) // k * k
        new = np.zeros_like(self.grid)
        src = self.grid[:, max(sy, 0): n + min(sy, 0), max(sx, 0): n + min(sx, 0)]
        new[:, max(-sy, 0): n - max(sy, 0), max(-sx, 0): n - max(sx, 0)] = src
        self.grid = new
        self.ox += sx
        self.oy += sy

    def update(self, obs: dict, action: np.ndarray | None) -> None:
        """Advance the pose with the action just taken, then write this observation into the grid."""
        if action is not None:
            left, right = int(action[4]), int(action[5])
            d = left - right
            for _ in range(TICS_PER_STEP):
                self.tic += 1
                if d == 0:
                    self.turn_held = 0
                    continue
                self.turn_held += 1
                if self.tic > TURN_DEAD_TICS:
                    deg = TURN_SLOW_DEG if self.turn_held <= TURN_SLOW_TICS else TURN_FAST_DEG
                    self.theta += d * np.radians(deg)
            vf, vr = float(obs["vitals"][14]), float(obs["vitals"][15])
            c, s = np.cos(self.theta), np.sin(self.theta)
            vel = np.array([vf * c + vr * s, vf * s - vr * c], dtype=np.float32)  # player frame -> spawn frame
            step = TICS_PER_STEP * 0.5 * (vel + self.prev_vel)
            self.x += float(step[0])
            self.y += float(step[1])
            self.prev_vel = vel

        self._recenter()
        g = self.grid
        px, py = self._cells(np.array([self.x]), np.array([self.y]))
        g[0, py, px] = 1.0
        dist = np.asarray(obs["rays"]["distance"], dtype=np.float32) * 1024.0
        hit = np.asarray(obs["rays"]["hit_type"], dtype=np.int64)
        ang = self.theta + _RAY_ANG
        ca, sa = np.cos(ang), np.sin(ang)
        r = dist[:, None] * _RAY_SAMPLES[None, :]                       # (64, 32) free-space points
        fx, fy = self._cells(self.x + r * ca[:, None], self.y + r * sa[:, None])
        g[1, fy, fx] = 1.0
        hx, hy = self._cells(self.x + dist * ca, self.y + dist * sa)
        for t, ch in ((1, 2), (2, 3), (3, 4)):                             # wall, door, locked door
            m = (hit == t) & (dist < 1023.0)
            g[ch, hy[m], hx[m]] = 1.0
        ent = obs["visible_entities"]
        keys = (np.asarray(ent["mask"]) > 0) & (np.asarray(ent["entity_type"]) == 9)
        if keys.any():
            f = np.asarray(ent["features"], dtype=np.float32)[keys]
            ea = self.theta + np.arctan2(f[:, 1], f[:, 2])               # encoding only, never used to aim
            kx, ky = self._cells(self.x + f[:, 0] * 1024.0 * np.cos(ea), self.y + f[:, 0] * 1024.0 * np.sin(ea))
            g[5, ky, kx] = 1.0

    def crops(self) -> np.ndarray:
        """(2*C, 15, 15) ego-centric, heading-up crops: fine (1 cell/px) and coarse (8 cells/px, max-pooled)."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        fwd, left = np.meshgrid(_CROP_OFF[::-1], _CROP_OFF[::-1], indexing="ij")  # row 0 = ahead, col 0 = left
        out = []
        for scale, grid in ((1, self.grid), (MAP_COARSE, self._coarse())):
            u = MAP_CELL_UNITS * scale
            wx = self.x + (fwd * c - left * s) * u
            wy = self.y + (fwd * s + left * c) * u
            cx = np.floor(wx / u).astype(np.int64) + grid.shape[2] // 2 - self.ox // scale
            cy = np.floor(wy / u).astype(np.int64) + grid.shape[1] // 2 - self.oy // scale
            inside = (cx >= 0) & (cx < grid.shape[2]) & (cy >= 0) & (cy < grid.shape[1])
            crop = grid[:, np.clip(cy, 0, grid.shape[1] - 1), np.clip(cx, 0, grid.shape[2] - 1)]
            out.append(crop * inside)
        return np.concatenate(out, axis=0).astype(np.float32)

    def _coarse(self) -> np.ndarray:
        k = MAP_COARSE
        return self.grid.reshape(MAP_CHANNELS, MAP_CELLS // k, k, MAP_CELLS // k, k).max(axis=(2, 4))
