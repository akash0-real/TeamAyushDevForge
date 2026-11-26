"""Evaluation entry point for DevForge Track 1 ("Can It Play DOOM?").

Implements DoomAgent using our custom CircularRay + EntityAttention + GRU
spatial-memory policy trained with Recurrent PPO.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from model import DoomSpatialPolicy, load_policy
from preprocess import NUM_ACTIONS, EgoMap, factored_to_multibinary, preprocess_obs

ENEMY_TYPES = (1, 2, 3)  # doom_env.spaces.ENTITY_TYPES: enemy_melee, enemy_hitscan, enemy_projectile
FIST_REACH = 100.0 / 1024.0  # entity features[0] is distance / 1024; melee range 64 + monster radius

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "best_agent.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        self.load_weights(WEIGHTS_PATH if WEIGHTS_PATH.is_file() else None)

    def load_weights(self, path) -> None:
        """(Re)build the model for a checkpoint (EgoMap variant picked from its keys) and warm it up."""
        self.model = load_policy(path, self.device) if path else DoomSpatialPolicy().to(self.device)
        self.model.eval()
        self.ego_map = EgoMap() if self.model.use_map else None
        self.temperature = 1.0  # action-sampling temperature (logits / T); tuned on the holdout maps
        # Turn head only, sharper sampling (less left/right jitter). Seeded test on v6, 10 seeds x 17 holdout maps: exit_points
        # 98 vs 89 at 0.5 (T3 16 vs 13 exits), mean +0.7 ± 3.3 (training/results/tt*_v6_*.csv). Rollback: tag shipped-v6.
        self.turn_temperature = 0.5
        # Extra per-head temperatures [move, strafe, turn, attack, use, weapon]. Strafe 0.5: seeded sweep vs v7, 20 seeds x 17 holdout,
        # +2.0 ± 1.6 mean, exit_points 210 vs 199, unseen real +2.0 (training/sweep_check.py). Rollback: tag shipped-v7.
        self.head_temperature = [1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        # Observation-based masking rules (docs/RULES.md §1), see _mask: "weapon" and/or "fist". Off: pooled over two
        # machines it was neutral (+1.8 ± 3.1) and "weapon" may hurt tier 4 (teammate 3 EVAL_I3 §6b).
        self.action_mask: frozenset[str] = frozenset()
        # Warm-up in __init__ (60 s limit) so the first act() of a map (10 s limit) isn't paying torch's lazy init:
        # measured ~4 s for the first act() on one pinned CPU core without this.
        with torch.inference_mode():
            zeros = {
                "rays": torch.zeros(1, 7, 64), "ent_feat": torch.zeros(1, 32, 5),
                "ent_type": torch.zeros(1, 32, dtype=torch.long), "ent_mask": torch.zeros(1, 32, dtype=torch.bool),
                "vitals": torch.zeros(1, 30), "map": torch.zeros(1, 12, 15, 15),
            }
            for _ in range(3):
                self.model.forward_step({k: v.to(self.device) for k, v in zeros.items()}, self.model.initial_state(1, self.device))
        self.reset()

    def reset(self) -> None:
        self.h = self.model.initial_state(batch_size=1, device=self.device)
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self.first_step = True
        if self.ego_map is not None:
            self.ego_map.reset()

    def act(self, observation: dict) -> np.ndarray:
        # Safety net only: an uncaught error would freeze the player for the rest of the map (docs/EVALUATION.md).
        # On an unexpected error we press nothing for this step and keep going; actions otherwise come only from
        # the network.
        try:
            return self._act(observation)
        except Exception as e:  # noqa: BLE001
            print(f"DoomAgent.act error, no-op this step: {e!r}", flush=True)
            self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
            return self.prev_action

    @staticmethod
    def _mask(observation: dict, logits: list, rules: frozenset) -> list:
        """Remove two never-useful options, read from observation fields only; the network still picks every action.
        Heads: [move, strafe, turn, attack, use, weapon]; option 1 of attack/weapon = press."""
        vit = np.asarray(observation["vitals"], dtype=np.float32)
        slot, ammo = float(vit[2]), float(vit[3])
        ent = observation["visible_entities"]
        enemy = (np.asarray(ent["mask"]) > 0) & np.isin(np.asarray(ent["entity_type"]), ENEMY_TYPES)
        near = enemy & (np.asarray(ent["features"], dtype=np.float32)[:, 0] <= FIST_REACH)
        logits = list(logits)
        if "weapon" in rules and slot >= 3 and ammo > 0:  # shotgun or better, loaded: don't cycle away from it
            logits[5] = logits[5].clone()
            logits[5][..., 1] = float("-inf")
        if "fist" in rules and slot <= 1 and not near.any():  # fist/chainsaw, nothing in reach: don't swing at air
            logits[3] = logits[3].clone()
            logits[3][..., 1] = float("-inf")
        return logits

    @torch.inference_mode()
    def _act(self, observation: dict) -> np.ndarray:
        enc = preprocess_obs(observation, self.prev_action)
        if self.ego_map is not None:
            # same call order as training (training/rewards.py): no action before the first observation
            self.ego_map.update(observation, None if self.first_step else self.prev_action)
            enc["map"] = self.ego_map.crops()
        self.first_step = False
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in enc.items()}
        logits, _, self.h = self.model.forward_step(obs_t, self.h)
        logits = [lg / self.temperature for lg in logits]
        logits[2] = logits[2] / self.turn_temperature  # heads: move, strafe, turn, attack, use, weapon
        logits = [lg / t for lg, t in zip(logits, self.head_temperature)]
        if self.action_mask:
            logits = self._mask(observation, logits, self.action_mask)
        factored, _, _ = self.model.sample_and_logprob(logits)
        action = factored_to_multibinary(factored.squeeze(0).cpu().numpy())
        self.prev_action = action
        return action
