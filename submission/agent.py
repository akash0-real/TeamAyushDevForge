"""Evaluation entry point for DevForge Track 1 ("Can It Play DOOM?").

Implements DoomAgent using our custom CircularRay + EntityAttention + GRU
spatial-memory policy trained with Recurrent PPO.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from model import DoomSpatialPolicy
from preprocess import NUM_ACTIONS, factored_to_multibinary, preprocess_obs

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "best_agent.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        self.model = DoomSpatialPolicy().to(self.device)
        if WEIGHTS_PATH.is_file():
            ckpt = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
        self.model.eval()

        self.h = self.model.initial_state(batch_size=1, device=self.device)
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)

    def reset(self) -> None:
        self.h = self.model.initial_state(batch_size=1, device=self.device)
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        enc = preprocess_obs(observation, self.prev_action)
        obs_t = {
            "rays": torch.from_numpy(enc["rays"]).unsqueeze(0).to(self.device),
            "ent_feat": torch.from_numpy(enc["ent_feat"]).unsqueeze(0).to(self.device),
            "ent_type": torch.from_numpy(enc["ent_type"]).unsqueeze(0).to(self.device),
            "ent_mask": torch.from_numpy(enc["ent_mask"]).unsqueeze(0).to(self.device),
            "vitals": torch.from_numpy(enc["vitals"]).unsqueeze(0).to(self.device),
        }
        logits, _, self.h = self.model.forward_step(obs_t, self.h)
        factored, _, _ = self.model.sample_and_logprob(logits, deterministic=False)
        action = factored_to_multibinary(factored.squeeze(0).cpu().numpy())
        self.prev_action = action
        return action
