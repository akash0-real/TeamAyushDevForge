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

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "best_agent.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        if WEIGHTS_PATH.is_file():
            self.model = load_policy(WEIGHTS_PATH, self.device)  # picks the EgoMap variant from the checkpoint
        else:
            self.model = DoomSpatialPolicy().to(self.device)
        self.model.eval()
        self.ego_map = EgoMap() if self.model.use_map else None
        self.reset()

    def reset(self) -> None:
        self.h = self.model.initial_state(batch_size=1, device=self.device)
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self.first_step = True
        if self.ego_map is not None:
            self.ego_map.reset()

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        enc = preprocess_obs(observation, self.prev_action)
        if self.ego_map is not None:
            # same call order as training (training/rewards.py): no action before the first observation
            self.ego_map.update(observation, None if self.first_step else self.prev_action)
            enc["map"] = self.ego_map.crops()
        self.first_step = False
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in enc.items()}
        logits, _, self.h = self.model.forward_step(obs_t, self.h)
        factored, _, _ = self.model.sample_and_logprob(logits, deterministic=False)
        action = factored_to_multibinary(factored.squeeze(0).cpu().numpy())
        self.prev_action = action
        return action
