"""Fast CI unit tests for submission/preprocess.py, model.py, and agent.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))

from doom_env import spaces as S
from preprocess import factored_to_multibinary, preprocess_obs


def _make_dummy_obs(num_entities: int = 0) -> dict:
    obs = {
        "rgb": np.zeros((S.SCREEN_HEIGHT, S.SCREEN_WIDTH, 3), dtype=np.uint8),
        "depth": np.zeros((S.SCREEN_HEIGHT, S.SCREEN_WIDTH), dtype=np.uint8),
        "vitals": np.zeros((len(S.VITALS_FIELDS),), dtype=np.float32),
        "rays": {
            "distance": np.full((S.NUM_RAYS,), 0.5, dtype=np.float32),
            "hit_type": np.ones((S.NUM_RAYS,), dtype=np.int64),
        },
        "visible_entities": {
            "features": np.zeros((S.MAX_ENTITIES, len(S.ENTITY_FEATURES)), dtype=np.float32),
            "entity_type": np.zeros((S.MAX_ENTITIES,), dtype=np.int64),
            "mask": np.zeros((S.MAX_ENTITIES,), dtype=np.int8),
        },
    }
    obs["vitals"][0] = 100.0
    if num_entities > 0:
        obs["visible_entities"]["mask"][:num_entities] = 1
        obs["visible_entities"]["entity_type"][:num_entities] = 2
        obs["visible_entities"]["features"][:num_entities, 0] = 0.4
        obs["visible_entities"]["features"][:num_entities, 2] = 1.0
    return obs


def test_preprocess_and_factored_mapping():
    obs = _make_dummy_obs(num_entities=3)
    enc = preprocess_obs(obs)
    assert enc["rays"].shape == (7, 64)
    assert enc["ent_feat"].shape == (32, 5)
    assert enc["ent_type"].shape == (32,)
    assert enc["ent_mask"].shape == (32,)
    assert enc["vitals"].shape == (30,)
    assert np.all(np.isfinite(enc["rays"]))
    assert np.all(np.isfinite(enc["vitals"]))

    # Verify mutually exclusive movement/turn buttons
    mb = factored_to_multibinary(np.array([1, 2, 1, 1, 0, 1]))
    assert mb.shape == (9,)
    assert mb.dtype == np.int8
    assert mb.tolist() == [1, 0, 0, 1, 1, 0, 1, 0, 1]


def test_model_and_agent_zero_and_nonzero_entities():
    torch = pytest.importorskip("torch")
    from agent import DoomAgent
    from model import DoomSpatialPolicy

    model = DoomSpatialPolicy()
    for n_ent in (0, 5):
        enc = preprocess_obs(_make_dummy_obs(num_entities=n_ent))
        obs_t = {
            "rays": torch.from_numpy(enc["rays"]).unsqueeze(0),
            "ent_feat": torch.from_numpy(enc["ent_feat"]).unsqueeze(0),
            "ent_type": torch.from_numpy(enc["ent_type"]).unsqueeze(0),
            "ent_mask": torch.from_numpy(enc["ent_mask"]).unsqueeze(0),
            "vitals": torch.from_numpy(enc["vitals"]).unsqueeze(0),
        }
        h = model.initial_state(1)
        logits, value, next_h = model.forward_step(obs_t, h)
        assert len(logits) == 6
        assert torch.isfinite(value).all()
        assert torch.isfinite(next_h).all()
        for lg in logits:
            assert torch.isfinite(lg).all()

    # Sequence BPTT forward with done reset
    T, B = 4, 2
    enc = preprocess_obs(_make_dummy_obs(num_entities=2))
    seq_obs = {
        "rays": torch.from_numpy(enc["rays"]).view(1, 1, 7, 64).expand(T, B, -1, -1).clone(),
        "ent_feat": torch.from_numpy(enc["ent_feat"]).view(1, 1, 32, 5).expand(T, B, -1, -1).clone(),
        "ent_type": torch.from_numpy(enc["ent_type"]).view(1, 1, 32).expand(T, B, -1).clone(),
        "ent_mask": torch.from_numpy(enc["ent_mask"]).view(1, 1, 32).expand(T, B, -1).clone(),
        "vitals": torch.from_numpy(enc["vitals"]).view(1, 1, 30).expand(T, B, -1).clone(),
    }
    dones = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    logits_s, values_s, _ = model.forward_sequence(seq_obs, model.initial_state(B), dones)
    assert values_s.shape == (T, B)
    assert torch.isfinite(values_s).all()

    # DoomAgent contract test
    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    agent.reset()
    for n_ent in (0, 4):
        act = agent.act(_make_dummy_obs(num_entities=n_ent))
        assert act.shape == (9,)
        assert set(np.unique(act)).issubset({0, 1})


def test_action_mask_removes_only_the_masked_options():
    torch = pytest.importorskip("torch")
    from agent import DoomAgent

    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    agent.action_mask = frozenset({"weapon", "fist"})
    loaded = _make_dummy_obs()
    loaded["vitals"][2], loaded["vitals"][3] = 3.0, 20.0  # shotgun with ammo
    fist = _make_dummy_obs()
    fist["vitals"][2] = 1.0  # fists, no enemy in view
    for _ in range(200):
        assert agent.act(loaded)[8] == 0  # never SELECT_NEXT_WEAPON
        assert agent.act(fist)[6] == 0  # never ATTACK
    close = _make_dummy_obs(num_entities=1)
    close["vitals"][2] = 1.0
    close["visible_entities"]["features"][0, 0] = 50.0 / 1024.0  # enemy within reach: attack allowed again
    assert any(agent.act(close)[6] for _ in range(200))


def test_turn_temperature_only_sharpens_the_turn_head():
    torch = pytest.importorskip("torch")
    from agent import DoomAgent

    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    agent.turn_temperature = 1e-3  # near-greedy turning
    obs = _make_dummy_obs(num_entities=2)
    torch.manual_seed(0)
    turns = {tuple(agent.act(obs)[4:6]) for _ in range(50)}
    agent.reset()
    assert len(turns) <= 2  # at most the greedy choice (+ the first step before the GRU state settles)
