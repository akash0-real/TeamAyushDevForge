"""EgoMap memory grid + map-model agent. Run: uv run pytest tests/test_egomap.py -q"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "submission")]

from doom_env import make_env  # noqa: E402
from model import DoomSpatialPolicy, warm_start_map_model  # noqa: E402
from preprocess import EgoMap  # noqa: E402

WAD = str(ROOT / "maps" / "two_rooms.wad")


def test_heading_matches_engine_exactly():
    """Turn model (slow-turn ramp + 7 dead tics at episode start) must track the true angle with zero drift."""
    env = make_env(WAD, "MAP01", episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    em = EgoMap()
    em.update(obs, None)
    a0 = env.privileged_state()["angle"]
    rng = np.random.default_rng(0)
    a = np.zeros(9, np.int8)
    for i in range(300):
        if i % 3 == 0:
            a = np.zeros(9, np.int8)
            a[rng.choice([4, 5])] = rng.random() < 0.7
            a[0] = rng.random() < 0.5
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
        em.update(obs, a)
        true = env.privileged_state()["angle"] - a0
        err = (math.degrees(em.theta) - true + 180) % 360 - 180
        assert abs(err) < 0.01, (i, err)
    env.close()


def test_warm_start_reproduces_baseline_when_map_is_ignored():
    torch.manual_seed(0)
    base = DoomSpatialPolicy()
    mapm = DoomSpatialPolicy(use_map=True)
    warm_start_map_model(mapm, base.state_dict())
    B = 2
    obs = {
        "rays": torch.rand(B, 7, 64), "ent_feat": torch.rand(B, 32, 5),
        "ent_type": torch.randint(0, 12, (B, 32)), "ent_mask": torch.rand(B, 32) > 0.5,
        "vitals": torch.rand(B, 30), "map": torch.rand(B, 12, 15, 15),
    }
    with torch.no_grad():
        lb, vb, _ = base.forward_step(obs, base.initial_state(B))
        lm, vm, _ = mapm.forward_step(obs, mapm.initial_state(B))
    assert torch.allclose(vb, vm, atol=1e-5)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(lb, lm))


def test_map_agent_plays_valid_fast_actions(tmp_path, monkeypatch):
    import agent as agent_mod

    ckpt = tmp_path / "map.pt"
    torch.save({"model": DoomSpatialPolicy(use_map=True).state_dict()}, ckpt)
    monkeypatch.setattr(agent_mod, "WEIGHTS_PATH", ckpt)
    torch.set_num_threads(1)
    env = make_env(WAD, "MAP01", episode_ticks=2100, render_buffers=False)
    ag = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    assert ag.model.use_map and ag.ego_map is not None
    for _ in range(2):  # two maps: reset must clear the memory grid
        obs, _ = env.reset(seed=0)
        ag.reset()
        assert ag.ego_map.grid.sum() == 0
        times = []
        for _ in range(150):
            t = time.perf_counter()
            act = ag.act(obs)
            times.append(time.perf_counter() - t)
            assert act.shape == (9,) and set(np.unique(act)) <= {0, 1}
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
        assert np.mean(times[1:]) < 0.040, f"{1000 * np.mean(times[1:]):.1f} ms/act"
    env.close()
