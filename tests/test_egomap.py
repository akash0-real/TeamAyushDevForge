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


def _blank_obs():
    return {
        "rays": {"distance": np.ones(64, np.float32), "hit_type": np.zeros(64, np.int64)},
        "visible_entities": {"features": np.zeros((32, 5), np.float32), "entity_type": np.zeros(32, np.int64),
                             "mask": np.zeros(32, np.int8)},
        "vitals": np.zeros(17, np.float32),
    }


def test_grid_scrolls_on_big_maps_and_keeps_memory():
    em = EgoMap()
    em.update(_blank_obs(), None)  # visited: spawn cell (64, 64)
    assert em.grid[0, 64, 64] == 1
    em.x = 2500.0  # 39 cells east of spawn: within 1/4 of the edge -> scroll
    em.update(_blank_obs(), None)
    assert em.ox == 32 and em.oy == 0
    assert em.grid[0, 64, 64 - 32] == 1, "spawn memory must move with the scroll"
    assert em.grid[0, 64, 64 + 39 - 32] == 1, "agent's current cell is marked inside the grid"
    assert em.crops().shape == (12, 15, 15)
    em.reset()
    assert em.ox == em.oy == 0 and em.grid.sum() == 0


def test_act_survives_internal_errors(monkeypatch):
    import agent as agent_mod

    env = make_env(WAD, "MAP01", episode_ticks=2100, render_buffers=False)
    ag = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    obs, _ = env.reset(seed=0)

    def boom(_obs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(ag, "_act", boom)
    act = ag.act(obs)
    assert act.shape == (9,) and act.dtype == np.int8 and not act.any()
    env.close()


def test_chunked_gru_matches_per_step_reference():
    """forward_sequence runs the GRU per stretch between episode ends; must equal stepping one tic at a time."""
    torch.manual_seed(0)
    m = DoomSpatialPolicy(use_map=True)
    T, B = 40, 3
    obs = {
        "rays": torch.rand(T, B, 7, 64), "ent_feat": torch.rand(T, B, 32, 5),
        "ent_type": torch.randint(0, 12, (T, B, 32)), "ent_mask": torch.rand(T, B, 32) > 0.5,
        "vitals": torch.rand(T, B, 30), "map": torch.rand(T, B, 12, 15, 15),
    }
    dones = torch.zeros(T, B)
    dones[0, 1] = dones[7, 0] = dones[7, 2] = dones[23, 1] = 1.0
    h0 = torch.rand(1, B, m.hidden_dim)
    with torch.no_grad():
        lg, val, hT = m.forward_sequence(obs, h0, dones)
        h, ref_v = h0, []
        for t in range(T):
            h = h * (1.0 - dones[t]).view(1, B, 1)
            _, v, h = m.forward_step({k: x[t] for k, x in obs.items()}, h)
            ref_v.append(v)
    assert torch.allclose(val, torch.stack(ref_v), atol=1e-5)
    assert torch.allclose(hT, h, atol=1e-5)


def test_load_policy_reads_the_gru_size_from_the_checkpoint(tmp_path):
    from model import load_policy

    ckpt = tmp_path / "big.pt"
    torch.save({"model": DoomSpatialPolicy(use_map=True, hidden_dim=512).state_dict()}, ckpt)
    m = load_policy(ckpt, torch.device("cpu"))
    assert m.hidden_dim == 512 and m.use_map
