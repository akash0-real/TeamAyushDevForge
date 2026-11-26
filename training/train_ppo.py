"""Custom Recurrent PPO (Truncated BPTT) trainer for DevForge Track 1 (DOOM).

Built from scratch in PyTorch for our DoomSpatialPolicy (CircularRay + EntityAttention + GRU).
Supports single-map sanity checks and multi-map vectorized training with automatic
zero-shot evaluation and checkpoint saving.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom as vd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

from rewards import ShapedDoomEnv

if __name__ != "__mp_main__":
    # Windows/spawn re-runs this file in every env worker as __mp_main__; workers never need torch, and loading it
    # there cost ~490 MB per worker (43 MB without) -- the RAM that now goes into more parallel envs.
    import torch
    import torch.nn as nn

    from model import DoomSpatialPolicy, warm_start_map_model

IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")

# Freedoom 2 split: MAP01-24 train, MAP25-32 are the holdout (training/eval_zeroshot.py). Never train on 25-32.
# "genN" = a freshly generated tier-N map on every map load (maps/gen_maps.py). 48 gen tokens ~= 2/3 of loads:
# real Freedoom maps step ~3.4x slower (12 envs: 358 vs 1231 env steps/s), so they get 1/3 of the time.
# tutor round 3: real Freedoom maps doubled (~half of loads; holdout real-map score fell 7.9 -> 2.2 while generated
# tiers improved), tier 1 cut (100% on holdout)
DEFAULT_TRAIN_MAPS = (["two_rooms"] + [f"MAP{i:02d}" for i in range(1, 25)] * 2
                      + ["gen1"] * 3 + ["gen2"] * 12 + ["gen3"] * 18 + ["gen4"] * 12)


def parse_map_specs(spec_list: list[str], episode_ticks: int) -> list[tuple[str, str, int]]:
    out = []
    for item in spec_list:
        if item == "two_rooms":
            out.append((str(ROOT / "maps" / "two_rooms.wad"), "MAP01", episode_ticks))
        elif item in ("gen1", "gen2", "gen3", "gen4"):
            out.append((item, "MAP01", episode_ticks))
        elif ":" in item:
            wad, m = item.split(":", 1)
            p = Path(wad) if Path(wad).is_absolute() else ROOT / wad
            out.append((str(p), m.upper(), episode_ticks))
        elif item.upper().startswith("MAP"):
            out.append((IWAD, item.upper(), episode_ticks))
        else:
            p = Path(item) if Path(item).is_absolute() else ROOT / item
            out.append((str(p), "MAP01", episode_ticks))
    return out


def make_thunk(map_specs: list[tuple[str, str, int]], swap_every: int, seed: int, use_map: bool = False):
    def _init():
        return ShapedDoomEnv(map_specs=map_specs, swap_every=swap_every, seed=seed, use_map=use_map)
    return _init


def obs_to_torch(obs_dict: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "rays": torch.as_tensor(obs_dict["rays"], dtype=torch.float32, device=device),
        "ent_feat": torch.as_tensor(obs_dict["ent_feat"], dtype=torch.float32, device=device),
        "ent_type": torch.as_tensor(obs_dict["ent_type"], dtype=torch.long, device=device),
        "ent_mask": torch.as_tensor(obs_dict["ent_mask"], dtype=torch.bool, device=device),
        "vitals": torch.as_tensor(obs_dict["vitals"], dtype=torch.float32, device=device),
        **({"map": torch.as_tensor(obs_dict["map"], dtype=torch.float32, device=device)} if "map" in obs_dict else {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=DEFAULT_TRAIN_MAPS, help="Map specs: two_rooms, MAP01..MAP24, or path.wad:MAP01")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--episode-ticks", type=int, default=0, help="0 = random 2100-4200 per map load, like evaluation")
    ap.add_argument("--swap-every", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.015)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    ap.add_argument("--use-map", action="store_true", help="EgoMap memory grid input (submission/preprocess.py)")
    ap.add_argument("--init-from", type=str, default="", help="no-map checkpoint to warm-start a --use-map model")
    # Never write straight into submission/weights: the final checkpoint is picked by holdout eval.
    ap.add_argument("--save-path", type=str, default=str(ROOT / "training" / "checkpoints" / "main.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | envs={args.num_envs} | maps={args.maps}", flush=True)

    map_specs = parse_map_specs(args.maps, args.episode_ticks)
    envs = gym.vector.AsyncVectorEnv(
        [make_thunk(map_specs, args.swap_every, args.seed + i, args.use_map) for i in range(args.num_envs)],
        context="spawn",
        # SAME_STEP: the done step already returns the next episode's first obs (CleanRL-style buffers);
        # the default NEXT_STEP would insert one junk transition with an ignored action per episode.
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    model = DoomSpatialPolicy(use_map=args.use_map).to(device)
    if args.init_from:
        base = torch.load(args.init_from, map_location=device, weights_only=True)
        warm_start_map_model(model, base["model"] if "model" in base else base)
        print(f"Warm-started from {args.init_from}", flush=True)
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        print(f"Resumed weights from {args.resume}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DoomSpatialPolicy parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    T = args.rollout_steps
    B = args.num_envs
    batch_size = T * B
    num_updates = max(1, args.total_steps // batch_size)

    # Pre-allocate rollout buffers on CPU/GPU
    buf_rays = torch.zeros((T, B, 7, 64), dtype=torch.float32, device=device)
    buf_ent_feat = torch.zeros((T, B, 32, 5), dtype=torch.float32, device=device)
    buf_ent_type = torch.zeros((T, B, 32), dtype=torch.long, device=device)
    buf_ent_mask = torch.zeros((T, B, 32), dtype=torch.bool, device=device)
    buf_vitals = torch.zeros((T, B, 30), dtype=torch.float32, device=device)
    map_shape = envs.single_observation_space["map"].shape if args.use_map else None
    buf_map = torch.zeros((T, B, *map_shape), dtype=torch.float32, device=device) if args.use_map else None
    buf_actions = torch.zeros((T, B, 6), dtype=torch.long, device=device)
    buf_logprobs = torch.zeros((T, B), dtype=torch.float32, device=device)
    buf_rewards = torch.zeros((T, B), dtype=torch.float32, device=device)
    buf_dones = torch.zeros((T, B), dtype=torch.float32, device=device)
    buf_values = torch.zeros((T, B), dtype=torch.float32, device=device)

    obs, _ = envs.reset(seed=args.seed)
    next_obs = obs_to_torch(obs, device)
    next_done = torch.zeros(B, dtype=torch.float32, device=device)
    next_h = model.initial_state(B, device=device)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_return = -1e9
    recent_stats: list[dict[str, float]] = []

    global_step = 0
    t_start = time.perf_counter()

    try:
        for update in range(1, num_updates + 1):
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = max(5e-5, args.lr * frac)
            init_h = next_h.clone()
            t_roll = time.perf_counter()

            # 1. Rollout collection
            model.eval()
            with torch.inference_mode():
                for step in range(T):
                    global_step += B
                    buf_rays[step] = next_obs["rays"]
                    buf_ent_feat[step] = next_obs["ent_feat"]
                    buf_ent_type[step] = next_obs["ent_type"]
                    buf_ent_mask[step] = next_obs["ent_mask"]
                    buf_vitals[step] = next_obs["vitals"]
                    if buf_map is not None:
                        buf_map[step] = next_obs["map"]
                    buf_dones[step] = next_done

                    # Reset hidden state where previous step ended an episode
                    next_h = next_h * (1.0 - next_done).view(1, B, 1)
                    logits, value, next_h = model.forward_step(next_obs, next_h)
                    action, logprob, _ = model.sample_and_logprob(logits)

                    buf_actions[step] = action
                    buf_logprobs[step] = logprob
                    buf_values[step] = value

                    obs_np, rew_np, term_np, trunc_np, infos = envs.step(action.cpu().numpy())
                    done_np = np.logical_or(term_np, trunc_np)
                    buf_rewards[step] = torch.as_tensor(rew_np, dtype=torch.float32, device=device)
                    next_obs = obs_to_torch(obs_np, device)
                    next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

                    # Gymnasium 1.x packs finished-episode infos as dicts of per-env arrays plus "_key" masks
                    st = infos.get("final_info", {}).get("episode_stats")
                    if st is not None:
                        for idx in np.nonzero(infos["final_info"]["_episode_stats"])[0]:
                            recent_stats.append({k: float(v[idx]) for k, v in st.items() if not k.startswith("_")})

                # Bootstrap value
                boot_h = next_h * (1.0 - next_done).view(1, B, 1)
                _, next_value, _ = model.forward_step(next_obs, boot_h)

            t_upd = time.perf_counter()
            # 2. GAE
            advantages = torch.zeros_like(buf_rewards)
            last_gae = torch.zeros(B, dtype=torch.float32, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_nonterminal = 1.0 - next_done
                    next_val = next_value
                else:
                    next_nonterminal = 1.0 - buf_dones[t + 1]
                    next_val = buf_values[t + 1]
                delta = buf_rewards[t] + args.gamma * next_val * next_nonterminal - buf_values[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + buf_values

            # 3. PPO Update (chunked along env dimension to preserve exact GRU time order!)
            model.train()
            env_inds = np.arange(B)
            envs_per_mb = max(1, B // args.num_minibatches)
            pg_losses, v_losses, ent_losses = [], [], []

            for _ in range(args.update_epochs):
                np.random.shuffle(env_inds)
                for start in range(0, B, envs_per_mb):
                    mb = env_inds[start : start + envs_per_mb]
                    mb_obs = {
                        "rays": buf_rays[:, mb],
                        "ent_feat": buf_ent_feat[:, mb],
                        "ent_type": buf_ent_type[:, mb],
                        "ent_mask": buf_ent_mask[:, mb],
                        "vitals": buf_vitals[:, mb],
                    }
                    if buf_map is not None:
                        mb_obs["map"] = buf_map[:, mb]
                    logits, new_values, _ = model.forward_sequence(mb_obs, init_h[:, mb], buf_dones[:, mb])
                    _, new_logprob, entropy = model.sample_and_logprob(logits, buf_actions[:, mb])

                    logratio = new_logprob - buf_logprobs[:, mb]
                    ratio = logratio.exp()

                    mb_adv = advantages[:, mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    v_loss = 0.5 * ((new_values - returns[:, mb]) ** 2).mean()
                    ent_loss = entropy.mean()
                    loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    ent_losses.append(ent_loss.item())

            t_end = time.perf_counter()  # rollout = t_upd - t_roll, update = t_end - t_upd (logged below)
            if update % 10 == 0 or update == num_updates:
                torch.save({"model": model.state_dict(), "step": global_step}, save_path)  # latest, for holdout eval
            if update % 5 == 0 or update == num_updates:
                fps = global_step / (time.perf_counter() - t_start)
                if recent_stats:
                    window = recent_stats[-40:]
                    avg_ret = float(np.mean([s["return"] for s in window]))
                    exit_r = float(np.mean([s["exit_reached"] for s in window]))
                    kills = float(np.mean([s["kills"] for s in window]))
                    doors = float(np.mean([s["doors"] for s in window]))
                    keys = float(np.mean([s["keys"] for s in window]))
                    cells = float(np.mean([s["cells"] for s in window]))
                    # exit rate per map tier (0 = real Freedoom/template maps) over the last 200 episodes
                    by_tier = {}
                    for s in recent_stats[-200:]:
                        by_tier.setdefault(int(s.get("tier", 0)), []).append(s["exit_reached"])
                    tiers = " ".join(f"T{t}:{100*np.mean(v):.0f}%/{len(v)}" for t, v in sorted(by_tier.items()))
                    print(
                        f"upd {update:4d}/{num_updates} | step {global_step:8d} | fps {fps:6.0f} | "
                        f"ret {avg_ret:6.2f} | exit {exit_r*100:4.0f}% | kills {kills:4.1f} | "
                        f"doors {doors:3.1f} | keys {keys:3.1f} | cells {cells:4.1f} | "
                        f"ent {np.mean(ent_losses):.2f} | exit-by-tier {tiers} | "
                        f"rollout {t_upd - t_roll:.1f}s update {t_end - t_upd:.1f}s",
                        flush=True,
                    )
                    if avg_ret > best_return:
                        best_return = avg_ret
                        torch.save({"model": model.state_dict(), "step": global_step, "avg_return": avg_ret},
                                   save_path.with_suffix(".best.pt"))
                else:
                    print(
                        f"upd {update:4d}/{num_updates} | step {global_step:8d} | fps {fps:6.0f} | "
                        f"pg {np.mean(pg_losses):.3f} | v {np.mean(v_losses):.3f} | ent {np.mean(ent_losses):.2f}",
                        flush=True,
                    )

        print(f"Done. Latest: {save_path} | best train return: {save_path.with_suffix('.best.pt')} ({best_return:.2f})", flush=True)
    finally:
        envs.close()


if __name__ == "__main__":
    main()
