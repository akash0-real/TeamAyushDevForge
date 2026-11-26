# Training

Put your training pipeline here: environment setup, reward shaping, curriculum, model code, logs.
Judges review this folder during manual review.

You may modify the engine, maps and game variables freely **for training**. Evaluation always uses
the vanilla engine through the official `doom_env` wrapper, so train on `doom_env` observations.

## Our pipeline

- `train_ppo.py`: custom recurrent PPO (GRU, truncated BPTT, per-env minibatches) for `submission/model.py`.
- `rewards.py`: training-only reward shaping from `env.privileged_state()` + ViZDoom's `DAMAGECOUNT`.
  The agent never sees any of this; it only gets the official observation dict.
- `eval_zeroshot.py`: scores a checkpoint through the real `submission/agent.py` on held-out maps.
- Map split: Freedoom 2 **MAP01–24 + `two_rooms` = training**, **MAP25–32 = holdout** (never trained on).

## Setup (once)

```powershell
uv sync
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cuXXX   # pick cuXXX for your GPU at pytorch.org
```

## Run

Smoke test (~2 min, must print `upd ... | ret ...` lines without crashing):
```powershell
uv run python training/train_ppo.py --maps two_rooms --num-envs 4 --total-steps 20000 --save-path training/checkpoints/smoke.pt
```

Main run (default maps = MAP01–24 + two_rooms, episode length random 2100–4200 ticks like evaluation):
```powershell
uv run python training/train_ppo.py --num-envs 12 --total-steps 20000000 --seed 42 --save-path training/checkpoints/main.pt
```
- `--num-envs`: physical cores minus 2, max ~12.
- Writes `<save-path>` (latest, every 10 updates) and `<stem>.best.pt` (best training return). Pick the
  final checkpoint with `eval_zeroshot.py`, not by training return.
- Continue a stopped run: add `--resume training/checkpoints/main.pt`.

## Windows note
On Windows, several ViZDoom games in parallel stall on the default 15.6 ms system timer: 4 games gave
51 steps/s total. `rewards.py` raises the timer resolution to 1 ms in every worker, which measured
~1,400 env steps/s with 12 envs on `two_rooms`.
