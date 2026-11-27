# Review Cheat-Sheet — how to explain our DOOM agent

Read this once before a judge comes over. Short answers first; the details follow underneath.

---

## 1. The 30-second pitch

> "We built a DOOM-playing agent that learns entirely from trial and error, with no hand-written rules.
> It sees the world as 64 distance rays plus a list of visible things, and it builds its **own map in its head** while it
> plays: it works out where it is by counting its own turns and steps, so it can remember where a locked door was and
> come back once it finds the key. We trained it with our own PPO implementation on an **endless stream of
> randomly generated levels** plus real Freedoom maps. While it trained, we acted as its **tutor**: we read the logs,
> spotted which skill it was failing at, and changed its rewards and level mix."

## 2. Our three USPs (say these by name)

1. **EgoMap: a self-built spatial memory.** The game never tells the agent where it is. We track its heading from its
   own turn presses, using Doom's exact turning rules that we measured tic by tic, with **0.0° error** over 600 steps.
   Position comes from its observed velocity and drifts only 1–2 grid cells over a long episode. Everything it sees is
   drawn into a 128×128 grid: where it has been, walls, doors, **locked doors** and **keys**. The network reads two
   small crops of that map. That's how Tier 3 works: remember the locked door, find the key, come back.
2. **Infinite procedural levels with a solvability proof.** `maps/gen_maps.py` creates a brand-new level on every
   map load (4 difficulty tiers matching the competition). For key/lock levels, its self-test *proves* each key is
   reachable and the exit is reachable *only* after unlocking. The agent can't memorize maps because it never sees the
   same generated map twice.
3. **"Tutor" training loop.** We didn't just press train and wait. Example: Tier 3 was stuck because the agent almost
   never found the key. Round 1 raised the key reward from 3 to 8 and made more Tier 3 maps. Round 2 added a
   curriculum of small 3-room key maps. Each round resumed from the same weights, so nothing learned was lost.

## 3. How it works, in plain words

| Piece | What it does | Where |
|---|---|---|
| **Eyes** | 64 rays all around (distance + what was hit), up to 32 visible things (enemies, items, keys) | official observation |
| **Body sense** | Health, ammo, armor, which keys it holds, its own speed | official observation |
| **Memory map** | EgoMap grid, built only from the above + its own past actions | `submission/preprocess.py` |
| **Brain** | Small neural nets for rays (circular CNN), things (attention), vitals, map (CNN) → a GRU (short-term memory) → buttons | `submission/model.py` |
| **Hands** | 6 action groups: move, strafe, turn, shoot, use, switch weapon (so it can't press forward+back at once) | `submission/model.py` |
| **Teacher** | PPO: plays 12 games in parallel, makes rewarded actions a bit more likely each update | `training/train_ppo.py` |
| **Report card** | Rewards (training only): exit +25, key +8, locked door +4, door +0.8, kill +1, new area +0.04, damage −, death −3 | `training/rewards.py` |
| **Exam** | Holdout maps it never trained on: Freedoom MAP25–32 and 8 generated maps (plus the official test map `two_rooms`, which *is* in training, so it's a sanity check only) | `training/eval_zeroshot.py` |

**Size and speed:** 759K parameters, about 4 ms per decision (the limit is 40 ms), 355 MB RAM (the limit is 8 GB),
weights 3 MB.

## 4. Numbers to quote

| Agent | Holdout score | Exit % |
|---|---:|---:|
| Random | −13.2 | 0% |
| Plain model (no memory map) | 9.9 | 18% |
| **Our shipped agent (with EgoMap)** | **34.1** | **29%** |

- Holdout per tier (shipped): Tier 1 combat **100% exits**, Tier 2 exploration **50%**, Tier 3 key→locked
  door→exit **solved** on our unseen Tier 3 maps (needs the longer time cap), Tier 4 not yet. Unseen real Freedoom maps
  (MAP25–32): **0% exits** (0/80 runs); it explores and fights, scoring partial points. The "~11% real-map exits" we
  reported earlier all came from `two_rooms`, which is also a training map (found by teammate 3's review).
- **The memory map alone took the score from 9.9 to 34.1.** That's the proof the USP works.
- **Demo video** (`demo/agent_demo.mp4`, 81 s at 2× speed): the official test map exited in 246 steps; an unseen Tier 3 map
  with key → locked door → exit in 364 steps; a real Freedoom map explored with 6 kills. All three were the first attempt.
  The right half of the video is the agent's memory map building up live.

## 5. Questions judges will likely ask

**"Is any of this hard-coded?"** No. Every button press comes from the trained network. The memory map is only an
*input* to the network, never used to choose actions; the rules explicitly allow "learned memory modules" and
"preprocessing and encoding observations". No pathfinding, no aimbot, no `if enemy: shoot`.

**"Isn't using engine data cheating?"** We use the engine's internal state (position, kills, door state) **only
during training**, to compute rewards. The rules allow anything during training. At evaluation the agent only gets the
official observation; the reward code isn't even imported by the agent.

**"Why not Stable-Baselines3?"** The rules require our own architecture and pipeline. Our PPO is written from scratch
in PyTorch, with a recurrent (GRU) policy and per-environment minibatches so the memory sees episodes in order.

**"How does it know where it is without coordinates?"** Dead reckoning, like a ship's navigator. We measured Doom's
turning rules: the first 5 tics of a held turn are slow, after that fast, and turns are ignored for the first 7 tics
of a level. So we can compute its heading exactly from its own button presses. Position comes from its observed
velocity.

**"Is it overfitting?"** See section 6.

**"Why rays instead of pixels?"** Rays and entity lists are official, compact and fast. One decision takes ~4 ms
instead of a pixel CNN's tens of ms, and they carry exactly what matters (distances, doors, locked doors, enemies).

**"What doesn't work yet?"** Tier 4 (two locks) and big real maps. Tier 3 works but is slow, so on short time caps
it can run out of time. Position drifts 1–2 cells over long episodes. With more time: more training steps,
Prioritized Level Replay, curiosity rewards (RND).

**"What did you find hard?"**
- Windows made ViZDoom 7–10× slower when running several games (a 15.6 ms system timer). We found it and fixed it.
- A generator bug made some "Tier 3" maps without a lock. We caught it because the agent exited one without a key.
- The agent found keys too rarely, which led to tutor rounds 1 and 2.

## 6. Overfitting — our evidence

- **Generated maps can't be memorized:** every training episode on a generated map is a brand-new level. The
  holdout generated maps come from seeds training never uses, and the agent does as well or better on them as during
  training (Tier 1 100% holdout vs ~90% in training; Tier 2 50% vs ~35–55%).
- **Real maps:** we train on Freedoom MAP01–24 and test on MAP25–32. Overfitting would show as much better scores on the
  training maps than on the test maps. Measured with the shipped agent (same eval, 2 seeds):
  **training maps MAP01–08: score 14.3, 0% exits** vs **unseen MAP25–32: score −4.6, 0% exits** (10 seeds,
  teammate 3; the earlier 7.9 / 11% wrongly included `two_rooms`). It isn't memorization: an agent that memorized routes would at least finish the maps it practised on,
  and it finishes none of them. Most of the gap is difficulty: Freedoom's early maps are designed to be easier than
  its late maps (it gets 5.6 kills and explores 94 cells there vs fewer on the late maps).
  One-liner for judges: *"It scores a bit higher on the training maps, but those are Freedoom's easier early
  levels, and it doesn't finish them either. It learned general skills, not routes."*
- **Honest caveat:** our generated levels look boxier than hand-made maps, so the real risk is "overfitting to our
  generator's style". That's why a third of training is real Freedoom maps, and why we report real-map scores
  separately.
- **Checkpoint selection:** picking the best checkpoint by a small holdout set can itself overfit, so the final pick
  uses 3 seeds, and we only switch when the improvement is clear.

## 7. Who did what

- **Core** (main laptop, RTX 4050): trainer, reward shaping, map generator, EgoMap, training runs, tutor rounds, shipping.
- **Teammate 2** (RTX 3050 / Linux): evaluation script, holdout scoring, official validation on 1 pinned CPU.
- **Teammate 3**: project brief, docs that match the code, judge-facing README.

## 8. Glossary (if you get a word you don't know)

- **PPO:** the learning algorithm. The agent plays, gets rewards, and its network is nudged so the actions that went
  better than expected become more likely, but never too much in one update ("clipping").
- **GRU:** the network's short-term memory. It carries information from step to step within one level.
- **Holdout:** maps the agent never trained on. Our "exam".
- **Dead reckoning:** working out your position from your own movements instead of a GPS.
- **Curriculum:** teaching easy versions first (small key maps) before hard ones.
- **Proxy score:** our local guess at the hidden official score: exit counts most, then keys and doors, kills,
  exploration, and minus points for damage and early death.
