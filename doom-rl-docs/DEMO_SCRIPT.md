# 3-Minute Live Demo Script

All facts here are from `TASKS_TEAMMATE_DOCS.md` §4 and `REVIEW_CHEATSHEET.md`. Times are targets; it's fine to run a
few seconds over.

---

### 0:00–0:30 — Pitch

> "This agent plays DOOM levels it has never seen, using only what the game lets it sense. Nothing is hand-written:
> every button press comes from the trained network. It sees 64 distance rays and a list of visible things, and it
> **draws its own map in its head** while it plays. It works out where it is by counting its own turns and steps,
> so it can remember a locked door and come back once it has the key. We trained it with our own PPO on an
> **endless stream of generated levels** plus real Freedoom maps."

### 0:30–1:30 — What's on screen (the demo video)

> **TODO(core):** `REVIEW_CHEATSHEET.md` refers to `demo/agent_demo.mp4`, but it isn't in the repo yet (checked at
> 19:25 IST). Push it, or use the live command at the bottom instead.

Point at the screen as you talk:

- **Left half: the game.** "This is exactly what's being played. The agent only gets the official rays, entities
  and vitals, not this picture."
- **Right half: the agent's memory map (EgoMap), building up live.** "Every step, what it senses is drawn into this
  grid: where it has been, walls, doors."
- **The red locked door.** "Here it walks past a locked door it can't open yet. Watch: the door stays on the map
  after it leaves."
- **The yellow key.** "Now it finds the key in another branch."
- **Coming back.** "With the key in hand, it heads back to the door it remembered, unlocks it, and reaches the exit.
  That's Tier 3, key → locked door → exit, on a level it never trained on."
- From the cheat-sheet, if the video shows them: the official test map (a training map) is exited in 246 steps, the unseen Tier 3
  map is solved in 364 steps, and on a real Freedoom map it explores and gets 6 kills. All three were first attempts.

### 1:30–2:30 — Three things we built (one number each)

1. **EgoMap, a self-built spatial memory.** "The game never tells it where it is. We measured Doom's turning rules
   tic by tic, so it tracks its heading from its own button presses with **0.0° error over 600 steps**."
2. **Infinite generated levels with a solvability proof.** "Every training map load is a brand-new level, in four
   difficulty tiers. For key levels, a self-test *proves* the key is reachable and the exit opens only after
   unlocking. On key levels it has never seen, it reaches the exit in **100% and 50%** of runs with the longer time
   cap."
3. **A 'tutor' training loop.** "When Tier 3 stalled at about **0.1 keys per episode**, we read the logs, raised the
   key reward from 3 to 8, added small 3-room key levels, and resumed from the same weights."

Backup numbers if asked:
- **Size and speed:** 759K parameters, p50 **4.3 ms** per decision (the limit is 40 ms), 355 MB RAM.
- **Holdout score:** random **−13.2**, plain model **9.9**, shipped model with EgoMap **34.1** (holdout v1).
  The shipped model also trained longer (645k vs. 184k steps), so don't say the map alone explains the whole gap.

### 2:30–3:00 — Limitations and next steps

> "What doesn't work yet: Tier 4, with two locks, is still at 0% exits. On real Freedoom levels it has never seen, it
> doesn't reach the exit yet, though it explores and fights there for partial points. Tier 3 works but is slow, so short time caps can run out. Its
> position estimate drifts 1–2 grid cells over a long level; its heading stays exact. With more time: many more
> training steps than our ~0.7M, Prioritized Level Replay, and curiosity rewards."

---

## If they ask for a live game

Run this from the repo root on any laptop with the environment set up:

```powershell
uv run python training/eval_zeroshot.py --weights submission/weights/best_agent.pt --maps two_rooms --seeds 0 --device cpu
```

It plays the official test map with the shipped weights and prints the result: exit, keys, doors, kills, cells,
health and ms/step.
