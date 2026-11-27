# Team Ayush — Gameplay & Live EgoMap Demo Videos

Side-by-side recordings (left: first-person DOOM + live neural action buttons; right: live 128×128×6 EgoMap spatial memory grid with zero engine coordinates).

- **`FULL_DEMO_SHOWCASE.mp4`** (17 MB) — All 4 clips stitched together.
- **`03_tier3_keycard_backtrack.mp4`** (7.3 MB) — Unseen Tier-3 keycard → red locked door → backtrack → unlock → **`EXIT REACHED!` in 516 steps** (2 kills, 1 key, 1 locked door unlocked, 76 cells).
- **`02_tier2_unseen_exit.mp4`** (3.7 MB) — Unseen Tier-2 multi-room (`gen_t2_1`) → **`EXIT REACHED!` in 123 steps** (`egomap_s77` 31.00 checkpoint).

*(Stored on the orphan `demos` branch to keep `main` lean at ~3.6 MB for `validate.py` and the scorer.)*

- **`SHOWCASE_V11_FINAL.mp4`** (8.6 MB) — Shipped **`v11`** (`t2r7b_0447`, side-by-side DOOM + live 6-channel EgoMap): Unseen T1 exit (`16` steps), unseen T2 exit (`70` steps), unseen T3 key→lock exit (`171` steps, 1 key), **unseen T4 two-key two-lock exit (`505` steps, 4 kills, 2 keys, 50 HP, `gen_t4_exam2` seed 13)**.

- `ALL_DOOM_DEMOS_COMPILED.mp4` (72 MB, 5m 34s, 1280x660): full chronological compilation combining `SHOWCASE_V11_FINAL.mp4` (0:00-0:48) + `SHOWCASE_V3_SHIPPED.mp4` (0:48-3:40) + `FULL_DEMO_SHOWCASE.mp4` (3:40-5:34).
