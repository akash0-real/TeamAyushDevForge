"""v6 highlight reel: one exiting run per tier on unseen holdout maps, captioned, 2x speed.
usage: python training/results/t3_checks/demo_reel.py <out.mp4>   (needs ffmpeg on PATH)"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "env")]
from doom_env import make_env  # noqa: E402
from agent import DoomAgent  # noqa: E402

CLIPS = [("maps/holdout/gen_t1_1.wad", "Tier 1 - combat"), ("maps/holdout/gen_t2_1.wad", "Tier 2 - exploration"),
         ("maps/holdout/gen_t3_1.wad", "Tier 3 - key, locked door, exit")]
FONT = "arial.ttf"  # ffmpeg runs with cwd=C:/Windows/Fonts: a drive colon breaks the filter syntax
out = Path(sys.argv[1]).resolve(); tmp = Path(tempfile.mkdtemp())
agent = None
parts = []
for i, (wad, title) in enumerate(CLIPS):
    for seed in range(4):  # first seed whose run reaches the exit
        env = make_env(str(ROOT / wad), "MAP01", episode_ticks=4200, render_buffers=True)
        if agent is None:
            agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        obs, _ = env.reset(seed=seed); agent.reset(); torch.manual_seed(seed)
        frames, term, trunc = [obs["rgb"]], False, False
        while not (term or trunc):
            obs, _, term, trunc, _ = env.step(agent.act(obs)); frames.append(obs["rgb"])
        won = bool(env.privileged_state()["exit_reached"]); env.close()
        if won:
            break
    cap = f"v6 (unseen map)   {title}   {'EXIT' if won else 'no exit'} in {len(frames) - 1} steps"
    cap = cap.replace(",", " /").replace(":", "")  # commas/colons break ffmpeg's filter syntax
    clip = tmp / f"c{i}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "160x120",
                           "-r", "17.5", "-i", "-", "-vf",
                           f"scale=960:720:flags=neighbor,drawtext=fontfile={FONT}:text='{cap}':x=16:y=16:fontsize=26:"
                           "fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], stdin=subprocess.PIPE, cwd="C:/Windows/Fonts")
    ff.stdin.write(np.ascontiguousarray(np.stack(frames), dtype=np.uint8).tobytes()); ff.stdin.close(); ff.wait()
    parts.append(clip); print(cap, flush=True)
lst = tmp / "list.txt"; lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
assert out.stat().st_size > 0
print("wrote", out)
