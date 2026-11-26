"""Training-only geodesic progress shaping (tutor round 6; teammate 3's proposal 23:20, item 1).

Phi(s) = -(walking distance to the nearest key the agent still needs, or to the exit once it has them all) / SCALE.
The shaping reward is Phi(s') - Phi(s): it pays for real progress along the map's walkable floor, and it telescopes,
so looping or standing still earns nothing. (A discounted gamma*Phi(s') - Phi(s) with gamma < 1 would pay a little for
standing still far from the goal, so gamma = 1 is used.) Walking distances come from a BFS over a 16-unit grid built
from the level's walls, step heights and doors. This is training-time scaffolding only (docs/RULES.md §1); the
submission never imports it and never sees these distances.
"""

from __future__ import annotations

import struct
from collections import deque

import numpy as np

from doom_env.mapinfo import load_mapinfo, parse_udmf
from doom_env.wad import map_lumps, read_wad

CELL = 16.0  # grid resolution in map units (players are 32 wide; corridors >= 64)
SCALE = 1000.0  # 1,000 units of progress = +1 reward
# Height changes don't block: Doom routes go over drop-offs and up lifts, and treating >24-unit steps as walls left the
# first goal unreachable on 8/20 training maps (12/20 -> 17/20 reachable with this setting; training/check_geodesic.py).
STEP_MAX = float("inf")
HEADROOM = 56.0  # player height
SEED_RADIUS = 48.0  # free cells within this radius of a target are BFS sources (switch exits sit on walls)
DOOM_EXITS = (11, 51, 52, 124, 197, 198)
UDMF_EXITS = (243, 244)  # Exit_Normal, Exit_Secret (Hexen/ZDoom numbering, used by our generated maps)
KEY_INDEX = {5: 0, 6: 1, 13: 2, 40: 3, 39: 4, 38: 5}  # thing type -> doom_env KeyTracker index (vitals order)


def _targets(wad: str, map_name: str) -> tuple[list, list]:
    """(exit line midpoints, [(x, y, key index)]) read from the WAD (binary Doom format or UDMF)."""
    lumps = map_lumps(read_wad(wad), map_name)
    exits, keys = [], []
    if "TEXTMAP" in lumps:
        b = parse_udmf(lumps["TEXTMAP"].decode("utf-8", "replace"))
        verts = [(float(v.get("x", 0)), float(v.get("y", 0))) for v in b.get("vertex", [])]
        for ln in b.get("linedef", []):
            if int(ln.get("special", 0) or 0) in UDMF_EXITS:
                (x1, y1), (x2, y2) = verts[int(ln["v1"])], verts[int(ln["v2"])]
                exits.append(((x1 + x2) / 2, (y1 + y2) / 2))
        for th in b.get("thing", []):
            if int(th.get("type", 0)) in KEY_INDEX:
                keys.append((float(th.get("x", 0)), float(th.get("y", 0)), KEY_INDEX[int(th["type"])]))
    else:
        vx = [struct.unpack_from("<hh", lumps["VERTEXES"], i) for i in range(0, len(lumps["VERTEXES"]), 4)]
        for i in range(0, len(lumps["LINEDEFS"]), 14):
            v1, v2, _fl, sp, _tag, _f, _b = struct.unpack_from("<HHHHhHH", lumps["LINEDEFS"], i)
            if sp in DOOM_EXITS:
                exits.append(((vx[v1][0] + vx[v2][0]) / 2, (vx[v1][1] + vx[v2][1]) / 2))
        for i in range(0, len(lumps["THINGS"]), 10):
            x, y, _a, t, _f = struct.unpack_from("<hhhhh", lumps["THINGS"], i)
            if t in KEY_INDEX:
                keys.append((float(x), float(y), KEY_INDEX[t]))
    return exits, keys


class GeoField:
    """Walkable grid + lazily computed BFS distance fields for one level."""

    def __init__(self, wad: str, map_name: str, floor: np.ndarray, ceil: np.ndarray):
        info = load_mapinfo(wad, map_name)
        g = info.lines
        self.exits, self.keys = _targets(wad, map_name)
        pts = np.concatenate([g.p1, g.p2])
        self.x0, self.y0 = pts.min(axis=0) - 4 * CELL
        self.w = int((pts[:, 0].max() - self.x0) / CELL) + 5
        self.h = int((pts[:, 1].max() - self.y0) / CELL) + 5
        doors = set(info.door_sectors)
        n = len(floor)

        def blocks(i: int) -> bool:
            f, b = int(g.front[i]), int(g.back[i])
            if b < 0 or f < 0 or g.impassable[i] or f >= n or b >= n:
                return True
            if f in doors or b in doors:
                return False  # doors are closed at rest but open on USE
            if abs(floor[f] - floor[b]) > STEP_MAX:
                return True
            return min(ceil[f], ceil[b]) - max(floor[f], floor[b]) < HEADROOM

        wall = np.zeros((self.h, self.w), bool)
        for i in range(len(g.p1)):
            if blocks(i):
                (x1, y1), (x2, y2) = g.p1[i], g.p2[i]
                k = int(np.hypot(x2 - x1, y2 - y1) / 4.0) + 2  # dense samples: no 4-connected gaps
                t = np.linspace(0.0, 1.0, k)
                cx = ((x1 + (x2 - x1) * t - self.x0) / CELL).astype(int)
                cy = ((y1 + (y2 - y1) * t - self.y0) / CELL).astype(int)
                wall[cy, cx] = True
        self.wall = wall
        self._fields: dict[tuple, np.ndarray] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.y0) / CELL), int((x - self.x0) / CELL)

    def _field(self, sources: tuple) -> np.ndarray:
        """Multi-source 4-connected BFS distance (in cells; -1 = unreachable) from free cells near `sources`."""
        if sources not in self._fields:
            h, w, wall = self.h, self.w, self.wall.ravel()
            dist = np.full(h * w, -1, np.int32)
            q = deque()
            r = int(SEED_RADIUS / CELL)
            for x, y in sources:
                cy, cx = self._cell(x, y)
                for yy in range(max(cy - r, 0), min(cy + r + 1, h)):
                    for xx in range(max(cx - r, 0), min(cx + r + 1, w)):
                        j = yy * w + xx
                        if not wall[j] and dist[j] < 0:
                            dist[j] = 0
                            q.append(j)
            while q:
                j = q.popleft()
                d = dist[j] + 1
                x = j % w
                for k in (j - w, j + w, j - 1 if x > 0 else -1, j + 1 if x < w - 1 else -1):
                    if 0 <= k < h * w and dist[k] < 0 and not wall[k]:
                        dist[k] = d
                        q.append(k)
            self._fields[sources] = dist.reshape(h, w)
        return self._fields[sources]

    def target(self, held: np.ndarray) -> tuple:
        """BFS sources for the current goal: keys not yet held, else the exits."""
        missing = tuple((x, y) for x, y, k in self.keys if held[k] < 0.5)
        return missing or tuple(self.exits)

    def phi(self, x: float, y: float, sources: tuple) -> float | None:
        """-walking distance / SCALE, or None where the goal is unreachable or there is no goal."""
        if not sources:
            return None
        f = self._field(sources)
        cy, cx = self._cell(x, y)
        win = f[max(cy - 1, 0): cy + 2, max(cx - 1, 0): cx + 2]  # the player's own cell can be a wall cell
        ok = win[win >= 0]
        return None if ok.size == 0 else -float(ok.min()) * CELL / SCALE
