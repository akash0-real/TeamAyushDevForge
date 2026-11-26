"""Procedural training maps ("infinite" map pool): random rooms + corridors + doors + keys, written as
node-less UDMF WADs (namespace "zdoom"), the same format as the template's own test maps.

Tiers (match the evaluation categories):
  1  combat:   1-2 rooms, several enemies, exit
  2  explore:  4-7 rooms in a tree (+ loops), doors, enemies in some rooms, exit in the farthest room
  3  keys:     like 2, the exit room is behind a locked door, its key is in another branch (backtracking)
  4  gauntlet: 7-11 rooms, two locks (key 1 is itself behind lock 2), more enemies, fewer pickups

Training uses seeds < 1e9. Seeds >= HOLDOUT_SEED_BASE are reserved for held-out evaluation maps.

    python maps/gen_maps.py --tier 3 --seed 1 --out my.wad     # one map
    python maps/gen_maps.py --selftest                         # solvability + loads in doom_env
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "env"))
from doom_env.wad import Lump, write_wad  # noqa: E402

HOLDOUT_SEED_BASE = 1_000_000_000
CELL = 64  # map units per grid cell
SLOT = 11  # cells per room slot

# Freedoom 2 thing ids (same doomednums as Doom 2)
ENEMIES_EASY = (3004, 9, 3001)  # zombieman, shotgun guy, imp
ENEMIES_HARD = (3004, 9, 3001, 3001, 3002)  # + demon (melee)
ITEMS = (2011, 2012, 2007, 2008, 2018)  # stimpack, medikit, clip, shells, green armor
SHOTGUN = 2001
# lock id (ZDoom) -> key card thing
LOCKS = {1: 13, 2: 5, 3: 6}  # red, blue, yellow card
# shortcut: keycards only (locks 1-3); skull keys (locks 129-131) untested in doom_env, add once verified.

WALLS = ("STARTAN2", "STARTAN2", "BROWN1", "GRAY1", "STONE2")
FLOORS = ("FLOOR0_1", "FLOOR4_8", "FLAT5_4", "FLOOR5_1")
CEILS = ("CEIL1_1", "CEIL3_5", "FLAT1")


class MapSpec:
    """Grid + metadata of one generated map (kept for the self-test)."""

    def __init__(self) -> None:
        self.grid: list[list[int]] = []  # region id per cell, 0 = solid
        self.sectors: list[dict] = []  # per region id - 1
        self.things: list[tuple[float, float, int, int]] = []  # x, y, type, angle
        self.start: tuple[int, int] = (0, 0)
        self.exit_cells: set[tuple[int, int]] = set()
        self.key_cells: dict[int, tuple[int, int]] = {}  # lock -> cell
        self.door_lock: dict[int, int] = {}  # door region -> lock (0 = plain door)


def _rooms_layout(rng: random.Random, n_rooms: int):
    """Grow n slots on a slot grid; return slots, tree edges (parent, child) and room rects."""
    slots = [(0, 0)]
    edges = []
    while len(slots) < n_rooms:
        px, py = rng.choice(slots)
        dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        s = (px + dx, py + dy)
        if s not in slots:
            edges.append((slots.index((px, py)), len(slots)))
            slots.append(s)
    minx, miny = min(s[0] for s in slots), min(s[1] for s in slots)
    slots = [(x - minx, y - miny) for x, y in slots]
    rects = []
    for sx, sy in slots:
        w, h = rng.randint(4, 8), rng.randint(4, 8)
        x0 = sx * SLOT + 1 + rng.randint(0, SLOT - 2 - w)
        y0 = sy * SLOT + 1 + rng.randint(0, SLOT - 2 - h)
        rects.append((x0, y0, w, h))
    return slots, edges, rects


def _tree_depths(n: int, edges: list[tuple[int, int]]) -> list[int]:
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    depth = [-1] * n
    depth[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


def generate(tier: int, seed: int) -> MapSpec:
    rng = random.Random(seed * 10 + tier)
    n_rooms = {1: rng.randint(1, 2), 2: rng.randint(4, 7), 3: rng.randint(4, 7), 4: rng.randint(7, 11)}[tier]
    # Tier 3 needs 2 dead ends (exit + key), tier 4 needs 3 (exit + 2 keys); a chain layout has only 1, so re-roll.
    need_leaves = {3: 2, 4: 3}.get(tier, 0)
    while True:
        slots, edges, rects = _rooms_layout(rng, n_rooms)
        depth = _tree_depths(n_rooms, edges)
        children = {i: [b for a, b in edges if a == i] for i in range(n_rooms)}
        leaves = [i for i in range(1, n_rooms) if not children[i]]
        if len(leaves) >= need_leaves:
            break
    W = (max(s[0] for s in slots) + 1) * SLOT + 2
    H = (max(s[1] for s in slots) + 1) * SLOT + 2
    m = MapSpec()
    m.grid = [[0] * W for _ in range(H)]

    def new_sector(ceil: int, kind: str) -> int:
        m.sectors.append({"ceil": ceil, "kind": kind, "floor_t": rng.choice(FLOORS), "ceil_t": rng.choice(CEILS),
                          "light": rng.choice((144, 160, 176, 192, 208, 224))})
        return len(m.sectors)

    room_ids = []
    for x0, y0, w, h in rects:
        rid = new_sector(rng.choice((128, 160, 192)), "room")
        room_ids.append(rid)
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                m.grid[y][x] = rid

    # Locks: exit room = deepest leaf; lock 1 gates it, key 1 sits in another deep leaf (tier 4: also locked).
    exit_room = max(leaves, key=lambda i: depth[i]) if leaves else 0
    locked_rooms: list[tuple[int, int]] = []  # (room behind the lock, lock id)
    key_rooms: dict[int, int] = {}  # lock id -> room holding its key
    if tier >= 3 and len(leaves) >= 2:
        colors = rng.sample(sorted(LOCKS), 2)
        others = sorted((i for i in leaves if i != exit_room), key=lambda i: -depth[i])
        locked_rooms.append((exit_room, colors[0]))
        key_rooms[colors[0]] = others[0]
        if tier == 4 and len(others) >= 2:
            locked_rooms.append((others[0], colors[1]))
            key_rooms[colors[1]] = others[1]
    gated = {r for r, _ in locked_rooms}

    # Extra loop edges between adjacent slots (never into gated rooms, or they'd bypass the lock).
    all_edges = list(edges)
    if tier in (2, 4):
        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                adjacent = abs(slots[i][0] - slots[j][0]) + abs(slots[i][1] - slots[j][1]) == 1
                if adjacent and (i, j) not in edges and (j, i) not in edges and not {i, j} & gated and rng.random() < 0.3:
                    all_edges.append((i, j))

    def center(i: int) -> tuple[int, int]:
        x0, y0, w, h = rects[i]
        return x0 + w // 2, y0 + h // 2

    door_p = {1: 0.0, 2: 0.6, 3: 0.5, 4: 0.5}[tier]
    lock_of_room = dict(locked_rooms)
    for a, b in all_edges:
        (ax, ay), (bx, by) = center(a), center(b)
        path = []
        if slots[a][1] == slots[b][1]:  # same row: horizontal then vertical
            for x in range(min(ax, bx), max(ax, bx) + 1):
                path += [(x, ay), (x, ay + 1)]
            for y in range(min(ay, by), max(ay, by) + 2):
                path += [(bx, y), (bx + 1, y)]
        else:  # same column: vertical then horizontal
            for y in range(min(ay, by), max(ay, by) + 1):
                path += [(ax, y), (ax + 1, y)]
            for x in range(min(ax, bx), max(ax, bx) + 2):
                path += [(x, by), (x, by + 1)]
        cid = new_sector(128, "corridor")
        cells = []
        for x, y in path:
            if m.grid[y][x] == 0:
                m.grid[y][x] = cid
                cells.append((x, y))
        lock = lock_of_room.get(b, 0)
        if lock or rng.random() < door_p:
            # door slab = this corridor's cells touching room b (the child, so a lock gates b's side)
            rb = room_ids[b]
            slab = [(x, y) for x, y in cells
                    if any(m.grid[y + dy][x + dx] == rb for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
            if slab:
                did = new_sector(0, "door")
                m.door_lock[did] = lock
                for x, y in slab:
                    m.grid[y][x] = did
            elif lock:
                raise RuntimeError("locked edge without a door slab")

    def free_cell(room: int, taken: set) -> tuple[int, int] | None:
        """Random interior cell of a room not used yet; None when the room is full (small rooms)."""
        x0, y0, w, h = rects[room]
        free = [(x, y) for y in range(y0 + 1, y0 + h - 1) for x in range(x0 + 1, x0 + w - 1) if (x, y) not in taken]
        if not free:
            return None
        c = rng.choice(free)
        taken.add(c)
        return c

    taken: set[tuple[int, int]] = set()
    sx, sy = center(0)
    m.start = (sx, sy)
    taken |= {(sx + dx, sy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    # Exit pad: one cell in the exit room (or far corner of the only room), its own sector, walk-over exit.
    if n_rooms == 1:
        x0, y0, w, h = rects[0]
        ex, ey = (x0 + w - 2, y0 + h - 2) if (sx - x0) < w / 2 else (x0 + 1, y0 + 1)
        taken.add((ex, ey))
    else:
        ex, ey = free_cell(exit_room, taken)  # exit room is a leaf != start, so it has free cells
    pad = new_sector(m.sectors[m.grid[ey][ex] - 1]["ceil"], "exit")
    m.grid[ey][ex] = pad
    m.exit_cells = {(ex, ey)}

    def put(cell: tuple[int, int] | None, ttype: int, angle: int = 0) -> None:
        if cell is not None:
            m.things.append(((cell[0] + 0.5) * CELL, (cell[1] + 0.5) * CELL, ttype, angle))

    put(m.start, 1, rng.choice(range(0, 360, 45)))
    for lock, room in key_rooms.items():  # keys first: key rooms are leaves, never the (crowded) start room
        c = free_cell(room, taken)
        assert c is not None, "no room for a key"
        m.key_cells[lock] = c
        put(c, LOCKS[lock])

    enemies = ENEMIES_HARD if tier == 4 else ENEMIES_EASY
    for i in range(n_rooms):
        if tier == 1:
            n_en = rng.randint(2, 4) if (i > 0 or n_rooms == 1) else 0
        elif tier == 4:
            n_en = rng.randint(1, 3) if i > 0 else 0
        else:
            n_en = rng.randint(1, 2) if (i > 0 and rng.random() < 0.6) else 0
        for _ in range(n_en):
            c = free_cell(i, taken)
            if c is None or (n_rooms == 1 and abs(c[0] - sx) + abs(c[1] - sy) < 3):
                continue
            put(c, rng.choice(enemies), rng.choice(range(0, 360, 45)))
        n_items = rng.randint(0, 1) if tier == 4 else rng.randint(1, 2)
        for _ in range(n_items):
            put(free_cell(i, taken), rng.choice(ITEMS))
    if tier >= 2 and rng.random() < 0.7:
        put(free_cell(rng.randrange(n_rooms), taken), SHOTGUN)
    return m


def _edges(m: MapSpec):
    """Unit cell-boundary edges -> merged linedefs: (x1, y1, x2, y2, front, back)."""
    H, W = len(m.grid), len(m.grid[0])
    kind = lambda r: m.sectors[r - 1]["kind"] if r else "solid"  # noqa: E731
    raw = []
    for y in range(H):
        for x in range(W):
            r = m.grid[y][x]
            if not r:
                continue
            x0, y0, x1, y1 = x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL
            # (neighbor, segment with this cell on the right side = Doom front side)
            for (dx, dy), seg in (((0, 1), (x0, y1, x1, y1)), ((0, -1), (x1, y0, x0, y0)),
                                  ((-1, 0), (x0, y0, x0, y1)), ((1, 0), (x1, y1, x1, y0))):
                n = m.grid[y + dy][x + dx] if 0 <= y + dy < H and 0 <= x + dx < W else 0
                if n == 0:
                    raw.append((*seg, r, 0))
                elif n != r and (dx, dy) in ((1, 0), (0, 1)):  # each two-sided edge once
                    if kind(r) == "door" and kind(n) != "door":
                        raw.append((seg[2], seg[3], seg[0], seg[1], n, r))  # flip: door must be the back side
                    else:
                        raw.append((*seg, r, n))
    # merge collinear contiguous segments with identical sides
    groups: dict = {}
    for x1, y1, x2, y2, f, b in raw:
        horiz = y1 == y2
        key = (horiz, y1 if horiz else x1, (x2 > x1) if horiz else (y2 > y1), f, b)
        groups.setdefault(key, []).append((x1, y1, x2, y2))
    out = []
    for (horiz, _, fwd, f, b), segs in groups.items():
        segs.sort(key=lambda s: (s[0], s[1]) if fwd else (-s[0], -s[1]))
        cur = list(segs[0])
        for s in segs[1:]:
            if (s[0], s[1]) == (cur[2], cur[3]):
                cur[2], cur[3] = s[2], s[3]
            else:
                out.append((*cur, f, b))
                cur = list(s)
        out.append((*cur, f, b))
    return out


def to_textmap(m: MapSpec, seed: int) -> str:
    rng = random.Random(seed)
    wall = rng.choice(WALLS)
    L = ['namespace = "zdoom";']
    for s in m.sectors:
        ceil = 0 if s["kind"] == "door" else s["ceil"]
        L.append(f'sector {{ texturefloor = "{s["floor_t"]}"; textureceiling = "{s["ceil_t"]}"; '
                 f'heightfloor = 0; heightceiling = {ceil}; lightlevel = {s["light"]}; }}')
    verts: dict[tuple[int, int], int] = {}
    vlines, slines, llines = [], [], []

    def v(x: int, y: int) -> int:
        if (x, y) not in verts:
            verts[(x, y)] = len(verts)
            vlines.append(f"vertex {{ x = {float(x)}; y = {float(y)}; }}")
        return verts[(x, y)]

    def side(sector: int, mid: str, top: str) -> int:
        slines.append(f'sidedef {{ sector = {sector - 1}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "-"; }}')
        return len(slines) - 1

    for x1, y1, x2, y2, f, b in _edges(m):
        a, c = v(x1, y1), v(x2, y2)
        if b == 0:
            llines.append(f"linedef {{ v1 = {a}; v2 = {c}; sidefront = {side(f, wall, '-')}; blocking = true; }}")
            continue
        fk, bk = m.sectors[f - 1]["kind"], m.sectors[b - 1]["kind"]
        sf = side(f, "-", "BIGDOOR2" if bk == "door" else wall)
        sb = side(b, "-", wall)
        extra = ""
        if bk == "door":
            lock = m.door_lock[b]
            if lock:
                extra = f" special = 13; arg0 = 0; arg1 = 16; arg2 = 150; arg3 = {lock}; playeruse = true; repeatspecial = true;"
            else:
                extra = " special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true;"
        elif "exit" in (fk, bk):
            extra = " special = 243; playercross = true;"
        llines.append(f"linedef {{ v1 = {a}; v2 = {c}; sidefront = {sf}; sideback = {sb}; twosided = true;{extra} }}")
    skill = "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true;"
    things = [f"thing {{ x = {x}; y = {y}; angle = {ang}; type = {t}; {skill} }}" for x, y, t, ang in m.things]
    return "\n".join(L + vlines + slines + llines + things) + "\n"


def write_map(tier: int, seed: int, path: str | Path) -> MapSpec:
    m = generate(tier, seed)
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", to_textmap(m, seed).encode()), Lump("ENDMAP", b"")])
    return m


# ------------------------------------------------------------------ self-test
def _reachable(m: MapSpec, held: set[int]) -> set[tuple[int, int]]:
    seen, q = {m.start}, deque([m.start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            r = m.grid[n[1]][n[0]]
            if r and n not in seen and m.door_lock.get(r, 0) in held | {0}:
                seen.add(n)
                q.append(n)
    return seen


def check_solvable(m: MapSpec) -> None:
    held: set[int] = set()
    while True:
        reach = _reachable(m, held)
        new = {lock for lock, c in m.key_cells.items() if c in reach} - held
        if not new:
            break
        held |= new
    assert m.exit_cells & reach, "exit unreachable even with every reachable key"
    if m.key_cells:
        assert not (m.exit_cells & _reachable(m, set())), "exit reachable without keys: lock is bypassable"


def selftest() -> None:
    import tempfile

    import numpy as np
    from doom_env import make_env
    from doom_env.mapinfo import load_mapinfo

    tmp = Path(tempfile.mkdtemp())
    for tier in (1, 2, 3, 4):
        for seed in list(range(25)) + [HOLDOUT_SEED_BASE + k for k in (1, 2)]:
            m = generate(tier, seed)
            check_solvable(m)
            n_locks = sum(1 for lk in m.door_lock.values() if lk)
            assert n_locks == {3: 1, 4: 2}.get(tier, 0), (tier, seed, n_locks)
        for seed in range(3):
            p = tmp / f"t{tier}_{seed}.wad"
            m = write_map(tier, seed, p)
            info = load_mapinfo(p, "MAP01")
            want_locked = sum(1 for lk in m.door_lock.values() if lk)
            got_locked = sum(1 for lk in info.door_sectors.values() if lk > 0)
            assert got_locked == want_locked, (tier, seed, got_locked, want_locked)
            assert len(info.door_sectors) == len(m.door_lock), (tier, seed, info.door_sectors, m.door_lock)
            env = make_env(str(p), "MAP01", episode_ticks=400, render_buffers=False)
            obs, _ = env.reset(seed=0)
            rng = np.random.default_rng(seed)
            for _ in range(60):
                obs, _, term, trunc, _ = env.step(rng.integers(0, 2, 9))
                if term or trunc:
                    break
            env.close()
            print(f"tier {tier} seed {seed}: {len(m.sectors)} sectors, {len(m.door_lock)} doors "
                  f"({want_locked} locked), {len(m.things)} things  OK", flush=True)
    print("selftest passed: 100 maps solvable, 12 maps load in doom_env with matching door/lock detection")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        write_map(a.tier, a.seed, a.out or f"gen_t{a.tier}_{a.seed}.wad")
