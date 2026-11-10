"""
world.py - course scene for the flight simulator (config/worlds/*.json).

Coordinates (cm): X = east, Y = north, Z = up. Headings: deg, seen from above, 0 = east,
clockwise positive (Tello yaw sense): 90 = south, 180 = west, -90 = north.

World holds axis-aligned boxes (walls, boards, table slabs / legs, chair, marker sign plates),
markers, floor tape lines and the human spec. It answers:
  collide(lo, hi)      -> list of solid boxes overlapping the AABB [lo, hi]
  tof_below(pos)       -> distance from pos down to the first surface below (table top or floor)
  support_z(x, y, z)   -> height of that surface
"""
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB6 = os.path.dirname(HERE)
DEFAULT_WORLD = os.path.join(LAB6, "config", "worlds", "eab06.json")


def load_jsonc(path):
    """JSON with // line comments (outside strings) and trailing commas tolerated."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    out = []
    for line in text.splitlines():
        in_str = False
        esc = False
        cut = len(line)
        for i, ch in enumerate(line):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "/" and line[i:i + 2] == "//":
                cut = i
                break
        out.append(line[:cut])
    text = "\n".join(out)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


def heading_vec(deg):
    """unit vector (x, y) of a heading (0 = east, clockwise positive)"""
    a = math.radians(deg)
    return np.array([math.cos(a), -math.sin(a)])


def rgb2bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


class Box(object):
    __slots__ = ("name", "lo", "hi", "color", "solid", "markers")

    def __init__(self, name, lo, hi, color=(200, 200, 200), solid=True):
        self.name = name
        self.lo = np.array(lo, float)
        self.hi = np.array(hi, float)
        self.color = color                 # BGR
        self.solid = solid
        self.markers = []                  # Marker objects drawn on this box


class Marker(object):
    def __init__(self, mid, pos, facing_deg, size, margin):
        self.id = int(mid)
        self.pos = np.array(pos, float)
        self.facing = float(facing_deg)
        self.size = float(size)
        self.margin = float(margin)

    def corners(self, offset=0.3, with_margin=True):
        """world corners TL, TR, BR, BL as seen from the front (4x3), pushed `offset` cm out"""
        n = heading_vec(self.facing)
        normal = np.array([n[0], n[1], 0.0])
        # facing east (normal +X) -> a viewer in front faces west -> viewer's right is north (+Y)
        right = _right_of_viewer(n)
        up = np.array([0.0, 0.0, 1.0])
        h = self.size / 2.0 + (self.margin if with_margin else 0.0)
        c = self.pos + normal * offset
        return np.array([c - right * h + up * h, c + right * h + up * h,
                         c + right * h - up * h, c - right * h - up * h])


def _right_of_viewer(n):
    """marker normal n (x, y) -> right direction of a viewer looking at the marker"""
    # viewer looks along d = -n = (cos a, -sin a); right = heading a + 90 = (-sin a, -cos a) = (d_y, -d_x)
    d = -np.asarray(n, float)
    return np.array([d[1], -d[0], 0.0])


class World(object):
    def __init__(self, path=None):
        self.path = path or DEFAULT_WORLD
        spec = load_jsonc(self.path)
        self.spec = spec
        self.name = spec.get("name", os.path.basename(self.path))
        st = spec.get("start", {})
        self.start_pos = np.array(st.get("pos", [0, 0, 0]), float)
        self.start_heading = float(st.get("heading_deg", 0.0))
        fl = spec.get("floor", {})
        self.floor_lo = np.array(fl.get("min", [-300, -300]), float)
        self.floor_hi = np.array(fl.get("max", [1000, 700]), float)
        self.floor_color = rgb2bgr(fl.get("color", [150, 150, 145]))
        self.sky_color = rgb2bgr(spec.get("sky_color", [205, 210, 215]))
        self.boxes = []
        for b in spec.get("boxes", []):
            self.boxes.append(Box(b.get("name", "box"), b["min"], b["max"],
                                  rgb2bgr(b.get("color", [200, 200, 200])), b.get("solid", True)))
        for t in spec.get("tables", []):
            self._add_table(t)
        margin = float(spec.get("marker_margin", 2.5))
        self.markers = []
        for m in spec.get("markers", []):
            mk = Marker(m["id"], m["pos"], m.get("facing_deg", 0.0), m.get("size", 15.0), margin)
            self.markers.append(mk)
            self._mount(mk, m.get("mount", "auto"))
        self.lines = []
        for ln in spec.get("lines", []):
            self.lines.append(dict(name=ln.get("name", ""), a=np.array(ln["from"], float),
                                   b=np.array(ln["to"], float), width=float(ln.get("width", 4)),
                                   color=rgb2bgr(ln.get("color", [30, 30, 30])),
                                   dashed=bool(ln.get("dashed", False))))
        self.human_spec = spec.get("human")
        self._refresh_arrays()

    # ---- building -----------------------------------------------------------
    def _add_table(self, t):
        x0, x1 = t["x"]
        y0, y1 = t["y"]
        top = float(t["top_z"])
        th = float(t.get("thickness", 4))
        leg = float(t.get("leg", 5))
        col = rgb2bgr(t.get("color", [190, 160, 120]))
        name = t.get("name", "table")
        self.boxes.append(Box(name + "_top", [x0, y0, top - th], [x1, y1, top], col))
        legc = tuple(int(c * 0.7) for c in col)
        for i, (lx, ly) in enumerate([(x0, y0), (x1 - leg, y0), (x0, y1 - leg), (x1 - leg, y1 - leg)]):
            self.boxes.append(Box("%s_leg%d" % (name, i), [lx, ly, 0], [lx + leg, ly + leg, top - th], legc))

    def _mount(self, mk, mode):
        if mode != "sign":
            n = heading_vec(mk.facing)
            for b in self.boxes:
                # face of b whose outward normal is n, containing the marker centre
                for ax in (0, 1):
                    if abs(n[ax]) < 0.99:
                        continue
                    face = b.hi[ax] if n[ax] > 0 else b.lo[ax]
                    if abs(mk.pos[ax] - face) > 2.0:
                        continue
                    o = 1 - ax
                    if b.lo[o] <= mk.pos[o] <= b.hi[o] and b.lo[2] <= mk.pos[2] <= b.hi[2]:
                        b.markers.append(mk)
                        return
        # free-hanging sign plate behind the marker (1 cm thick, white, solid)
        n = heading_vec(mk.facing)
        h = mk.size / 2.0 + mk.margin + 0.5
        back = mk.pos[:2] - n * 1.0
        lo = np.minimum(mk.pos[:2], back)
        hi = np.maximum(mk.pos[:2], back)
        side = np.array([abs(n[1]), abs(n[0])]) * h      # extent along the face
        lo2 = lo - side
        hi2 = hi + side
        b = Box("sign%d" % mk.id, [lo2[0], lo2[1], mk.pos[2] - h], [hi2[0], hi2[1], mk.pos[2] + h],
                (240, 240, 240))
        b.markers.append(mk)
        self.boxes.append(b)

    def _refresh_arrays(self):
        solid = [b for b in self.boxes if b.solid]
        self._solid = solid
        self._lo = np.array([b.lo for b in solid]) if solid else np.zeros((0, 3))
        self._hi = np.array([b.hi for b in solid]) if solid else np.zeros((0, 3))

    def marker(self, mid):
        for m in self.markers:
            if m.id == mid:
                return m
        return None

    # ---- queries ------------------------------------------------------------
    def collide(self, lo, hi, extra=()):
        """solid boxes (static + `extra` (name, lo, hi) tuples) overlapping AABB [lo, hi]"""
        hits = []
        if len(self._lo):
            m = np.all((self._lo < hi) & (self._hi > lo), axis=1)
            hits = [self._solid[i] for i in np.nonzero(m)[0]]
        for e in extra:
            if np.all((e.lo < hi) & (e.hi > lo)):
                hits.append(e)
        return hits

    def support_z(self, x, y, z, half=0.0):
        """top of the highest surface at (x, y) (footprint +- half) that is not above z (+0.5); floor = 0"""
        if not len(self._lo):
            return 0.0
        m = ((self._lo[:, 0] <= x + half) & (self._hi[:, 0] >= x - half) &
             (self._lo[:, 1] <= y + half) & (self._hi[:, 1] >= y - half) & (self._hi[:, 2] <= z + 0.5))
        if not np.any(m):
            return 0.0
        return max(0.0, float(self._hi[m, 2].max()))

    def tof_below(self, pos):
        """distance (cm) from pos straight down to the first surface (table top or floor)"""
        return float(pos[2]) - self.support_z(pos[0], pos[1], pos[2])
