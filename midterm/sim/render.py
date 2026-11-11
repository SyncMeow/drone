"""
render.py - small software renderer (numpy + OpenCV) for the flight simulator.

Draws flat-shaded boxes, floor + tape lines (+ optional grid) and ArUco marker textures for a pinhole
camera, optionally followed by lens distortion (calibrate.xml) via a precomputed remap.
Occlusion: objects (boxes / multi-part figures) are ordered back to front with a separating-axis test
on their bounding boxes (exact for non-overlapping axis-aligned boxes), back faces are culled, and
faces inside a multi-part object are sorted by depth.

Camera convention (OpenCV): x right, y down, z forward. R = world->camera rotation (rows = camera
right, down, forward expressed in world coordinates), pos = camera position (world, cm).
"""
import math

import cv2
import numpy as np

from physics import body_axes

ARUCO_DICT = cv2.aruco.DICT_4X4_250
_LIGHT = np.array([0.35, -0.55, 0.76])
_LIGHT /= np.linalg.norm(_LIGHT)

# box corner i = x | y << 1 | z << 2; faces as cyclic corner lists + outward normal
_BOX_FACES = [((0, 2, 6, 4), (-1, 0, 0)), ((1, 3, 7, 5), (1, 0, 0)),
              ((0, 1, 5, 4), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
              ((0, 1, 3, 2), (0, 0, -1)), ((4, 5, 7, 6), (0, 0, 1))]
_BITS = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], float)


# ---------------------------------------------------------------- cameras
def look_at(eye, target):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= max(np.linalg.norm(f), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:                      # looking straight down / up
        r = np.array([0.0, -1.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.array([r, d, f])


def drone_camera(pos, psi, pitch, roll, cam_pitch, forward_cm=4.0, up_cm=2.0):
    """camera pose of the drone: body pitch / roll tilt the camera too"""
    f, r, u = body_axes(psi, pitch, roll)
    c = math.radians(cam_pitch)
    fc = f * math.cos(c) - u * math.sin(c)
    uc = u * math.cos(c) + f * math.sin(c)
    p = np.asarray(pos, float) + f * forward_cm + u * up_cm
    return p, np.array([r, -uc, fc])


def intrinsics_fov(w, h, hfov_deg):
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, (w - 1) / 2.0], [0, fx, (h - 1) / 2.0], [0, 0, 1.0]])


# ---------------------------------------------------------------- textures
_TEX = {}


def marker_texture(mid, size_cm=15.0, margin_cm=2.5, cell_px=20):
    key = (mid, size_cm, margin_cm, cell_px)
    if key not in _TEX:
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        n = 6 * cell_px
        if hasattr(cv2.aruco, "generateImageMarker"):     # OpenCV >= 4.7
            m = cv2.aruco.generateImageMarker(d, int(mid), n)
        else:                                            # OpenCV < 4.7
            m = cv2.aruco.drawMarker(d, int(mid), n)
        b = int(round(margin_cm / size_cm * n))
        m = cv2.copyMakeBorder(m, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
        m = (m.astype(np.float32) * (225.0 / 255.0) + 18).astype(np.uint8)   # paper white / ink black
        _TEX[key] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return _TEX[key]


# ---------------------------------------------------------------- scene objects
class Obj(object):
    """renderable: verts (N, 3) + faces [(idx, normal, color, tex, bias)], multi = sort faces by depth"""
    __slots__ = ("name", "verts", "faces", "lo", "hi", "multi", "highlight")

    def __init__(self, name, multi=False):
        self.name = name
        self.verts = np.zeros((0, 3))
        self.faces = []
        self.multi = multi
        self.highlight = False

    def _add_verts(self, v):
        i0 = len(self.verts)
        self.verts = np.vstack([self.verts, v]) if len(self.verts) else np.array(v, float)
        return i0

    def add_box(self, lo, hi, color, R=None, center=None):
        """axis-aligned box lo..hi, or (R 3x3 columns = box axes, center, size = hi - lo)"""
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        if R is None:
            v = lo + _BITS * (hi - lo)
            normals = [np.array(n, float) for _, n in _BOX_FACES]
        else:
            size = hi - lo
            v = np.asarray(center, float) + ((_BITS - 0.5) * size) @ np.asarray(R).T
            normals = [np.asarray(R) @ np.array(n, float) for _, n in _BOX_FACES]
        i0 = self._add_verts(v)
        for (idx, _), n in zip(_BOX_FACES, normals):
            self.faces.append((np.array(idx) + i0, n, shade(color, n), None, 0.0))

    def add_quad(self, corners, normal, tex, bias=1.0):
        """textured quad, corners TL TR BR BL (world)"""
        i0 = self._add_verts(np.asarray(corners, float))
        self.faces.append((np.arange(4) + i0, np.asarray(normal, float), (230, 230, 230), tex, bias))

    def finish(self):
        self.lo = self.verts.min(axis=0)
        self.hi = self.verts.max(axis=0)
        return self


def shade(color, n):
    k = 0.72 + 0.28 * max(0.0, float(np.dot(n, _LIGHT)))
    return tuple(int(min(255, c * k)) for c in color)


def world_objects(world):
    """static render objects from a World (one per box, markers drawn after their host box)"""
    objs = []
    for b in world.boxes:
        o = Obj(b.name)
        o.add_box(b.lo, b.hi, b.color)
        for mk in b.markers:
            n = np.array([math.cos(math.radians(mk.facing)), -math.sin(math.radians(mk.facing)), 0.0])
            o.add_quad(mk.corners(offset=0.3), n, marker_texture(mk.id, mk.size, mk.margin))
        objs.append(o.finish())
    return objs


# ---------------------------------------------------------------- renderer
def _clip_near(P, near):
    z = P[:, 2]
    if np.all(z >= near):
        return P
    if np.all(z < near):
        return None
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        ia, ib = a[2] >= near, b[2] >= near
        if ia:
            out.append(a)
        if ia != ib:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * t)
    return np.array(out) if len(out) >= 3 else None


class Renderer(object):
    def __init__(self, world, K, size, dist=None, near=2.0, grid=False):
        self.world = world
        self.K = np.asarray(K, float)
        self.W, self.H = size
        self.near = near
        self.grid = grid
        self.static = world_objects(world)
        self.ox = self.oy = 0
        self.cw, self.ch = self.W, self.H
        self.map1 = self.map2 = None
        if dist is not None and np.any(np.abs(dist) > 0):
            self._build_distortion(np.asarray(dist, float))
        self._floor_items()

    # lens distortion: output (distorted) pixel <- pinhole canvas pixel K * undistort(p)
    def _build_distortion(self, dist):
        xs, ys = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-4)
        if hasattr(cv2, "undistortPointsIter"):
            und = cv2.undistortPointsIter(pts, self.K, dist, None, self.K, crit)
        else:
            und = cv2.undistortPoints(pts, self.K, dist, None, self.K)
        und = und.reshape(self.H, self.W, 2)
        mx, my = und[..., 0], und[..., 1]
        self.ox = int(max(0, math.ceil(-mx.min()))) + 2
        self.oy = int(max(0, math.ceil(-my.min()))) + 2
        self.cw = int(math.ceil(mx.max())) + self.ox + 3
        self.ch = int(math.ceil(my.max())) + self.oy + 3
        self.map1, self.map2 = cv2.convertMaps((mx + self.ox).astype(np.float32),
                                               (my + self.oy).astype(np.float32), cv2.CV_16SC2)

    def _floor_items(self):
        w = self.world
        lo, hi = w.floor_lo, w.floor_hi
        self.floor_poly = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]], float)
        self.line_polys = []
        for ln in w.lines:
            a, b = ln["a"], ln["b"]
            d = b - a
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            u = d / L
            nrm = np.array([-u[1], u[0]]) * ln["width"] / 2.0
            segs = [(0.0, L)]
            if ln["dashed"]:
                segs = [(s, min(s + 15.0, L)) for s in np.arange(0.0, L, 30.0)]
            for s0, s1 in segs:
                p0, p1 = a + u * s0, a + u * s1
                q = np.array([p0 - nrm, p1 - nrm, p1 + nrm, p0 + nrm])
                self.line_polys.append((np.hstack([q, np.full((4, 1), 0.2)]), ln["color"]))

    # -- projection helpers
    def project(self, Pw, pos, R):
        """world points (N, 3) -> (uv (N, 2) in OUTPUT pinhole pixels, z (N,)); no distortion"""
        Pc = (np.asarray(Pw, float) - pos) @ R.T
        z = Pc[:, 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.K[0, 0] * Pc[:, 0] / zs + self.K[0, 2]
        v = self.K[1, 1] * Pc[:, 1] / zs + self.K[1, 2]
        return np.stack([u, v], axis=1), z

    def _uv(self, Pc):
        u = self.K[0, 0] * Pc[:, 0] / Pc[:, 2] + self.K[0, 2] + self.ox
        v = self.K[1, 1] * Pc[:, 1] / Pc[:, 2] + self.K[1, 2] + self.oy
        return np.stack([u, v], axis=1)

    def _fill(self, img, Pc, color):
        P = _clip_near(Pc, self.near)
        if P is None:
            return
        uv = self._uv(P)
        uv = np.clip(uv, -2e6, 2e6)
        cv2.fillConvexPoly(img, np.round(uv * 16).astype(np.int32), color, cv2.LINE_AA, 4)

    def _texture(self, img, Pc, tex, noise_px, rng):
        if np.any(Pc[:, 2] < self.near):
            self._fill(img, Pc, (230, 230, 230))
            return
        dst = self._uv(Pc).astype(np.float32)
        if noise_px and rng is not None:
            dst = dst + rng.normal(0, noise_px, dst.shape).astype(np.float32)
        x0, y0 = np.floor(dst.min(axis=0)).astype(int) - 1
        x1, y1 = np.ceil(dst.max(axis=0)).astype(int) + 2
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        n = tex.shape[0]
        src = np.array([[-0.5, -0.5], [n - 0.5, -0.5], [n - 0.5, n - 0.5], [-0.5, n - 0.5]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst - np.array([x0, y0], np.float32))
        roi = img[y0:y1, x0:x1].copy()
        cv2.warpPerspective(tex, M, (x1 - x0, y1 - y0), dst=roi, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
        img[y0:y1, x0:x1] = roi

    def _order(self, objs, pos):
        n = len(objs)
        if n < 2:
            return list(range(n))
        lo = np.array([o.lo for o in objs])
        hi = np.array([o.hi for o in objs])
        c = pos
        dist = np.linalg.norm(np.maximum(np.maximum(lo - c, c - hi), 0.0), axis=1)
        e = 1e-6
        below = hi[:, None, :] <= lo[None, :, :] + e                    # i below j on axis
        cannot = np.any(below & (c[None, None, :] >= hi[:, None, :]), axis=2)
        cannot |= np.any(below.transpose(1, 0, 2) & (c[None, None, :] <= lo[:, None, :]), axis=2)
        sep = np.any(below | below.transpose(1, 0, 2), axis=2)
        can = ~cannot
        nearer = dist[:, None] < dist[None, :]
        can = np.where(sep, can, nearer)
        after = can & ~can.T                                             # i must be drawn after j
        np.fill_diagonal(after, False)
        indeg = after.sum(axis=1)
        done = np.zeros(n, bool)
        order = []
        for _ in range(n):
            cand = np.nonzero((indeg == 0) & ~done)[0]
            if len(cand) == 0:
                cand = np.nonzero(~done)[0]                              # cycle: farthest first
            k = cand[np.argmax(dist[cand])]
            done[k] = True
            order.append(k)
            indeg -= after[:, k]
        return order

    def render(self, pos, R, dyn=(), noise_px=0.0, rng=None, highlight=()):
        """BGR image (H, W, 3) of the scene seen from camera (pos, R)"""
        pos = np.asarray(pos, float)
        R = np.asarray(R, float)
        img = np.empty((self.ch, self.cw, 3), np.uint8)
        img[:] = self.world.sky_color
        # floor, grid, tape lines
        self._fill(img, (self.floor_poly - pos) @ R.T, self.world.floor_color)
        if self.grid:
            self._draw_grid(img, pos, R)
        for q, col in self.line_polys:
            self._fill(img, (q - pos) @ R.T, col)
        # objects, back to front
        objs = []
        for o in list(self.static) + list(dyn):
            Pc = (o.verts - pos) @ R.T
            z = Pc[:, 2]
            if np.all(z < self.near):
                continue
            if np.all(z > self.near):
                uv = self._uv(Pc)
                if (np.all(uv[:, 0] < 0) or np.all(uv[:, 0] > self.cw) or
                        np.all(uv[:, 1] < 0) or np.all(uv[:, 1] > self.ch)):
                    continue
            objs.append((o, Pc))
        order = self._order([o for o, _ in objs], pos)
        for k in order:
            o, Pc = objs[k]
            red = o.highlight or o.name in highlight
            faces = []
            for f in o.faces:
                idx, nrm, col, tex, bias = f
                if np.dot(nrm, pos - o.verts[idx[0]]) <= 1e-9:
                    continue
                faces.append(f)
            if o.multi:
                faces.sort(key=lambda f: -(Pc[f[0], 2].mean() - f[4]))
            for idx, nrm, col, tex, bias in faces:
                if tex is not None:
                    self._texture(img, Pc[idx], tex, noise_px, rng)
                else:
                    if red:
                        col = (col[0] // 3, col[1] // 3, min(255, col[2] // 2 + 140))
                    self._fill(img, Pc[idx], col)
        if self.map1 is not None:
            return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        return img

    def _draw_grid(self, img, pos, R, step=100.0):
        lo, hi = self.world.floor_lo, self.world.floor_hi
        segs = []
        for x in np.arange(math.ceil(lo[0] / step) * step, hi[0] + 1, step):
            segs.append(((x, lo[1]), (x, hi[1])))
        for y in np.arange(math.ceil(lo[1] / step) * step, hi[1] + 1, step):
            segs.append(((lo[0], y), (hi[0], y)))
        col = tuple(int(c * 0.8) for c in self.world.floor_color)
        for a, b in segs:
            self.line3d(img, (a[0], a[1], 0.1), (b[0], b[1], 0.1), pos, R, col, 1, canvas=True)

    def line3d(self, img, a, b, pos, R, color, thick=1, canvas=False):
        """3D segment with near clipping (no distortion: use on observer views)"""
        P = (np.array([a, b], float) - pos) @ R.T
        za, zb = P[0, 2], P[1, 2]
        if za < self.near and zb < self.near:
            return
        if za < self.near:
            P[0] = P[0] + (P[1] - P[0]) * (self.near - za) / (zb - za)
        elif zb < self.near:
            P[1] = P[1] + (P[0] - P[1]) * (self.near - zb) / (za - zb)
        uv = self._uv(P) if canvas else self._uv(P) - np.array([self.ox, self.oy])
        uv = np.clip(uv, -1e5, 1e5)
        p = np.round(uv * 16).astype(np.int64)
        cv2.line(img, (int(p[0, 0]), int(p[0, 1])), (int(p[1, 0]), int(p[1, 1])), color, thick,
                 cv2.LINE_AA, 4)


# ---------------------------------------------------------------- dynamic figures
def drone_model(pos, psi, pitch, roll, scale=1.0, color=(60, 60, 60), hit=False):
    """small box drone for the observer view (scaled up for visibility)"""
    f, r, u = body_axes(psi, pitch, roll)
    Rb = np.stack([f, r, u], axis=1)                    # columns: box x = forward, y = right, z = up
    c = np.asarray(pos, float) + u * 2.0 * scale
    o = Obj("drone", multi=True)
    s = scale
    body = (40, 40, 230) if hit else color
    o.add_box([-5 * s, -4 * s, -2 * s], [5 * s, 4 * s, 2 * s], body, Rb, c)
    o.add_box([-1.5 * s, -2 * s, -1 * s], [1.5 * s, 2 * s, 1 * s], (40, 40, 220), Rb, c + f * 5.5 * s)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cc = c + (f * sx * 6 + r * sy * 6 + u * 2) * s
            o.add_box([-3.5 * s, -3.5 * s, -0.4 * s], [3.5 * s, 3.5 * s, 0.4 * s], (150, 150, 150), Rb, cc)
    return o.finish()
