"""第二段跟隨: 人拿 marker 0 以固定速度繞半徑 1 m 的圓走, 看 marker 0 有沒有一直留在畫面裡
用法: python run_follow.py [人走路速度 cm/s, 預設 25]
"""
import math
import sys

import numpy as np

import sim_world as W

SPEED = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
W.NOISE = 1.0
np.random.seed(2)
R, C = 100.0, np.array([0.0, 200.0])


def m0(t):
    fs = W.STATE_T.get("S2_FOLLOW")
    k = 0 if fs is None else max(0.0, t - fs - 1.0)
    a = -math.pi / 2 + k * SPEED / R
    return (C[0] + R * math.cos(a), C[1] + R * math.sin(a), "face", 80.0)


W.SCENE[0] = m0
d = W.run(lambda i: ord('h') if i == 3 else -1, max_iter=520, start=(0.0, 50.0))
fs = W.STATE_T["S2_FOLLOW"]
half_fov = math.degrees(math.atan(W.W_IMG / 2.0 / W.K[0, 0]))
seen, offs = 0, []
for (tt, P, psi, c, H) in d.trace:
    if tt < fs + 1:
        continue
    mx, mz = m0(tt)[:2]
    v = np.array([mx, mz]) - P
    p = math.radians(psi)
    f = np.array([math.sin(p), math.cos(p)])
    r = np.array([math.cos(p), -math.sin(p)])
    b = abs(math.degrees(math.atan2(v @ r, v @ f)))
    offs.append(b)
    seen += b < half_fov
print("\n人走 %.0f cm/s: marker 0 在畫面裡 %.0f%% 的時間, 偏離畫面中心最大 %.0f 度 (中位數 %.0f)" %
      (SPEED, 100.0 * seen / len(offs), max(offs), float(np.median(offs))))
