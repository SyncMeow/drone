"""第一段 + 第二段: 按 g 起飛開始; 第二段跟著走一段弧線的 marker 0; marker 0 收起來後按 j 進第三段
用法: python run_seg12.py [角點雜訊 px] [機頭漂移 deg/s]      例: python run_seg12.py 1.5 2
"""
import math
import sys

import numpy as np

import sim_world as W

W.NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
W.FakeTello.DRIFT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
np.random.seed(0)

M2 = (90.0, 320.0)                                    # marker 2 在起點右前方 (板子寬度未知, 這裡假設 90 cm)
W.SCENE[1] = lambda t: (0.0, 150.0, "wall")           # marker 1 在起點正前方 150 cm
W.SCENE[2] = lambda t: (M2[0], M2[1], "wall")


def m0(t):                                            # 手持 marker 0: 進入第二段 3 秒後沿弧線走, 11 秒後收起來
    fs = W.STATE_T.get("S2_FOLLOW")
    if fs is None:
        return (-60.0, 330.0, "face")
    k = t - fs
    if k > 11:
        return None
    a = min(max(k - 3, 0), 8) * 0.35
    return (-60.0 + 120 * math.sin(a), 330.0 + 120 * (1 - math.cos(a)), "face")


W.SCENE[0] = m0
W.SCENE[3] = lambda t: None


def script(i):
    if i == 5:
        return ord('g')
    fs = W.STATE_T.get("S2_FOLLOW")
    if fs is not None and W.clock.t > fs + 12 and "S3_ALIGN" not in W.STATE_T:
        return ord('j')                               # marker 0 收起來後按 j
    return -1


d = W.run(script, max_iter=700)
print("\nstates:", W.states_line())
r = W.pose_at(d, "S1_PAUSE2")
if r:
    print("end of right move: x = %.0f (marker 2 at x = %.0f), heading %.1f deg" % (r[0][0], M2[0], r[1]))
r = W.pose_at(d, "S1_PAUSE3")
if r:
    print("stopped in front of marker 2: %.0f cm (target %.0f)" % (M2[1] - r[0][1], 
          W.P.M2_STOP_DIST))
fs, fe = W.STATE_T.get("S2_FOLLOW"), W.STATE_T.get("S3_ALIGN", 1e9)
ds = []
for (tt, P, psi, c, H) in d.trace:
    spec = m0(tt)
    if fs is not None and spec and fs + 1 <= tt < fe:
        ds.append(np.hypot(spec[0] - P[0], spec[1] - P[1]))
if ds:
    print("segment 2 distance to marker 0: min %.0f, max %.0f cm" % (min(ds), max(ds)))
