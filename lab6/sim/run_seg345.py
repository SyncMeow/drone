"""第三 ~ 第五段: 在 marker 3 前按 j 開始, 一路做到降落
用法: python run_seg345.py [角點雜訊 px] [機頭漂移 deg/s]
場景: marker 3 掛在桌邊 (高 60 cm, 歪 8 度), 桌面下緣約 67 cm, 桌子深度 75 cm;
      鑽過去右轉 90 度後, 牆在右前方 150 cm, marker 4 / 5 高 120 cm, 兩者相距 250 cm
"""
import math
import sys

import numpy as np

import sim_world as W

W.NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
W.FakeTello.DRIFT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
np.random.seed(1)

M3_Z, M3_H, TABLE_DEPTH = 170.0, 60.0, 75.0
WALL_X = 190.0
W.SCENE[3] = lambda t: (0.0, M3_Z, "wall", M3_H, 8.0)
W.SCENE[4] = lambda t: (WALL_X, 300.0, "wall", 120.0, 90.0)
W.SCENE[5] = lambda t: (WALL_X, 550.0, "wall", 120.0, 90.0)

d = W.run(lambda i: ord('j') if i == 5 else -1, max_iter=1400, start=(15.0, 0.0))
print("\n狀態:", W.states_line())
prev = None
for (tt, P, psi, c, H) in d.trace:
    if prev is not None and prev[1] < M3_Z <= P[1]:
        bottom = M3_H - 7.5
        print("穿過 marker 3 時: 高度 %.0f cm, marker 3 下緣 %.0f cm -> 間隙 %.0f cm" % (H, bottom, bottom - H))
    prev = P
under = [H for (tt, P, psi, c, H) in d.trace if M3_Z <= P[1] <= M3_Z + TABLE_DEPTH]
if under:
    print("在桌子底下的最高高度: %.0f cm" % max(under))
a, b = W.STATE_T.get("S4_LEFT"), W.STATE_T.get("S5_ALIGN", 1e9)
hs = [H for (tt, P, psi, c, H) in d.trace if a is not None and a <= tt < b]
if hs:
    print("第四段向左時的高度: %.0f ~ %.0f cm (marker 4/5 在 120)" % (min(hs), max(hs)))
P = d.trace[-1][1]
print("降落點: 離牆 %.0f cm (目標 %.0f), 和 marker 5 左右差 %.0f cm" %
      (WALL_X - P[0], W.P.M5_DIST, P[1] - 550.0))
