"""第五段: 面向牆 (marker 5 在前方 190 cm, 偏左 40 cm) 按 l, 對準後降落"""
import sim_world as W

W.SCENE[5] = lambda t: (190.0, 40.0, "wall", 80.0, 90.0)
d = W.run(lambda i: ord('l') if i == 5 else -1, max_iter=400, start=(0.0, 0.0), start_psi=90.0)
print("\n狀態:", W.states_line())
P = d.trace[-1][1]
print("降落點: 離牆 %.0f cm (目標 %.0f), 和 marker 5 左右差 %.0f cm" %
      (190.0 - P[0], W.P.M5_DIST, P[1] - 40.0))
