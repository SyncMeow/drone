"""第五段: 面向牆 (marker 5 在前方 190 cm, 偏左 40 cm) 按 l, 對準後降落"""
import sim_world as W

W.SCENE[5] = lambda t: (190.0, 40.0, "wall", 80.0, 90.0)
d = W.run(lambda i: ord('l') if i == 5 else -1, max_iter=400, start=(0.0, 0.0), start_psi=90.0)
print("\nstates:", W.states_line())
P = d.trace[-1][1]
print("landing point: %.0f cm from wall (target %.0f), %.0f cm sideways from marker 5" %
      (190.0 - P[0], W.P.M5_DIST, P[1] - 40.0))
