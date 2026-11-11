"""control.py - 控制小工具: 限速 / 速度倍率 / P 控制 / EMA 平滑 / 手動限速包裝"""
import numpy as np

from core.params import P


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def limit(v, max_speed):
    return int(max(-max_speed, min(max_speed, v)))


def scale_rc(lr, fb, ud, yaw):
    """套用整體速度倍率"""
    return (limit(lr * P.SPEED_SCALE_MOVE, 100), limit(fb * P.SPEED_SCALE_MOVE, 100),
            limit(ud * P.SPEED_SCALE_UPDOWN, 100), limit(yaw * P.SPEED_SCALE_TURN, 100))


def p_ctrl(err, kp, dead, cap, min_speed=0):
    """P 控制: 死區內 0; 否則 kp*err, 可設最小速度, 再限速"""
    if abs(err) < dead:
        return 0
    out = kp * err
    if min_speed and abs(out) < min_speed:
        out = min_speed if out > 0 else -min_speed
    return limit(out, cap)


class Ema(object):
    """指數移動平均, 把 marker 量測平滑化 (去掉單張雜訊造成的抖動)"""

    def __init__(self, alpha=None):
        self.alpha = P.EMA_ALPHA if alpha is None else alpha
        self.v = None

    def reset(self):
        self.v = None

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self.v = x if self.v is None else self.alpha * x + (1 - self.alpha) * self.v
        return self.v


class LimitedDrone(object):
    """給 keyboard() 用: send_rc_control 依 MANUAL_MAX 限速, 其餘轉給 drone"""

    def __init__(self, drone, max_speeds):
        self._drone = drone
        self._max = max_speeds

    def send_rc_control(self, lr, fb, ud, yaw):
        vals = (lr, fb, ud, yaw)
        self._drone.send_rc_control(*scale_rc(*[limit(v, m) for v, m in zip(vals, self._max)]))

    def __getattr__(self, name):
        return getattr(self._drone, name)
