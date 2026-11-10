"""vision.py - 相機校正 / ArUco 偵測 + solvePnP / 畫面標示"""
import math
import os

import cv2
import numpy as np

from core.control import wrap180
from core.params import P

LAB6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_calib(path=None):
    path = path or os.path.join(LAB6, "calibrate.xml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("intrinsic").mat()
    dist = fs.getNode("distortion").mat()
    fs.release()
    if K is None or dist is None:
        raise RuntimeError("cannot read intrinsic/distortion from " + path)
    return K, dist


def make_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(P.ARUCO_DICT)
    if hasattr(aruco, "ArucoDetector"):                # OpenCV >= 4.7
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return lambda g: detector.detectMarkers(g)
    params = aruco.DetectorParameters_create()         # OpenCV < 4.7
    return lambda g: aruco.detectMarkers(g, dictionary, parameters=params)


def marker_obj_points(size=None):
    """marker 四個角在 marker 座標系的位置 (cm), 給 solvePnP 用"""
    half = (P.MARKER_SIZE if size is None else size) / 2
    return np.array([[-half,  half, 0],
                     [ half,  half, 0],
                     [ half, -half, 0],
                     [-half, -half, 0]], dtype=np.float32)


def draw_axes(img, K, dist, rvec, tvec, length):
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, length)
    else:
        cv2.aruco.drawAxis(img, K, dist, rvec, tvec, length)


def put(frame, text, org, color=(0, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def detect_markers(frame, detect, obj_pts, K, dist):
    """回傳 {id: dict(t=(x, y, z) cm, d=距離 cm, ang=角度 deg)}
    x 右正, y 下正, z 前正; ang = marker 法向量與 "正對無人機" 的夾角 (0 = 正對, 正 = marker 面朝畫面右側)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect(gray)
    found = {}
    if ids is None:
        return found
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    for c, mid in zip(corners, ids.flatten()):
        ok, rvec, tvec = cv2.solvePnP(obj_pts, c.reshape(4, 2).astype(np.float32), K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        t = tvec.flatten()
        d = float(np.linalg.norm(t))
        mid = int(mid)
        if mid in found and found[mid]["d"] <= d:
            continue                                   # 同 id 出現兩次: 取近的
        R, _ = cv2.Rodrigues(rvec)
        zp = R @ np.array([0.0, 0.0, 1.0])
        ang = wrap180(math.degrees(math.atan2(zp[2], zp[0])) + 90.0)
        pm = -R.T @ tvec.flatten()                    # 無人機在 marker 座標系的位置: x = 沿牆的左右, z = 離牆距離
        found[mid] = dict(t=t, d=d, ang=ang, pm=pm)
        draw_axes(frame, K, dist, rvec, tvec, P.MARKER_SIZE)
        cx, cy = c[0].mean(axis=0).astype(int)
        put(frame, "id %d  %.0f cm  %.0f deg" % (mid, d, ang), (cx - 80, cy - 40))
        put(frame, "x %.0f y %.0f z %.0f" % (t[0], t[1], t[2]), (cx - 80, cy - 15))
    return found
