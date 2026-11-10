"""
params.py - 讀 config/params.json, 用 P.S3_DIST 這種屬性存取

* JSON 可以有 "//" 註解 (字串裡的 "//" 不算); 最上層是分組, 每組底下的 key 攤平成 P.<KEY>
* 少 key / 多 key (拼錯) / 重複 key 都直接報錯
* 非 JSON 的值在這裡轉換: ARUCO_DICT (名稱 -> cv2.aruco 常數), MANUAL_MAX (list -> tuple),
  START_KEYS ({"g": 狀態} -> {ord('g'): 狀態})
* apply_speed_scale(): 依速度倍率換算用時間決定距離的秒數 (保留基準值, 呼叫幾次結果都一樣)
"""
import json
import os

import cv2

from core import states

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "params.json")

# 所有參數名稱 (params.json 必須剛好是這些)
KEYS = tuple("""
    SPEED_SCALE_MOVE SPEED_SCALE_UPDOWN SPEED_SCALE_TURN SCALE_TIMES
    MARKER_SIZE ARUCO_DICT M0_ID M1_ID M2_ID M3_ID M4_ID M5_ID
    APPROACH_DIST SAFE_MIN_DIST APPROACH_MAX_TIME NEAR_LOST_EXTRA
    SIDE_SPEED SIDE_MODE SIDE_ALIGN_CM SIDE_MIN_TIME SIDE_TIME SIDE_MAX_TIME
    FWD_SPEED FWD_MAX_TIME M2_STOP_DIST M2_TOL
    LEFT_SPEED LEFT_MIN_TIME LEFT_MAX_TIME
    S2_START_HOLD FOLLOW_DIST FOLLOW_DEAD FOLLOW_FB_MAX FOLLOW_FB_KP FOLLOW_YAW_KP FOLLOW_YAW_DEAD
    FOLLOW_YAW_MAX FOLLOW_YAW_MIN FOLLOW_LAT_KP FOLLOW_LAT_MAX FOLLOW_LEAD FOLLOW_RATE_ALPHA
    FOLLOW_PRED_MAX YAW_RC_TO_DEG FOLLOW_EMA_ALPHA FOLLOW_SEARCH_DIR FOLLOW_SEARCH_LEFT_MAX
    FOLLOW_SEARCH_TIME FOLLOW_LOST_HOLD FOLLOW_TURN_FIRST_DEG FOLLOW_SEARCH_YAW
    S2_AUTO_S3 S3_BACK_TO_S2 S3_BACK_FRAMES S2_TO_S3_M3_GAP S2_TO_S3_TIME
    S3_DIST S3_DIST_TOL S3_YAW_EXTRA_TIME S3_YAW_TOL S3_YAW_DEAD S3_ANG_ALPHA S3_LAT_TOL
    S3_LAT_KI S3_LAT_I_MAX S3_LAT_MIN CAM_PITCH_DEG S3_PASS_ALT S3_CLEAR S3_DOWN_SPEED
    S3_LOST_EXTRA MIN_ALT S3_DOWN_MAX_TIME S3_UNDER_SPEED S3_UNDER_TIME S3_PAST_CM
    S3_UNDER_CM_PER_RC S3_TURN_DEG S3_UP_SPEED S3_UP_MAX_TIME S3_UP_ALT
    S3_FIND4_SPEED S3_FIND4_BACK S3_FIND4_TIME
    S4A_FB_MAX S4A_UD_KP S4A_UD_MAX S4A_LAT_TOL S4A_DIST_TOL S4A_UD_TOL M4_DIST
    SEEK_SPEED SEEK_EDGE_X_DEG SEEK_EDGE_Y_DEG SEEK_WAIT SEEK_TIME ALIGN_MAX_TIME
    S4_FIND_M4_TIME S4_CLIMB_CM S4_CLIMB_SPEED S4_CLIMB_MAX_TIME S4_LEFT_SPEED S4_CROSS_TIME
    S4_DESCEND_AFTER S4_MAX_TIME
    M5_DIST M5_TOL S5_LAT_DEAD S5_LAT_TOL S5_LAT_MIN S5_YAW_DEAD S5_YAW_MIN S5_YAW_TOL
    S5_ANG_ALPHA S5_LAT_KP S5_LAT_MAX S5_LAT_TRIM S5_SIMPLE S5_ARRIVE_FRAMES S5_SETTLE
    S5_LAT_AIM S5_STILL_TIME S5_STILL_CM S5_TABLE_CHECK S5_TABLE_DROP S5_SCAN_FIRST
    S5_SCAN_CM S5_TABLE_LOST S5_SCAN_SPEED S5_TABLE_IN_CM S5_MAX_TIME
    PAUSE_TIME ARRIVE_FRAMES
    EMA_ALPHA LAT_KP LAT_DEAD LAT_MAX UD_KP UD_DEAD UD_MAX FB_KP FB_DEAD FB_MAX FB_MIN
    ALT_KP ALT_DEAD ALT_MAX ALIGN_UD_DEAD ALIGN_UD_MIN
    ALIGN_YAW ALIGN_YAW_KP ALIGN_YAW_DEAD ALIGN_YAW_MAX ALIGN_YAW_MIN SIGN_ALIGN_YAW
    HOLD_HEADING HEADING_KP HEADING_MAX HEADING_MIN HEADING_DEAD HEADING_SIGN HEADING_ABORT
    TURN_MAX TURN_TOL TURN_MAX_TIME DEBUG_HEADING
    MANUAL_MAX KEY_HOLD START_KEYS
    FRAME_IS_RGB BATTERY_PERIOD MONITOR_PERIOD STATUS_LOG_PERIOD TEMP_WARN SELF_LAND_TIME
    KEEPALIVE LOOP_WARN CONTROL_PERIOD WINDOW
""".split())

# 依時間決定距離 / 角度的參數: 速度變慢時秒數要變長, 飛的距離才會一樣
TIMES_MOVE = ("SIDE_TIME", "SIDE_MIN_TIME", "SIDE_MAX_TIME", "FWD_MAX_TIME", "LEFT_MIN_TIME",
              "LEFT_MAX_TIME", "APPROACH_MAX_TIME", "ALIGN_MAX_TIME", "S3_UNDER_TIME",
              "S4_CROSS_TIME", "S4_MAX_TIME", "S5_MAX_TIME", "S3_FIND4_BACK", "S3_FIND4_TIME",
              "SEEK_TIME")
TIMES_UPDOWN = ("S3_DOWN_MAX_TIME", "S3_LOST_EXTRA", "S3_UP_MAX_TIME", "S4_CLIMB_MAX_TIME")
TIMES_TURN = ("TURN_MAX_TIME", "FOLLOW_SEARCH_TIME", "FOLLOW_SEARCH_LEFT_MAX")


def strip_comments(text):
    """去掉 // 到行尾的註解; 字串 ("...") 裡的 // 不動"""
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _no_dup(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("params: duplicate key %r" % k)
        d[k] = v
    return d


class Params(object):
    """參數物件: P.S3_DIST; 不存在的名稱會 AttributeError"""

    def __init__(self, values, path=""):
        self.__dict__["_path"] = path
        self.__dict__["_base_times"] = {}
        self.__dict__.update(values)

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise AttributeError("params: unknown parameter %r" % name)
        self.__dict__[name] = value

    def as_dict(self):
        return {k: self.__dict__[k] for k in KEYS}


def load(path=CONFIG):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        tree = json.loads(strip_comments(text), object_pairs_hook=_no_dup)
    except ValueError as e:
        raise ValueError("params: cannot parse %s: %s" % (path, e))
    flat = {}
    for group, items in tree.items():
        if not isinstance(items, dict):
            raise ValueError("params: top level %r must be a group {...}" % group)
        for k, v in items.items():
            if k in flat:
                raise ValueError("params: %r appears in more than one group" % k)
            flat[k] = v
    missing = [k for k in KEYS if k not in flat]
    unknown = [k for k in flat if k not in KEYS]
    if missing or unknown:
        raise ValueError("params: %s: missing %s, unknown %s" % (path, missing, unknown))

    # ---- 非 JSON 型別的轉換 ----
    name = flat["ARUCO_DICT"]
    if not hasattr(cv2.aruco, name):
        raise ValueError("params: ARUCO_DICT %r is not a cv2.aruco dictionary" % name)
    flat["ARUCO_DICT"] = getattr(cv2.aruco, name)
    flat["MANUAL_MAX"] = tuple(flat["MANUAL_MAX"])
    keys = {}
    for ch, st in flat["START_KEYS"].items():
        if len(ch) != 1 or st not in states.ALL:
            raise ValueError("params: bad START_KEYS entry %r: %r" % (ch, st))
        keys[ord(ch)] = st
    flat["START_KEYS"] = keys
    return Params(flat, path)


def apply_speed_scale(p=None):
    """用時間決定距離的參數依速度倍率換算 (以第一次看到的值為基準, 重複呼叫不會一直縮)"""
    p = P if p is None else p
    if not p.SCALE_TIMES:
        return
    base_times = p.__dict__["_base_times"]
    for names, scale in ((TIMES_MOVE, p.SPEED_SCALE_MOVE), (TIMES_UPDOWN, p.SPEED_SCALE_UPDOWN),
                         (TIMES_TURN, p.SPEED_SCALE_TURN)):
        for n in names:
            base = base_times.setdefault(n, getattr(p, n))
            setattr(p, n, base / max(scale, 0.05))


P = load()
