"""telemetry.py - 讀 Tello 狀態 (優先讀狀態封包: 不送指令、不會卡住主迴圈)"""


def state_field(drone, key):
    """從 Tello 一直回傳的狀態封包讀值 (不送指令、不會卡住); 讀不到回傳 None"""
    try:
        v = drone.get_current_state().get(key)
        return None if v is None else float(v)
    except Exception:
        return None


def read_yaw(drone):
    v = state_field(drone, "yaw")
    if v is not None:
        return v
    try:
        return float(drone.get_yaw())
    except Exception:
        return None


def read_alt(drone):
    """離地高度 (cm): 優先用機身下方的 TOF, 讀不到再用高度計; 都沒有回傳 None"""
    v = state_field(drone, "tof")
    if v is None:
        try:
            v = float(drone.get_distance_tof())
        except Exception:
            v = None
    if v is not None and 0 < v < 800:                  # TOF 超出範圍時會回傳很大的值
        return v
    v = state_field(drone, "h")
    if v is not None:
        return v
    try:
        return float(drone.get_height())
    except Exception:
        return None


def read_tof(drone):
    """第三段下降用的離地高度 (cm): 只用機身下方 TOF; 讀不到 / 超出範圍回傳 None。
    不用高度計 h: h 是相對起飛點的高度, 從桌上起飛或飛久了會偏 (實飛 log 出現過 h=-60, 害它直接開始鑽)"""
    v = state_field(drone, "tof")
    if v is None:
        try:
            v = float(drone.get_distance_tof())
        except Exception:
            v = None
    return v if v is not None and 0 < v < 800 else None


def read_battery(drone):
    """電量: 只從狀態封包讀 (舊版 djitellopy 的 get_battery() 會送指令等回應, 訊號差時會卡住主迴圈好幾秒)"""
    v = state_field(drone, "bat")
    return None if v is None else int(v)


def read_temp(drone):
    """機身溫度 (°C): 狀態封包的 (templ, temph); 讀不到回傳 None"""
    hi = state_field(drone, "temph")
    if hi is None:
        return None
    lo = state_field(drone, "templ")
    return (hi if lo is None else lo), hi
