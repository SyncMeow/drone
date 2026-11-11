"""console.py - 終端機輸出"""
from time import strftime     # log 時間戳一律用真實時間 (模擬用假時鐘也一樣)


def log(msg):
    """印到終端機並馬上寫進檔案 (python main.py > output.log 時 stdout 有緩衝, 當掉會掉最後幾行)"""
    print("[%s] %s" % (strftime("%H:%M:%S"), msg), flush=True)
