#!/usr/bin/env bash
# 在新電腦上設定 drone 作業環境 (macOS / Linux)
# 用法: bash setup.sh
#   - 沒有 conda 會自動裝 miniconda 到 ~/miniconda3
#   - 建立 / 更新 conda 環境 uav (Python 3.9 + djitellopy + opencv-contrib)
#   - 修正 run / env 腳本裡寫死的 miniconda 路徑
set -euo pipefail

ENV_NAME=uav
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---------- 1. conda ----------
find_conda() {
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        return 0
    fi
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/homebrew/Caskroom/miniconda/base"; do
        if [ -x "$d/bin/conda" ]; then CONDA_BASE="$d"; return 0; fi
    done
    return 1
}

if ! find_conda; then
    echo ">> 找不到 conda, 安裝 miniconda 到 ~/miniconda3"
    OS="$(uname -s)"; ARCH="$(uname -m)"
    case "$OS-$ARCH" in
        Darwin-arm64)  F=Miniconda3-latest-MacOSX-arm64.sh ;;
        Darwin-x86_64) F=Miniconda3-latest-MacOSX-x86_64.sh ;;
        Linux-x86_64)  F=Miniconda3-latest-Linux-x86_64.sh ;;
        Linux-aarch64) F=Miniconda3-latest-Linux-aarch64.sh ;;
        *) echo "不支援的系統: $OS-$ARCH"; exit 1 ;;
    esac
    TMP="$(mktemp -d)"
    curl -fsSL "https://repo.anaconda.com/miniconda/$F" -o "$TMP/miniconda.sh"
    bash "$TMP/miniconda.sh" -b -p "$HOME/miniconda3"
    rm -rf "$TMP"
    CONDA_BASE="$HOME/miniconda3"
    "$CONDA_BASE/bin/conda" init "$(basename "${SHELL:-zsh}")" || true
    # 新版 conda 要先同意 channel 條款, 用 conda-forge 就不需要 defaults
    "$CONDA_BASE/bin/conda" config --set auto_activate_base false || true
fi
echo ">> conda: $CONDA_BASE"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------- 2. 環境 ----------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">> 更新環境 $ENV_NAME"
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo ">> 建立環境 $ENV_NAME"
    conda env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"

# opencv-python 跟 opencv-contrib-python 同時存在會互相蓋掉, 只留 contrib (要有 cv2.aruco)
pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1 || true
pip install --force-reinstall --no-deps "opencv-contrib-python==4.10.0.84" >/dev/null

# ---------- 3. 修正腳本裡的 conda 路徑 ----------
for f in run env; do
    [ -f "$f" ] || continue
    # 不管原本寫的是哪台電腦的路徑 (/Users/... 或 /home/...), 都換成這台的 conda
    sed -E -i.bak "s#/[^'\" ]*/(miniconda3|anaconda3|miniforge3)#$CONDA_BASE#g" "$f" && rm -f "$f.bak"
    chmod +x "$f"
done
chmod +x pack rm 2>/dev/null || true
mkdir -p log out/lab1 out/lab2 out/lab3

# ---------- 4. 檢查 ----------
python - <<'PY'
import sys, cv2, numpy, djitellopy
print("python      ", sys.version.split()[0])
print("opencv      ", cv2.__version__, "| aruco:", hasattr(cv2, "aruco"))
print("numpy       ", numpy.__version__)
print("djitellopy  ", getattr(djitellopy, "__version__", "ok"))
PY

# ---------- 5. 模擬器 / 錄影檢查 (失敗不會中斷 setup, 只印 FAILED) ----------
# (a) 模擬器自我測試 (不需要螢幕, 約 1 秒)
if python lab6/sim/selftest.py --quick >/tmp/uav_selftest.log 2>&1; then
    echo "sim selftest  OK"
else
    echo "sim selftest  FAILED (詳細輸出: /tmp/uav_selftest.log)"
fi
# (b) 錄影: 用 mp4v 寫一個小 mp4 再讀回來 (flight log 的 fpv_raw.mp4 / fpv_hud.mp4 要用)
if python - <<'PY' 2>/dev/null
import os, sys, tempfile
import cv2, numpy as np
path = os.path.join(tempfile.mkdtemp(), "check.mp4")
w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (160, 120))
ok = w.isOpened()
for i in range(10):
    w.write(np.full((120, 160, 3), i * 20, np.uint8))
w.release()
cap = cv2.VideoCapture(path)
n = 0
while ok:
    r, _ = cap.read()
    if not r:
        break
    n += 1
cap.release()
os.remove(path)
sys.exit(0 if ok and n >= 8 else 1)
PY
then
    echo "mp4 recording OK"
else
    echo "mp4 recording FAILED (cv2.VideoWriter mp4v 不能用: 飛行 log 照寫, 只是沒有錄影)"
fi

echo
echo "完成. 之後用:  conda activate $ENV_NAME"
