# CLAUDE.md

這是 DJI Tello 無人機課程的作業資料夾 (每週一個 lab + 期中賽道)。使用者是資工系學生，用中文溝通。

## 環境

- macOS, conda 環境 `uav` (Python 3.9): `conda activate uav`
- 新電腦: `bash setup.sh` (沒 conda 會裝 miniconda, 依 `environment.yml` 建 `uav`, 修正 run/env 的 conda 路徑)
- 套件: `djitellopy`, `opencv-contrib-python` (有 `cv2.aruco`), `numpy`
- OpenCV 新舊 aruco API 都要能跑: 程式裡用 `hasattr(cv2.aruco, "ArucoDetector")` 判斷,
  舊版用 `DetectorParameters_create()` / `drawMarker()`, 新版用 `ArucoDetector` / `generateImageMarker()`
- 跑實機前電腦要連 Tello 的 Wi-Fi; 實飛只有使用者能做, 不要自己執行會連無人機的程式

## 資料夾

```
drone/
├── lab1.py ~ lab4-2.py   早期 lab (影像處理, 在根目錄)
├── Tello_Video/          lab5 (相機校正 calibrate.py, ArUco 距離 lab5.py, keyboard_djitellopy.py)
├── lab6/
│   ├── lab6.py           lab6: 鍵盤手動 + ArUco PID 追蹤
│   ├── midterm.py        期中賽道 (主要在改的檔案)
│   ├── midterm_v*_backup.py  midterm 的舊版備份
│   ├── keyboard_djitellopy.py  課程給的鍵盤函式, 不要改 (要限速就在 midterm.py 裡包一層)
│   ├── calibrate.xml     相機內參 intrinsic / distortion (OpenCV FileStorage)
│   ├── pyimagesearch/    課程給的 PID / ObjCenter
│   └── sim/              midterm.py 的離線模擬器 (見下方)
├── pack / run / rm       作業打包 / 執行腳本 (只認根目錄的 labN.py)
└── 113550023_LabN.zip    繳交檔 (學號 113550023), 只放該 lab 的 .py, 不要有 .DS_Store / __MACOSX
```

## midterm.py 結構

- 檔案開頭的 docstring 寫了整個賽道流程與按鍵, 改流程時一起更新
- 參數全部在檔案上方, 中文註解; 標 `<-- 實測` 的是要現場量的值
- `Course` 類別是狀態機: `enter(state)` 切換狀態 (會重設平滑、計數器), `step()` 每個控制週期回傳 rc `(左右, 前後, 上下, 旋轉)`
- 五段賽道:
  1. `S1_*`: 接近 marker 1 → 右移到 marker 2 正前方 → 往前停在 marker 2 前 → 左移直到看到手持 marker 0
  2. `S2_FOLLOW`: 跟著手持 marker 0 (轉向 + 左右平移置中; 預測轉向 `FOLLOW_LEAD` 關著, 影像延遲會讓它轉反);
     找不到 marker 0 時一直往左轉找 (`FOLLOW_SEARCH_DIR`); 看過 marker 0 之後, marker 0 不在 + marker 3 在畫面
     持續 `S2_TO_S3_TIME` 秒就自動進第三段 (這段時間對準 marker 3, 不轉去找 marker 0); 按 j 也可以切;
     S3_ALIGN 時又看到 marker 0 會回到 S2_FOLLOW (`S3_BACK_TO_S2`)
  3. `S3_*`: 正對 marker 3 → 下降到 marker 3 下緣以下 (鏡頭傾角補償 + TOF 高度) → 鑽桌子 → 順時針轉 90° → 上升對正 marker 4
  4. `S4_*`: 維持 marker 4 的高度一路向左 (板子比 marker 4/5 低, 4/5 同高) 直到看到 marker 5
  5. `S5_ALIGN`: 對準 marker 5 (離牆 `M5_DIST`) 後降落在桌上
- 按鍵: `g h j k l` = 從手動進入第 1~5 段; `w s a d z x c v 1 2 3` = 手動操作, 自動中按到會立刻切回手動;
  `q`/ESC 結束; Ctrl+C 會送降落
- 量測: marker 位置用 solvePnP 的 cm 座標, 先 EMA 平滑再用小增益 P 控制 (`p_ctrl`: 死區 / 上限 / 最小速度);
  不要再用「像素對準畫面中心」+ 大增益, 實飛會左右快速來回甩
- 全域速度倍率 `SPEED_SCALE_MOVE / UPDOWN / TURN`: 套在所有 rc 上; `SCALE_TIMES=True` 時用時間決定距離的參數會自動換算秒數
- Tello 狀態 (電量 / 高度 / yaw / 溫度) 用 `state_field()` 從狀態封包讀, 不要在主迴圈呼叫會送指令等回應的 getter (會卡住迴圈)

## 使用者的習慣與規則

- **保留使用者調過的參數值**: 使用者常直接在編輯器改參數。改檔前先讀目前的檔案, 不要用舊版本覆蓋; 回報時說明保留了哪些值
- 改動要小而集中, 說明改了什麼、為什麼; 沒實飛驗證過的東西要講清楚
- 時間緊時不要跑不必要的測試; 必要時用 `sim/` 跑一兩個相關情境就好
- 使用者要求撤銷時, 完整還原到改動前的版本 (先確認使用者之後沒有再改過)
- 輸入法要切英文, OpenCV 視窗才收得到正確的按鍵碼 (注音輸入時會收到奇怪的碼, 例如 18)

## 模擬器 (lab6/sim)

```bash
cd lab6/sim
python run_seg12.py [角點雜訊px] [機頭漂移deg/s]   # 第一 + 二段
python run_seg345.py [雜訊] [漂移]                # 第三 ~ 五段
python run_follow.py [人走路速度cm/s]             # 第二段跟隨
python run_seg5.py                                # 第五段
```

- `sim_world.py` 用假 Tello 取代 djitellopy、用 `calibrate.xml` 把虛擬場地的 marker 投影成畫面、假按鍵、假時鐘;
  跑的是 midterm.py 原本的程式碼。`W.CAM_PITCH` 模擬鏡頭下傾
- 限制: rc 值直接當 cm/s 且沒有慣性 / 延遲; 沒有桌子 / 牆等實體; yaw 方向定義與程式假設相同 (抓不到符號反了)

## Tello 實機要注意

- 超過 15 秒沒收到指令會自己降落 → midterm.py 每 0.5 秒沒送指令就補送懸停 (`KEEPALIVE`)
- 過熱會自己降落, 冷卻前 `takeoff` 一直回 `error` → 畫面顯示溫度, 偵測到自己降落會印出溫度與電量
- 鏡頭往下傾 (`CAM_PITCH_DEG`, 預設 10°, 估計值): 同高度的東西在畫面偏上, 算高度 / 間隙時要補償
- 不要改影像解析度 (`set_video_resolution` 等): 相機校正是原解析度做的, 改了位置會算錯 (曾讓 marker 1 都飛不好)
- 影像串流在多台無人機的教室會很卡, 主要是 Wi-Fi 問題
- 錄 log 要連 stderr 一起: `python midterm.py > output.log 2>&1` (djitellopy 的指令 / 回應在 stderr)

## 尚未確認 (要實飛驗證)

- 自己降落是否真的是過熱 (看終端機的溫度訊息)
- `HEADING_SIGN`, `SIGN_ALIGN_YAW` 的正負號
- `CAM_PITCH_DEG`, `S3_PASS_ALT` (依 marker 3 下緣高度), `S3_UNDER_TIME`, `S5_LAT_TRIM` 的實際數值
