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
│   ├── main.py           期中賽道的入口 (原 midterm.py): python main.py [--no-record] [--no-hud-video] [--no-hud] [--scale S]
│   ├── core/             期中賽道程式 (主要在改的地方)
│   │   ├── app.py        主迴圈 run(drone, clock, io): 鍵盤 / keepalive / 狀態監控 / 影像 / 自動控制 / 顯示
│   │   ├── machine.py, segments.py, states.py   狀態機 Course 與五段的 step_s1 ~ step_s5
│   │   ├── control.py, vision.py, telemetry.py, params.py   p_ctrl / rc 倍率, ArUco + solvePnP, 狀態封包, 讀參數
│   │   ├── blackbox.py, flightlog.py, recorder.py, hud.py   flight log / 錄影 / HUD 視窗
│   │   └── display.py    視窗縮放 (塞進小螢幕, 只縮顯示)
│   ├── config/
│   │   ├── params.json   飛行參數 (JSON + // 中文註解), 值都是實飛調出來的
│   │   ├── sim_physics.json   3D 模擬器的物理參數 (估計值)
│   │   └── worlds/eab06.json  3D 模擬器的賽道 (估計值)
│   ├── logs/             每次執行的 log (gitignore), 見下方
│   ├── tools/plot_log.py 把一個 log 資料夾畫成一張 png
│   ├── sim/              模擬器 (見下方)
│   ├── backup/           midterm 的舊版備份
│   ├── keyboard_djitellopy.py  課程給的鍵盤函式, 不要改 (要限速就在 core/ 裡包一層, 見 LimitedDrone)
│   ├── calibrate.xml     相機內參 intrinsic / distortion (OpenCV FileStorage)
│   └── pyimagesearch/    課程給的 PID / ObjCenter
├── pack / run / rm       作業打包 / 執行腳本 (只認根目錄的 labN.py)
└── 113550023_LabN.zip    繳交檔 (學號 113550023), 只放該 lab 的 .py, 不要有 .DS_Store / __MACOSX
```

## 期中賽道程式結構 (main.py + core/)

- `main.py` 開頭的 docstring 寫了整個賽道流程與按鍵, 改流程時一起更新
- 程式輸出 (終端機 / log / HUD / 模擬器視窗) 全部是英文; 程式碼裡的中文註解保留
- 參數值都在 `config/params.json` (中文註解); 標 `<-- 實測` 的是要現場量的值。
  **參數值是實飛調出來的, 沒有使用者同意不要改**; 模擬器跟參數對不上時改模擬器的場地, 不改參數
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

## log 資料夾

每次執行 (實機或模擬) 都會寫 `lab6/logs/YYYYmmdd_HHMMSS_real/` (模擬器是 `_sim`):
`flight.log` (終端機輸出 + djitellopy 指令 / 回應), `telemetry.csv` (每個控制週期一列), `events.jsonl`
(狀態切換、按鍵、警告、起飛降落、分段計時), `params.json` (實際用的參數 + git commit), `fpv_raw.mp4`, `fpv_hud.mp4`;
模擬器另外有 `truth.csv` (真實位置 / 姿態 / 碰撞 / 人的位置 / marker 0 是否在畫面) 和 events 裡的 `collision`。
畫圖: `python tools/plot_log.py logs/<dir>`

## 使用者的習慣與規則

- **保留使用者調過的參數值**: 使用者常直接在編輯器改參數。改檔前先讀目前的檔案, 不要用舊版本覆蓋; 回報時說明保留了哪些值
- 改動要小而集中, 說明改了什麼、為什麼; 沒實飛驗證過的東西要講清楚
- 時間緊時不要跑不必要的測試; 必要時用 `sim/` 跑一兩個相關情境就好
- 使用者要求撤銷時, 完整還原到改動前的版本 (先確認使用者之後沒有再改過)
- 輸入法要切英文, OpenCV 視窗才收得到正確的按鍵碼 (注音輸入時會收到奇怪的碼, 例如 18)

## 模擬器 (lab6/sim)

```bash
cd lab6
python sim/run_sim.py                         # 3D 模擬器 + 真正的飛行程式 (core.app.run) + 完整 HUD; 按 g 飛全程
python sim/run_sim.py --ideal / --no-record / --scale 0.6 / --manual-only
python sim/scenarios/full_course.py --seeds 1-5   # 不開視窗跑全程, 印 PASS/FAIL、分段時間、碰撞、降落位置
python sim/selftest.py --quick                # 模擬器自我測試 (不需螢幕, setup.sh 也會跑)
cd sim && bash baseline/run_all.sh /tmp/new && diff -r baseline /tmp/new   # 舊的運動學劇本: 飛行邏輯的回歸測試
```

- 3D 模擬器: `tello_sim.py` (假 Tello, 有 `get_current_state()`), `physics.py`, `render.py`, `world.py`,
  `human.py` (拿 marker 0 的假人), `viewer.py` (視窗 + SimIO); 只用 numpy + OpenCV, 要輕 (目標 Mac 8 GB RAM)
- 視窗 = 左邊 HUD 畫面 + 右邊 3D 賽道; 自動縮小塞進螢幕 (預設假設 1440x900), 錄影是原尺寸
- 舊的運動學劇本 (`sim_world.py`, `run_seg12.py`, `run_seg345.py`, `run_follow.py`, `run_seg5.py`): rc 直接當 cm/s, 沒有慣性;
  改 `core/` 之後輸出要跟 `baseline/` 一模一樣 (控制邏輯沒變的證明)
- 賽道 `config/worlds/eab06.json` 與物理 `config/sim_physics.json` 都是估計值; **物理參數沒有跟實機校正過**
- rc -> cm/s 的倍率未知 (目前假設 0.85): 實飛時用 `telemetry.csv` 裡 marker 距離 (raw_z) 在固定 rc 時的斜率量出來
- 限制: 沒有地面效應 / Wi-Fi 卡頓 / 動態模糊; yaw 方向定義與程式假設相同 (抓不到符號反了); 假人很理想 (會等無人機)

## Tello 實機要注意

- 超過 15 秒沒收到指令會自己降落 → main.py (core/app.py) 每 0.5 秒沒送指令就補送懸停 (`KEEPALIVE`)
- 過熱會自己降落, 冷卻前 `takeoff` 一直回 `error` → 畫面顯示溫度, 偵測到自己降落會印出溫度與電量
- 鏡頭往下傾 (`CAM_PITCH_DEG`, 預設 10°, 估計值): 同高度的東西在畫面偏上, 算高度 / 間隙時要補償
- 不要改影像解析度 (`set_video_resolution` 等): 相機校正是原解析度做的, 改了位置會算錯 (曾讓 marker 1 都飛不好)
- 影像串流在多台無人機的教室會很卡, 主要是 Wi-Fi 問題
- 錄 log 要連 stderr 一起: `python main.py > output.log 2>&1` (djitellopy 的指令 / 回應在 stderr)

## 尚未確認 (要實飛驗證)

- 自己降落是否真的是過熱 (看終端機的溫度訊息)
- `HEADING_SIGN`, `SIGN_ALIGN_YAW` 的正負號
- `CAM_PITCH_DEG`, `S3_PASS_ALT` (依 marker 3 下緣高度), `S3_UNDER_TIME`, `S5_LAT_TRIM` 的實際數值
