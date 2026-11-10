# 期中專案重構 + 模擬賽道 計畫

狀態: **只有計畫, 還沒動任何程式** (2026-11-10)

## 0. 現況 (讀完程式後的盤點)

| 項目 | 現況 |
|---|---|
| `midterm.py` | 1534 行單一檔: 參數約 230 行 (~200 個全域常數) / 工具函式 / `Course` 狀態機 (`step()` 約 560 行) / `main()` 215 行 |
| log | `print` 與 `log()` 混用 (約 34 處 + 零散 print), 只能靠 `> output.log 2>&1`; 沒有每個控制週期的數值紀錄 |
| 錄影 | 沒有 |
| 畫面 | 影像上疊 3 行字 (狀態 / rc / 電量 / 溫度 / 計時); 狀態機的提示訊息 (`note`) 只印在終端機 |
| 模擬 | `sim/sim_world.py`: monkey-patch `midterm.Tello / time / cv2.waitKey`; 2D 位置 + 高度, rc 直接當 cm/s, 沒慣性 / 延遲 / 實體 / 遮擋 / 視窗; 第 1+2 段與第 3~5 段是兩個分開的場景 |
| 環境 | 這台是 WSL2, `uav` 環境已建好 (Python 3.9, OpenCV 4.10 Qt, 有 matplotlib / pillow / av; 沒有任何 3D 套件); WSLg 有 DISPLAY |

## 1. 目標檔案結構

```
lab6/
├── midterm.py                 入口 (很薄): python midterm.py [--sim] [--world ...] [--no-record]
├── config/
│   ├── params.json            所有飛行參數 (分組, 保留現在的值與中文說明)
│   ├── sim_physics.json       模擬物理參數
│   └── worlds/eab06.json      賽道場景 (marker / 障礙物 / 起點 / 手持 marker 0 的路徑)
├── course/                    (package 名稱可再議)
│   ├── params.py              讀 json -> P.S3_DIST 這種屬性; 檢查拼錯 / 缺少的 key; 速度倍率換算秒數
│   ├── vision.py              load_calib / make_detector / detect_markers
│   ├── control.py             limit / p_ctrl / Ema / scale_rc / wrap180
│   ├── telemetry.py           state_field / read_yaw / read_alt / read_tof / read_battery / read_temp
│   ├── states.py              狀態名稱常數 / PAUSES / WAIT_STATES / START_KEYS
│   ├── machine.py             Course: enter / begin / abort / track / align / seek / 計時
│   ├── segments.py            五段的 step 邏輯 (從 560 行的 step() 依段拆成 step_s1 ~ step_s5)
│   ├── flightlog.py           log 系統 (第 2 節)
│   ├── recorder.py            錄影 (第 3 節)
│   ├── hud.py                 操作介面 (第 4 節)
│   └── app.py                 主迴圈: 鍵盤 / keepalive / 監控 / 控制 / 顯示; 無人機與時鐘用參數傳入
├── sim/
│   ├── tello_sim.py           SimTello (djitellopy 介面 + 狀態封包) + 物理
│   ├── world.py               讀場景 json: marker / box / 手持 marker 路徑 / 碰撞 / TOF 往下量到什麼
│   ├── render.py              軟體 3D 繪圖器 (FPV 與觀察視角共用)
│   ├── viewer.py              模擬視窗 (FPV + 3D)
│   ├── run_sim.py             互動模擬入口
│   └── scenarios/             原本的 run_seg12 / seg345 / follow / seg5 (無視窗快速跑, 當回歸測試)
├── logs/                      每次執行一個資料夾 (加進 .gitignore)
├── keyboard_djitellopy.py     不動
└── calibrate.xml, pyimagesearch/, lab6.py   不動
```

重點:
- `app.py` 的主迴圈收 `drone` 和 `clock` 當參數 → 模擬器不用再 monkey-patch, 實機 / 模擬跑的是同一份迴圈
- 參數存取從全域 `S3_DIST` 改成 `P.S3_DIST` (機械式取代, 數值與邏輯不變)
- `midterm_v1/v2_backup.py` 移到 `backup/`, 並把重構前的 `midterm.py` 存成 `backup/midterm_v3_single.py`

## 2. log 系統

每次執行建立 `logs/20261110_203015_real/` (模擬是 `_sim`), 裡面:

| 檔案 | 內容 |
|---|---|
| `flight.log` | 給人看的文字: 現在所有 print / log / note 的訊息 + djitellopy 的指令與回應 (它用 `logging`, 直接掛 FileHandler, 不用再 `2>&1`); 每行有時間、等級 (INFO / WARN / ERROR)、目前狀態 |
| `telemetry.csv` | **每個控制週期一列**: 時間、狀態、送出的 rc(4)、看到哪些 marker、目標 marker 的原始量測與平滑後 (x, y, z, ang)、yaw / tof / h / 電量 / 溫度、迴圈耗時、影像 fps、按鍵、錄影的 frame 編號 |
| `events.jsonl` | 事件: 狀態切換 (含原因)、按鍵、切回手動、警告 (過熱 / 自己降落 / 主迴圈卡住)、計時分段 |
| `params.json` | 這次實際用的參數快照 + git commit → 之後看 log 知道是哪組參數飛的 |
| `fpv_raw.mp4` / `fpv_hud.mp4` | 見第 3 節 |

- 寫檔即時 flush (當掉不掉最後幾行); 終端機照樣印一份
- 模擬時另外多記「真實位置」(x, y, z, yaw, 速度, 碰撞), 可以直接比對「程式以為的」和「實際的」
- 附一支 `tools/plot_log.py`: 讀 telemetry.csv 畫狀態時間軸 + rc + 量測曲線 (matplotlib 已有)

## 3. 第一人稱錄影

- `fpv_raw.mp4`: 無人機原始畫面 (沒有疊字) → 之後可以離線重跑偵測 / 重現問題
- `fpv_hud.mp4`: 操作時看到的整個介面 (含狀態面板) → 回看時知道當下程式在想什麼
- 用背景執行緒 + queue 寫檔, 主迴圈只丟 frame, 不會被硬碟卡住; queue 滿就丟 frame 並記一筆警告
- 每張 frame 的編號寫進 telemetry.csv, 影片和數據對得起來
- 不改影像解析度 (照 CLAUDE.md), 用 OpenCV `VideoWriter` (mp4v), 不加新套件

## 4. 操作介面 (HUD)

操作時看不到終端機 → 所有該知道的東西都放進視窗。畫布 = 影像 960x720 + 右側面板 + 下方訊息列:

- **大字狀態**: 目前狀態、第幾段 (1~5 進度條)、自動 / 手動、FLYING / ON GROUND
- **目標**: 現在在追哪個 marker、有沒有看到、量測值 vs 目標值 (例如 距離 163 → 150)、到位計數
- **輸出**: rc 四軸的長條 (左右 / 前後 / 上下 / 旋轉)
- **機況**: 電量 / 溫度 / tof / h / yaw; 影像 fps 與「畫面停住 x.x 秒」; 超過門檻變紅
- **計時**: 總時間 + 各段時間
- **訊息列**: 最近幾行 log (原本只印在終端機的提示 / 警告), 警告用紅色並停留幾秒
- **按鍵提示**: 依狀態顯示可用的鍵
- 中文字用 pillow 畫 (OpenCV `putText` 不能畫中文); 找不到中文字型時自動退回英文
- 行為上的一個改變: 現在沒有新畫面時整個迴圈 `continue`, 視窗不更新; 新版影像停住時面板照樣更新 (控制邏輯仍然只在有新畫面時跑)

## 5. 模擬飛行環境

### 5.1 場景 (`config/worlds/eab06.json`)
- marker: id / 位置 (x, y, z) / 朝向 / 邊長
- 實體: 牆、板子、桌子 (桌面 + 桌腳)、椅子, 都用長方體描述 → 有碰撞、有遮擋、TOF 會量到桌面
- 起點位置 / 方向; 地上的線 (只畫出來)
- 手持 marker 0: 路徑點 + 走路速度 (劇本), 互動模式下也可以用方向鍵手動移動
- **整條賽道一個場景** (現在第 1+2 段與第 3~5 段是分開的, 尺寸也是估的)

### 5.2 物理 (`config/sim_physics.json`, 跑之前可改)
- 推力向量模型: 機身傾斜產生水平加速度, 重力一直作用; 內層速度控制把 rc 轉成目標速度再去追 → 有慣性、煞車會滑、起飛 / 降落 / 撞到後掉落都有重力
- 可調: rc→速度換算 (cm/s per rc)、最大傾角、傾斜反應時間、空氣阻力、垂直最大加速度、yaw 反應時間、指令延遲、**影像延遲與掉幀**、機頭漂移、陣風 / 位置漂移、感測器雜訊 (tof / yaw)、角點雜訊、鏡頭下傾角、地面效應、電量消耗、15 秒沒指令自動降落
- 機身傾斜會帶動鏡頭 (往前飛時畫面往下看) — 實機有、現在的模擬沒有
- 碰撞: 撞到實體 → 記錄事件、畫面標紅, 可設定「停住」或「墜落」
- 固定時間步長 (例如 100 Hz), 和畫面更新分開

### 5.3 繪圖與視窗
- 自己寫一個小的軟體繪圖器 (numpy + OpenCV, 深度排序 / 簡易 z-buffer): 給「相機位置 + 內參」就畫出長方體和 marker 貼圖
- **同一個繪圖器用兩次**: 無人機相機 (用 `calibrate.xml`, 含鏡頭畸變) → FPV, 交給原本的偵測流程; 觀察相機 → 3D 視角
- 視窗: 左邊 FPV + HUD (和實機同一個介面), 右邊 3D 賽道: 無人機 (含機頭方向 / 傾斜)、飛過的軌跡、障礙物、marker、相機視錐、碰撞標示; 滑鼠拖曳旋轉 / 縮放, 快捷鍵切俯視 / 側視 / 跟隨
- 模擬控制: 暫停 / 單步 / 倍速 / 重置
- 兩種模式: **互動** (即時, 鍵盤操作和實機一樣, 可以練操作) 與 **無視窗快速跑** (原本的劇本, 印重點數字, 也會產生 log)

### 5.4 為什麼不用現成 3D 引擎
pybullet / pyvista / panda3d 畫面比較好看, 但要加套件、WSLg 與 Mac 的 OpenGL 各有問題、而且和 OpenCV 視窗的事件迴圈要另外整合。自己畫長方體 + marker 的量不大, 兩台電腦行為一致, 所以建議自己畫。

## 6. 實作順序 (每一階段結束都是能跑的狀態)

| 階段 | 內容 | 驗證 |
|---|---|---|
| 0 | commit 目前狀態; 用現有模擬跑 4 個劇本, 存下輸出當基準 | — |
| 1 | 拆檔 + 參數改 json (純搬移, 不改行為) | 4 個劇本輸出和基準完全一樣 |
| 2 | log 系統 + 錄影 | 模擬跑一次, 檢查 log 資料夾內容 |
| 3 | HUD | 模擬視窗目視檢查 |
| 4 | 場景 + 物理 + 繪圖器 + 3D 視窗 | 物理參數設成「無慣性」時結果接近舊模擬; 再開慣性看各段 |
| 5 | 更新 CLAUDE.md / sim README, 整理備份檔 | — |

## 7. 風險與限制

- **重構後第一次實飛前, 行為只在模擬驗證過**。階段 1 用「輸出完全一樣」把關, 但 Wi-Fi、影像串流、真的 djitellopy 只有實飛能確認
- 物理參數一開始是估計值, 不代表真的 Tello。有了 telemetry.csv 之後可以用實飛 log 回頭校正 (例如前進 3 秒量距離、階躍反應)
- 模擬的 yaw 正負號定義仍和程式的假設相同 → `HEADING_SIGN` / `SIGN_ALIGN_YAW` 還是只能實飛確認
- JSON 不能寫註解, 現在參數旁的中文說明 (含 `<-- 實測`) 要有地方放 (見待決定事項)
- 錄兩支影片會多吃 CPU; 筆電跑不動時 `fpv_hud.mp4` 可關

## 8. 待決定事項

1. **繳交格式**: 期中要交單一 `.py` 嗎? 如果要, 拆檔後需要一支合併腳本產生單檔版
2. **賽道尺寸**: 需要賽道圖 (各 marker 位置 / 高度、桌子 / 板子 / 牆的尺寸) 才能建完整場景; 沒有的話先用現有劇本裡的估計值
3. **參數檔的註解**: 建議 `params.json` 允許 `//` 註解 (讀檔時去掉), 把現在的中文說明原樣搬過去
4. **實飛用哪台電腦**: 這台是 WSL2 (連 Tello 的 UDP 影像在 WSL2 上常有問題), 還是實飛都在 Mac?
5. **比賽日期**: 決定要不要把階段 2 / 3 (實飛時最有用) 排在階段 4 前面做完就先去實飛

## 9. Decisions (2026-11-10, authoritative: overrides anything above that conflicts)

1. Entry file is `main.py` (renamed from `midterm.py`). Package directory is `core/`. Many files are fine, no single-file build.
2. **All program output is English**: terminal, log files, HUD, sim window. No CJK rendering, no pillow text. Existing Chinese code comments / parameter notes stay as they are.
3. **Flight parameter values must not change.** They come from real flights. `config/params.json` carries exactly the current values (JSON with `//` comments, loader strips them; keep the existing Chinese notes next to each value). If the sim world disagrees with the parameters, change the world, never the parameters.
4. Sim must be light and fast (target Mac has 8 GB RAM): numpy + OpenCV only, no new packages, no GL. FPV stays 960x720 (calibration); the 3D view may be low-res.
5. Tello mass 80 g (battery + props included) -> `sim_physics.json`.
6. Sim has a dummy human who carries marker 0 (waits, then leads the drone around the chair to the blue line, then hides the marker).
7. Main loop signature: `core/app.py: run(drone, clock=time, io=None)`. `io.wait_key(ms) -> int` and `io.show(image)`; default io = cv2.waitKey / cv2.imshow. The sim supplies its own io (steps physics in wait_key, composes HUD + 3D view in show). No monkey-patching.
8. Phase 1 is verified with messages untouched (sim scenario output byte-identical to baseline); translation to English is a separate commit after that.

### Course layout (from the course slide; estimates, parameters win)

World frame, cm: X = east (right on the slide), Y = north (toward the wall), Z = up. Yaw 0 = facing +X... use any convention internally, but document it in the world file.

- Wall: along Y = 500, X from -100 to 800, markers 4 / 5 hang on it facing south.
- Segment 1 (drone flies east along Y ~ 0): dashed start line X = 0, solid line X = 50, take-off at (0, 0) facing east.
  Marker 1 at X = 200 facing west, on board 1 (behind it, ~60 deep, ~100 wide). Then a low box of unknown length, a 50 cm gap, a 100 cm long box, then marker 2 facing west on board 2, offset to the south (the drone's right). Slide says stop 115 cm before marker 2; the code uses `M2_STOP_DIST`.
  For marker heights and the marker 1 -> 2 offsets reuse the numbers in the old `sim/run_seg12.py` scene.
- Segment 2: human waits north-east of the drone's stop point with marker 0 at drone height, leads it around the chair at about (700, 330) one full circle, ends just east of the blue line (X = 650, Y 250..450) with the drone facing west, then hides marker 0.
- Segment 3: marker 3 at (500, 350) facing east, hanging on the east edge of table 1 (X 425..500, Y 250..500, tunnel under it; marker 3 centre Z ~ 60, table underside ~ 67 as in old `run_seg345.py`). 150 cm from blue line to marker 3. Drone flies west under the table, turns clockwise 90 deg to face north (the wall is then ~150 cm ahead).
- Marker 4 on the wall, 150 cm west of the table edge: about (275, 500). Marker 5 on the wall ~250 cm further west: about (25, 500). Same height, centre Z ~ 215 (real-flight log: tof 222..231 when aligned, `S3_UP_ALT` = 240).
- Segment 4: board 3 stands between them (X 130..190, Y 330..480), top lower than the markers (~150).
- Segment 5: landing table 50 cm wide, centre 150 cm from the wall below marker 5 (`M5_DIST` = 165): about X 0..60, Y 310..360, top Z ~ 75.
- Scoring: any manual takeover / collision / line miss = -5; must handle several markers in view; timed.
