# main.py 離線模擬器

不用接無人機，在電腦上跑 `../main.py` (`../core/`) 的完整流程。偵測、狀態機、控制都是 core/ 原本的程式碼，只有無人機、相機畫面、按鍵、時鐘是假的。

## 執行

```bash
cd ~/workspace/drone/lab6/sim
conda activate uav
python run_seg12.py            # 第一段 + 第二段 (按 g 開始, 最後按 j 進第三段)
python run_seg12.py 1.5 2      # 加上 1.5 px 角點雜訊 + 機頭每秒漂 2 度
python run_seg345.py           # 第三 ~ 第五段 (按 j 開始, 一路到降落)
python run_seg345.py 1.5 2
python run_follow.py 35        # 第二段跟隨: 人以 35 cm/s 繞圓走
python run_seg5.py             # 第五段 (按 l)
```

每支跑完會印出狀態順序 (`狀態@秒數`) 和重點數字，例如穿過 marker 3 時的高度間隙、降落點離牆多遠。

## 檔案

- `sim_world.py`: 假無人機 `FakeTello`、虛擬相機 (用 `../calibrate.xml` 把 `SCENE` 裡的 marker 投影成畫面)、假按鍵、假時鐘; 用 `core.app.run(drone, clock, io)` 直接傳入, 不 monkey-patch。
- `baseline/`: 六個劇本的輸出基準 (`bash baseline/run_all.sh /tmp/new && diff -r baseline /tmp/new`; 時間戳已遮掉)。
- `run_*.py`: 各段的劇本。改 `SCENE` 裡 marker 的位置 / 高度 / 角度，就能測不同場地。

場景座標 (俯視): x 向右、z 向前 (起點時機頭朝 +z)，高度單位 cm。
`SCENE[id] = lambda t: (x, z, mode, 高度, 角度)`，`mode` 是 `"wall"` (固定朝向，角度 0 = 面朝起點) 或 `"face"` (手持，永遠面向無人機)；回傳 `None` 代表看不到 (例如 marker 0 收起來)。

## 抓不到的問題

- rc 值直接當成 cm/s、deg/s，而且沒有慣性和延遲，所以秒數類參數 (`S3_UNDER_TIME`、`SIDE_TIME` …) 還是要實飛調。
- 模擬的 yaw 方向定義和程式的假設一樣，所以 `HEADING_SIGN`、`SIGN_ALIGN_YAW` 如果在真機上是反的，模擬看不出來。
- 沒有桌子、板子、牆等實體：不會撞，也不會擋住 marker。"不撞到 marker 3" 是從軌跡的高度算出來的。
- marker 之間不會互相遮擋 (後畫的蓋住先畫的)；畫面沒有模糊、反光、光線變化。
- 沒有電量、最低懸停高度、地面效應。
