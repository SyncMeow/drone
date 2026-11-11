"""
midterm.py - 期中賽道 (EAB06): 第一段 ~ 第五段

第一段 (marker 1, 2)
    S1_APPROACH : 面向 marker 1 前進, 停在 APPROACH_DIST (不撞到 marker 1)
    S1_RIGHT    : 向右平移避開板子; 移到 marker 2 正前方 (marker 2 的 x ~ 0) 就停
    S1_FORWARD  : 往前, 停在 marker 2 前 M2_STOP_DIST (太近會往後退)
    S1_LEFT     : 向左平移, 直到看到手持的 marker 0
第二段 (marker 0)
    S2_FOLLOW   : 跟著手持的 marker 0 (轉向 + 左右平移讓它保持在畫面中心, 並保持距離), 繞椅子飛過藍線;
                  會估計 marker 0 移動的方向, 提前往它要去的方向轉;
                  看過 marker 0 之後, marker 0 不在畫面但 marker 3 在 (對準 marker 3) 持續 S2_TO_S3_TIME 秒 -> 自動切到第三段;
                  也可以隨時按 j 切到第三段 (S3_ALIGN); S3_ALIGN 時 marker 0 又出現會回到 S2_FOLLOW
第三段 (marker 3, 4)
    S3_ALIGN    : 正對 marker 3, 停在 S3_DIST
    S3_DOWN     : 下降, 直到 marker 3 的下緣比鏡頭高 S3_CLEAR 以上 (保證從 marker 3 下面過, 不會撞到)
    S3_UNDER    : 保持低高度往前鑽過桌子 (S3_UNDER_TIME 秒)
    S3_TURN     : 原地順時針轉 S3_TURN_DEG 度
    S3_UP       : 往上升直到看到 marker 4, 或 TOF 到 S3_UP_ALT (marker 4 的高度)
    S3_FIND4    : 到高度還沒看到 marker 4: 往後退一點, 再往左 / 往右平移找 (不靠鑽了多遠)
    S3_ALIGN4   : 對正 marker 4 (左右 / 高度 / 距離 / 角度)
第四段 (marker 5)
    S4_LEFT     : 以對正 marker 4 時的高度 (板子比 marker 低, marker 4 / 5 同高) 一直向左平移,
                  維持高度越過板子, 看到 marker 5 就停 (S4_CLIMB_CM > 0 時會先多爬升一段)
第五段
    S5_ALIGN    : 對準 marker 5, 停在離牆 M5_DIST (桌子正上方)
    => 降落在桌上

每個狀態只看自己的目標 marker id, 同時看到多個 marker 時其他的會被忽略。
左右 / 上下 / 前後都用 marker 的公分位置 (solvePnP), 先平滑 (EMA) 再小增益修正, 不會左右快速來回甩。

按鍵 (視窗 "drone" 要在最上層, 輸入法要切成英文)
    g h j k l  從手動直接進入第 1 / 2 / 3 / 4 / 5 段 (還沒起飛時會先自動起飛)
    計時: 按 g 開始, 畫面右上顯示; 降落時 (或程式結束時) 在終端機印出總時間與各段時間 / 佔比
        g: 第一段開頭 (接近 marker 1)     h: 第二段 (跟著 marker 0, 按 j 切第三段)
        j: 第三段開頭 (對正 marker 3)     k: 第四段開頭 (先對正 marker 4; 2 秒沒看到就以目前方向為牆的方向)
        l: 第五段 (對準 marker 5 後降落)
    1 起飛 / 2 降落 / 3 停止
    w s a d z x c v 手動移動 (放開約 KEY_HOLD 秒後自動停止); 5 高度 / 6 電量
    自動飛行中按任何操作鍵 (w s a d x z c v 1 2 3) => 立刻切回手動, 要再按 g~l 才會回到自動
    q / ESC 結束 (停止並降落); Ctrl+C 也會降落
"""
import math
import os
import time
from time import strftime     # log 時間戳 (模擬器會把 time 換成假時鐘, 所以另外 import)

import cv2
import numpy as np
from djitellopy import Tello

from keyboard_djitellopy import keyboard

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 參數   (標 <-- 實測 的是圖上沒有寫、要現場量的)
# ============================================================================
# ---- 整體速度倍率 (所有自動飛行 + 鍵盤手動都會乘上) ----------------------------------
#  1.0 = 原本的速度; 0.5 = 一半; 1.5 = 1.5 倍 (rc 最多到 100)
SPEED_SCALE_MOVE = 1.3                # 平移: 前後 + 左右
SPEED_SCALE_UPDOWN = 1.3              # 上升 / 下降
SPEED_SCALE_TURN = 1.1                # 左右旋轉
# 用時間決定距離的動作 (例如鑽桌子往前 S3_UNDER_TIME 秒) 會自動依倍率調整秒數, 讓飛的距離不變
SCALE_TIMES = True

MARKER_SIZE = 15.0                    # marker 邊長 (cm)
ARUCO_DICT = cv2.aruco.DICT_4X4_250

M0_ID, M1_ID, M2_ID, M3_ID, M4_ID, M5_ID = 0, 1, 2, 3, 4, 5

# ---- 第一段 ------------------------------------------------------------------
APPROACH_DIST = 70.0                  # 停在 marker 1 前的距離 (cm); 必須 < 150
SAFE_MIN_DIST = 60.0                  # 離 marker 小於此值時絕不再往前 (cm)
APPROACH_MAX_TIME = 6.0               # 接近 marker 1 最多花這麼久 (s), 超過就直接進入右移
NEAR_LOST_EXTRA = 25.0                # 已靠近 (<= APPROACH_DIST + 此值) 時 marker 1 看不到, 視為抵達

SIDE_SPEED = 30                       # 向右平移速度
SIDE_MODE = "marker2"                 # "marker2": 移到 marker 2 正前方就停; "time": 固定移 SIDE_TIME 秒
SIDE_ALIGN_CM = 15.0                   # marker 2 的 x <= 此值 (cm) 就算已在它正前方
SIDE_MIN_TIME = 1.0                   # 至少右移這麼久 (s), 確保離開板子
SIDE_TIME = 2.0                       # "time" 模式的右移時間; 看不到 marker 2 時的備案
SIDE_MAX_TIME = 1.5                   # 右移最長時間 (s)

FWD_SPEED = 20                        # 還沒看到 marker 2 時往前飛的速度
FWD_MAX_TIME = 9.0                    # 往前飛最多幾秒還沒看到 marker 2 就停 (s)
M2_STOP_DIST = 150.0                  # 停在 marker 2 前的距離 (cm); 圖上寫約 115
M2_TOL = 5.0                          # M2_STOP_DIST ± 此值內算到位 (cm); 比這近會往後退

LEFT_SPEED = 30                       # 向左平移速度
LEFT_MIN_TIME = 0.0                   # 至少左移這麼久才開始找 marker 0 (s)
LEFT_MAX_TIME = 4.0                   # 左移這麼久還沒看到 marker 0 就停下來等 (s)

# ---- 第二段 ------------------------------------------------------------------
S2_START_HOLD = 1.0                   # 第一段左移看到 marker 0 進第二段時, 先原地懸停幾秒再開始跟 (s)
FOLLOW_DIST = 45.0                   # 跟 marker 0 保持的距離 (cm)
FOLLOW_DEAD = 10.0                    # 距離死區 (cm)
FOLLOW_FB_MAX = 45                    # 跟隨時前後速度上限 (原 35)
FOLLOW_FB_KP = 0.8                    # 跟隨時前後增益 (其他段用 FB_KP = 0.5)
FOLLOW_YAW_KP = 1.3                   # 轉向: (預測的) marker 0 偏離正前方角度 (deg) * 此值 (原 1.1)
FOLLOW_YAW_DEAD = 5.0
FOLLOW_YAW_MAX = 70                   # 轉向速度上限 (原 50)
FOLLOW_YAW_MIN = 8
FOLLOW_LAT_KP, FOLLOW_LAT_MAX = 0.6, 40       # 左右平移讓 marker 0 保持在畫面中心 (原上限 30)
# 預測轉動方向: 估計 marker 0 (人) 繞著無人機移動的角速度, 往 "FOLLOW_LEAD 秒後它會在的方向" 轉
FOLLOW_LEAD = 0.0                     # 預測多久之後 (s); 0 = 不預測 (原 0.5)
#   實飛: 影像有延遲時, 自己轉動造成的角度變化被算成「人在移動」, 預測方向會反過來 → 往反方向亂轉
#   (模擬加 0.3 s 影像延遲: 0.5 時 27% 時間轉錯方向、常跟丟; 0 時 4%), 所以先關掉
FOLLOW_RATE_ALPHA = 0.4               # 角速度估計的平滑程度
FOLLOW_PRED_MAX = 30.0                # 預測最多往前多轉幾度 (避免雜訊讓它亂轉)
YAW_RC_TO_DEG = 1.0                   # rc 轉向值 1 約等於每秒幾度 (用來扣掉自己轉動造成的角度變化)
FOLLOW_EMA_ALPHA = 0.6                # 跟隨時的平滑程度: 比其他段大 = 反應比較快 (人在走, 不能太延遲)
FOLLOW_SEARCH_DIR = "left"            # 找不到 marker 0 時: "left" = 一直往左轉找直到找到; "last" = 往最後出現的那邊轉 FOLLOW_SEARCH_TIME 秒
FOLLOW_SEARCH_LEFT_MAX = 20.0         # "left" 模式最多轉幾秒還找不到就懸停 (轉速 50 約 7 秒一圈 -> 約轉 3 圈)
FOLLOW_SEARCH_TIME = 1.0              # marker 0 跑出畫面時, 往它最後出現的那一側轉幾秒找回來 (s)
# 跑出畫面時往哪邊轉: 看 marker 0 最後一次出現在畫面的偏左還偏右 (用當下偵測到的原始位置, 不用平滑過的;
#   平滑值會落後, 人快速橫越時可能還停在另一邊)。FOLLOW_LEAD 的速度預測仍然關著 (影像延遲會讓它轉反)
FOLLOW_LOST_HOLD = 0.0                # marker 0 消失這麼久 (s) 以內當作漏抓: 照剛剛的轉向繼續轉, 不全速找; 0 = 關
#   (模擬加延遲時 0.4 反而比較常跟丟, 先關; 實飛如果漏抓一張就猛轉、轉到模糊, 可以試 0.3)
FOLLOW_TURN_FIRST_DEG = 40.0          # marker 0 偏離中心越多, 往前越慢 (先轉正再前進); 偏 40 度只剩 30% 前進; 0 = 不限制
FOLLOW_SEARCH_YAW = 50                # 找的時候的轉向速度 (原 35)
# 第二段自動接第三段: 看過 marker 0 之後, marker 0 不在畫面 + marker 3 在畫面 持續 S2_TO_S3_TIME 秒 -> 自動進 S3_ALIGN
#   (這段時間不找 marker 0, 對準 marker 3); 隨時按 j 也可以切
S2_AUTO_S3 = True                     # False = 只能按 j
S3_BACK_TO_S2 = True                  # 在 S3_ALIGN 時又看到 marker 0 (連續 S3_BACK_FRAMES 張) -> 回到第二段繼續跟
S3_BACK_FRAMES = 3
S2_TO_S3_M3_GAP = 0.5                 # 對準 marker 3 的這幾秒中, marker 3 漏抓不超過這麼久 (s) 不算中斷
S2_TO_S3_TIME = 3.0                   # 看過 marker 0 之後, "marker 0 不在 + marker 3 在畫面" 持續這麼久 (s) 就進第三段;
                                      #   這段時間不轉去找 marker 0, 而是對準 marker 3

# ---- 第三段: 鑽桌子 -------------------------------------------------------------
S3_DIST = 150.0                       # 在 marker 3 前多遠對正 + 下降 (cm)。太近的話下降時 marker 會跑出畫面上緣
S3_DIST_TOL = 10.0
S3_YAW_EXTRA_TIME = 5.0               # 對正 marker 3 超時後, 左右 / 角度還不正時最多再修幾秒; 還不行就懸停 (不會斜斜往前鑽)
# 開始下降前一定要 "正對" marker 3: 左右置中 + 角度正 (距離不用很準, 鑽的距離會依實際離 marker 3 多遠補)
S3_YAW_TOL = 3.0                      # 開始下降前角度 (平均過的) 要小於這個 (原本共用 ±6)
S3_YAW_DEAD = 2.0                     # 角度 (平均過的) 超過這個就轉
S3_ANG_ALPHA = 0.3                    # 角度平均: 越小越平均
S3_LAT_TOL = 8.0                      # 左右: marker 3 偏離畫面中心 (cm) 要小於這個
S3_LAT_KI, S3_LAT_I_MAX = 0.4, 12     # 左右積分: 偏一邊越久推越大力 (實飛 log: 一直往右飄, rc -11~-15 推不回來, 越來越偏)
S3_LAT_MIN = 8                        # 超時後只修左右 / 角度時的最小左右速度 (不然偏一點點時推不動)
CAM_PITCH_DEG = 10.0                  # Tello 鏡頭往下傾的角度: 同高度的東西在畫面上會偏上, 不補償會以為自己比較低 (撞到 marker 3 的原因)
S3_PASS_ALT = 40.0                    # 鑽桌子前, 機身下方 TOF 量到的高度也要 <= 這個值 (cm); 依 marker 3 下緣高度調, None = 不檢查
S3_CLEAR = 20.0                       # marker 3 下緣至少要比鏡頭高這麼多 (cm) 才往前鑽 (不撞到 marker 3)
S3_DOWN_SPEED = 25                    # 下降速度
S3_LOST_EXTRA = 0.6                   # 下降途中 marker 3 從畫面上緣消失時, 再多降幾秒保險 (s)
MIN_ALT = 30.0                        # 不再往下的最低高度 (cm, 機身下方 TOF 量測)
S3_DOWN_MAX_TIME = 8.0
S3_UNDER_SPEED = 44                   # 鑽桌子往前的速度
S3_UNDER_TIME = 6                     # 備案: 不知道離 marker 3 多遠時, 往前鑽幾秒
# 鑽桌子的距離以 marker 3 為基準: 開始鑽時量到離 marker 3 的距離 z, 往前飛 (z + S3_PAST_CM) 這麼遠就停
#   所以不管 S3_ALIGN / S3_DOWN 停在哪, 最後都停在 "marker 3 正下方再往前 S3_PAST_CM" 的地方
S3_PAST_CM = 150.0                     # 飛過 marker 3 正下方之後再往前多遠 (cm) = 桌子深度 + 餘量   <-- 要停的位置改這個
S3_UNDER_CM_PER_RC = 0.85             # 速度換算: rc 1 約等於每秒幾 cm  <-- 實測 (飛 3 秒量距離 / (3 * S3_UNDER_SPEED * SPEED_SCALE_MOVE))
#   0.85 是用原本的設定反推的 (S3_UNDER_SPEED=44, S3_UNDER_TIME=6 剛好飛 150+75 cm), 離 marker 3 正好 S3_DIST 時行為跟原本一樣
S3_TURN_DEG = 90.0                    # 鑽過去後順時針轉幾度 (面向牆上的 marker 4)
S3_UP_SPEED = 50                      # 往上找 marker 4 的速度
S3_UP_MAX_TIME = 5.0                  # 往上最多幾秒 (保險; 正常是看到 marker 4 或到 S3_UP_ALT 就停) (原 4: 升到 240 不夠)
S3_UP_ALT = 240.0                     # 往上升到 TOF 這麼高 (cm) 還沒看到 marker 4 就停下來左右找  <-- 實測; None = 只看時間
#   實飛 log: 對正 marker 4 時 tof 約 222~231 cm 且 marker 4 還在畫面偏上, 所以 marker 4 大約在 tof 240;
#   升的途中看到 marker 4 就會先停, 這個值只是 "看不到時升到哪" (設太低會在 marker 4 下面左右找, 找不到)
# 升到 marker 4 高度還沒看到 marker 4 (例如鑽得太短, marker 4 在左邊畫面外): 自己找, 不靠鑽了多遠
S3_FIND4_SPEED = 25                   # 找 marker 4 時平移 / 後退的速度
S3_FIND4_BACK = 1.0                   # 先往後退幾秒 (離牆遠一點, 畫面看得比較寬)
S3_FIND4_TIME = 4.0                   # 先往左找幾秒 (鑽太短 -> marker 4 在左邊), 再往右找 2 倍時間; 都找不到才停下來等
# 對正 marker 4 (S3_ALIGN4) 的速度與容許誤差 (其他段: 前後上限 25, 上下 0.3 / 20, 容許 左右 8 / 前後 8 / 高度 7)
S4A_FB_MAX = 40                       # 前後速度上限 (剛找到 m4 時常離牆很近, 要退很多)
S4A_UD_KP, S4A_UD_MAX = 0.5, 30       # 上下增益 / 上限
S4A_LAT_TOL = 10.0                    # 對正的容許誤差: 左右 (cm)
S4A_DIST_TOL = 15.0                   #                 前後 (cm)
S4A_UD_TOL = 12.0                     #                 上下 (cm)
M4_DIST = 150.0                       # 對正 marker 4 時離牆的距離 (cm); 圖上 marker 4 與桌子相距 150
SEEK_SPEED = 15                       # 對正時 marker 剛好跑出畫面邊緣: 往它最後出現的方向慢慢找回來的速度
SEEK_EDGE_X_DEG = 18.0                # marker 消失前偏離正前方超過這個角度 (左右) 才往那邊找 (畫面左右半寬約 27 度)
SEEK_EDGE_Y_DEG = 13.0                # 上下 (畫面上下半高約 21 度)
SEEK_WAIT = 0.6                       # 在畫面中間消失時先原地等這麼久 (s), 還沒出現就往最後的方向找
SEEK_TIME = 2.7                       # 最多找幾秒, 找不到就懸停 (s)
ALIGN_MAX_TIME = 10.0                 # 對正一個 marker 最多花幾秒, 超過就以目前位置繼續

# ---- 第四段: 越過板子 -------------------------------------------------------------
# 板子比 marker 4 / 5 低, 且 marker 4 和 5 一樣高: 對正 marker 4 後的高度 (= marker 高度) 已經高過板子,
# 所以不用爬升, 維持這個高度一路向左就能越過板子, 而且 marker 5 會剛好出現在畫面中間高度。
S4_FIND_M4_TIME = 2.0                 # 按 k 時先對正 marker 4; 這麼久 (s) 都沒看到 marker 4 就以目前方向直接進第四段
S4_CLIMB_CM = 0.0                     # 額外爬升多少 (cm); 0 = 不爬升. 若實測發現離板子太近, 可設 10~20 當餘量
S4_CLIMB_SPEED = 30
S4_CLIMB_MAX_TIME = 4.0
S4_LEFT_SPEED = 30                    # 向左平移速度
S4_CROSS_TIME = 1.8                     # 向左幾秒後算已經越過板子   <-- 實測
S4_DESCEND_AFTER = False              # 越過板子後是否降回爬升前的高度 (S4_CLIMB_CM > 0 時才有意義)
S4_MAX_TIME = 3.0                     # 向左最多幾秒還沒看到 marker 5 就停下來等

# ---- 第五段: 降落在桌上 -------------------------------------------------------------
M5_DIST = 165.0                       # 離牆 (marker 5) 多遠降落 (cm) = 桌子中心
M5_TOL = 8.0
S5_LAT_DEAD, S5_LAT_TOL, S5_LAT_MIN = 3.0, 10.0, 8  # 第五段左右: 死區 3 cm, 容許 10 cm (原 5, 桌子 50 cm 寬), 最小速度 8 (其他段是 6 / 8 / 0)
# 第五段轉向: 離 marker 5 約 165 cm 時 marker 角度量測很抖 (±5 度以上), 死區 5 + 最小轉速 8 會一直左右小轉;
# 每轉一點, marker 在畫面的左右位置就跳 (165 cm * sin 3° ≈ 9 cm), 左右平移跟著來回追 → 降落前一直左右晃
S5_YAW_DEAD, S5_YAW_MIN = 2.0, 6      # 角度 (平均過的) 超過 2 度就轉, 最小轉速 6 (原 10 / 0: 容許歪 10 度, 實飛歪 7 度落在桌子左邊)
S5_YAW_TOL = 3.0                      # 降落前角度 (平均過的) 要小於這個
S5_ANG_ALPHA = 0.15                   # 第五段角度平均: 越小越平均 (單張雜訊大), 但反應越慢
S5_LAT_KP, S5_LAT_MAX = 0.6, 25       # 第五段左右增益 / 速度上限 (其他段 0.4 / 15; 調大讓第五段快一點)
S5_LAT_TRIM = 0.0                     # 降落點左右微調 (cm, 面向 marker 5): 正 = 往右, 負 = 往左; 例如老是落在桌子右邊就設 -10
S5_SIMPLE = True                      # True = 簡單版: 連續 S5_ARRIVE_FRAMES 張 marker 5 在中心 + 距離對就降落
                                      #   (不煞車 / 不等停穩 / 不用 TOF 找桌子); False = 下面那套
S5_ARRIVE_FRAMES = 6                  # 簡單版: 連續幾張 (建議 5~8; 越大越穩但越慢)
S5_SETTLE = 0.8                       # 剛看到 marker 5 先煞車懸停幾秒 (第四段向左的速度會帶進來)
S5_LAT_AIM = 4.0                      # 左右偏差小於這個 (cm) 就不再推, 讓它停下來 (容許誤差 S5_LAT_TOL 內才降落)
S5_STILL_TIME, S5_STILL_CM = 1.0, 5.0 # 降落前要在範圍內 "停穩" 這麼久 (s): 這段時間左右晃動不超過 5 cm (前後 10 cm)
# 對準 marker 5 之後用 TOF 確認下面真的是桌子 (實飛: 對準 m5 / 離牆 167 / 角度 0 也停穩了, 但 tof 只比 h 小 13 cm,
#   下面根本不是桌子 -> m5 跟桌子沒有正好對齊, 只靠 m5 再怎麼準都會落在桌外)
S5_TABLE_CHECK = True                 # False = 照舊, 對準 m5 就降落
S5_TABLE_DROP = 30.0                  # TOF 比地板小這麼多 (cm) 就算在桌子上方 (桌子比這個高)
S5_SCAN_FIRST = 1                     # 下面不是桌子時先往哪邊找: 1 = 右 (之前都落在桌子左邊), -1 = 左
S5_SCAN_CM = 60.0                     # 往一邊最多找多遠 (cm), 找不到再往另一邊
S5_TABLE_LOST = 3.0                   # 找桌子時 marker 5 連續看不到這麼久 (s) 就直接降落
S5_SCAN_SPEED = 15                    # 找桌子的平移速度 (慢, TOF 才來得及反應)
S5_TABLE_IN_CM = 20.0                 # 找到桌子邊緣後再往裡面移多少 (cm) 才降落 (桌子寬約 50 -> 一半少一點)
S5_MAX_TIME = 12.0                    # 對準超過這麼久: 只要大致在桌子上方 (誤差 < 2 倍容許) 就降落

PAUSE_TIME = 0.8                      # 每個動作之間的懸停穩定時間 (s)
ARRIVE_FRAMES = 3                     # 連續幾張 frame 都到位才算抵達

# ---- 平滑的位置修正 ----------------------------------------------------------------
EMA_ALPHA = 0.35                      # 量測值平滑: 越小越平滑但反應越慢
LAT_KP, LAT_DEAD, LAT_MAX = 0.4, 6.0, 15      # 左右: x (cm) -> 速度
UD_KP, UD_DEAD, UD_MAX = 0.3, 15.0, 20        # 上下: y (cm) -> 速度
FB_KP, FB_DEAD, FB_MAX = 0.5, 8.0, 25         # 前後: 距離誤差 (cm) -> 速度
FB_MIN = 10                           # 前後誤差超出死區時的最小速度
ALT_KP, ALT_DEAD, ALT_MAX = 0.6, 4.0, 20      # 高度 (TOF/高度計, cm) 保持
ALIGN_UD_DEAD, ALIGN_UD_MIN = 5.0, 8          # 對正 marker 4 時高度要對得比較準 (第四段就用這個高度越過板子)
# 依 marker 角度轉成正對 (第三段對正 marker 3, 第三 / 五段對正 marker 4, 5)
ALIGN_YAW = True
ALIGN_YAW_KP, ALIGN_YAW_DEAD, ALIGN_YAW_MAX, ALIGN_YAW_MIN = 0.6, 5.0, 20, 8
SIGN_ALIGN_YAW = -1                   # 同 lab6 的 SIGN_YAW: marker 面朝畫面右側 -> 逆時針轉; 方向反了就乘 -1

# ---- 航向 (直線段不要歪 / 轉 90 度) --------------------------------------------------
HOLD_HEADING = True
HEADING_KP = 1.0
HEADING_MAX = 20
HEADING_MIN = 8
HEADING_DEAD = 3.0
HEADING_SIGN = 1                      # Tello 的 yaw 順時針為正; 若發現越修越歪, 乘 -1
HEADING_ABORT = 45.0                  # 直線段偏離超過此角度就停止修正
TURN_MAX = 40                         # 轉 90 度時的旋轉速度上限
TURN_TOL = 4.0                        # 轉到目標 ± 此角度內算完成
TURN_MAX_TIME = 8.0
DEBUG_HEADING = False

# ---- 手動 -----------------------------------------------------------------------
MANUAL_MAX = (30, 30, 50, 80)         # (左右, 前後, 上下, 旋轉)
KEY_HOLD = 0.2
# 從手動進入第 X 段的按鍵
START_KEYS = {ord('g'): "S1_APPROACH", ord('h'): "S2_FOLLOW", ord('j'): "S3_ALIGN",
              ord('k'): "S3_ALIGN4", ord('l'): "S5_ALIGN"}   # k: 先對正 marker 4 再進第四段

FRAME_IS_RGB = True                   # djitellopy 2.5 的 frame_read.frame 是 RGB (畫面像負片 / 紅藍對調就是這個); 舊版是 BGR -> 設 False
BATTERY_PERIOD = 3.0
# 狀態監控 (溫度 / 電量 / 高度): 只讀 Tello 自己一直送來的狀態封包, 不送指令, 不會卡住迴圈
MONITOR_PERIOD = 0.5                  # 多久讀一次 (s)
STATUS_LOG_PERIOD = 10.0              # 多久在終端機 (log) 印一行狀態 (s)
TEMP_WARN = 85                        # 機身溫度 (temph, °C) 超過就印警告  <-- 估計值, 看 log 再調
SELF_LAND_TIME = 1.5                  # 程式以為在飛, 但高度 h 連續這麼久 (s) 都是 0 → 印出「可能自己降落」
# Tello 超過 15 秒沒收到任何指令會自己降落。手動模式下沒按鍵時程式原本完全不送指令,
# 所以每隔 KEEPALIVE 秒沒送過指令就補送一次懸停 (0, 0, 0, 0)
KEEPALIVE = 0.5
LOOP_WARN = 1.0                       # 主迴圈卡住超過這麼久 (s) 就在終端機印出來 (方便查 "不理會輸入")
CONTROL_PERIOD = 0.05
WINDOW = "drone"

# 狀態
MANUAL = "MANUAL"
S1_APPROACH, S1_PAUSE1, S1_RIGHT, S1_PAUSE2 = "S1_APPROACH", "S1_PAUSE1", "S1_RIGHT", "S1_PAUSE2"
S1_FORWARD, S1_PAUSE3, S1_LEFT, S1_WAIT0 = "S1_FORWARD", "S1_PAUSE3", "S1_LEFT", "S1_WAIT0"
S2_FOLLOW = "S2_FOLLOW"
S3_ALIGN, S3_DOWN, S3_UNDER, S3_PAUSE1 = "S3_ALIGN", "S3_DOWN", "S3_UNDER", "S3_PAUSE1"
S3_TURN, S3_UP, S3_WAIT4, S3_ALIGN4 = "S3_TURN", "S3_UP", "S3_WAIT4", "S3_ALIGN4"
S3_FIND4 = "S3_FIND4"
S4_CLIMB, S4_LEFT, S4_WAIT5 = "S4_CLIMB", "S4_LEFT", "S4_WAIT5"
S5_ALIGN, FINISHED = "S5_ALIGN", "FINISHED"
S5_TABLE = "S5_TABLE"
SPLIT_STATES = (S2_FOLLOW, S3_ALIGN, S4_CLIMB, S5_ALIGN)  # 分段計時的分界 (第 3 段結束 = 進第四段 S4_CLIMB)
PAUSES = {S1_PAUSE1: S1_RIGHT, S1_PAUSE2: S1_FORWARD, S1_PAUSE3: S1_LEFT, S3_PAUSE1: S3_TURN}
WAIT_STATES = (S1_WAIT0, S3_WAIT4, S4_WAIT5, FINISHED)     # 停下來等的狀態: 這時也可以按 g~l 換段

NON_MOVE_KEYS = (ord('1'), ord('2'), ord('3'), ord('5'), ord('6'))
OPERATION_KEYS = tuple(ord(c) for c in "wsadxzcv123")


# ============================================================================
# 相機 / ArUco
# ============================================================================
def load_calib(path=None):
    path = path or os.path.join(HERE, "calibrate.xml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("intrinsic").mat()
    dist = fs.getNode("distortion").mat()
    fs.release()
    if K is None or dist is None:
        raise RuntimeError("cannot read intrinsic/distortion from " + path)
    return K, dist


def make_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    if hasattr(aruco, "ArucoDetector"):                # OpenCV >= 4.7
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return lambda g: detector.detectMarkers(g)
    params = aruco.DetectorParameters_create()         # OpenCV < 4.7
    return lambda g: aruco.detectMarkers(g, dictionary, parameters=params)


def draw_axes(img, K, dist, rvec, tvec, length):
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, length)
    else:
        cv2.aruco.drawAxis(img, K, dist, rvec, tvec, length)


def put(frame, text, org, color=(0, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


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
        draw_axes(frame, K, dist, rvec, tvec, MARKER_SIZE)
        cx, cy = c[0].mean(axis=0).astype(int)
        put(frame, "id %d  %.0f cm  %.0f deg" % (mid, d, ang), (cx - 80, cy - 40))
        put(frame, "x %.0f y %.0f z %.0f" % (t[0], t[1], t[2]), (cx - 80, cy - 15))
    return found


# ============================================================================
# 控制小工具
# ============================================================================
def limit(v, max_speed):
    return int(max(-max_speed, min(max_speed, v)))


def scale_rc(lr, fb, ud, yaw):
    """套用整體速度倍率"""
    return (limit(lr * SPEED_SCALE_MOVE, 100), limit(fb * SPEED_SCALE_MOVE, 100),
            limit(ud * SPEED_SCALE_UPDOWN, 100), limit(yaw * SPEED_SCALE_TURN, 100))


# 依時間決定距離 / 角度的參數: 速度變慢時秒數要變長, 飛的距離才會一樣
_TIMES_MOVE = ("SIDE_TIME", "SIDE_MIN_TIME", "SIDE_MAX_TIME", "FWD_MAX_TIME", "LEFT_MIN_TIME",
               "LEFT_MAX_TIME", "APPROACH_MAX_TIME", "ALIGN_MAX_TIME", "S3_UNDER_TIME",
               "S4_CROSS_TIME", "S4_MAX_TIME", "S5_MAX_TIME", "S3_FIND4_BACK", "S3_FIND4_TIME")
_TIMES_UPDOWN = ("S3_DOWN_MAX_TIME", "S3_LOST_EXTRA", "S3_UP_MAX_TIME", "S4_CLIMB_MAX_TIME")
_TIMES_TURN = ("TURN_MAX_TIME", "FOLLOW_SEARCH_TIME", "FOLLOW_SEARCH_LEFT_MAX")
_TIMES_MOVE = _TIMES_MOVE + ("SEEK_TIME",)
_BASE_TIMES = {}


def apply_speed_scale():
    if not SCALE_TIMES:
        return
    g = globals()
    for names, scale in ((_TIMES_MOVE, SPEED_SCALE_MOVE), (_TIMES_UPDOWN, SPEED_SCALE_UPDOWN),
                         (_TIMES_TURN, SPEED_SCALE_TURN)):
        for n in names:
            base = _BASE_TIMES.setdefault(n, g[n])
            g[n] = base / max(scale, 0.05)


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

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
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


def log(msg):
    """印到終端機並馬上寫進檔案 (python midterm.py > output.log 時 stdout 有緩衝, 當掉會掉最後幾行)"""
    print("[%s] %s" % (strftime("%H:%M:%S"), msg), flush=True)


# ============================================================================
# 賽道狀態機
# ============================================================================
class Course(object):
    def __init__(self, drone):
        self.drone = drone
        self.state = MANUAL
        self.t0 = 0.0
        self.ema = Ema()
        self.ema5 = Ema(S5_ANG_ALPHA)      # 第五段: marker 5 角度的平均
        self.ema_id = None
        self.ema_seen = 0.0
        self.arrive = 0
        self.heading_ref = None
        self.heading_ok = False
        self.turn_target = None
        self.last = None                   # 最後一次看到目標 marker 的平滑量測 (x, y, z, ang)
        self.last_bearing = None           # 第二段: marker 0 最後出現的方向 (deg)
        self.prev_bearing = None           # 第二段: 上一次的方向 / 時間, 用來估角速度
        self.prev_bt = 0.0
        self.bearing_rate = 0.0            # marker 0 繞著無人機移動的角速度 (deg/s, 右/順時針為正)
        self.last_yaw_cmd = 0
        self.last_seen = 0.0
        self.lost_since = None
        self.alt0 = None                   # 進入狀態時的高度
        self.hold_alt = None               # 鑽桌子時要維持的高度
        self.alt_left = None               # 第四段向左時要維持的高度
        self.notes = set()
        self.want_land = False
        self.last_dbg = 0.0

    # ---------------- 狀態切換 ----------------
    def enter(self, state, now):
        if state != self.state:
            print("[%.1f] %s -> %s" % (now, self.state, state))
        prev = self.state
        self.state = state
        self.t0 = now
        self.arrive = 0
        self.ema.reset()
        self.ema5.reset()
        self.ang5 = None
        self.ang3 = None
        self.ema3a = Ema(S3_ANG_ALPHA)                     # S3_ALIGN: marker 3 角度的平均
        self.s5_hist = []                                  # 第五段: 到位後的位置紀錄 (判斷有沒有停穩)
        self.s5_n = 0                                      # 第五段簡單版: 連續到位的張數
        self.ema.alpha = FOLLOW_EMA_ALPHA if state == S2_FOLLOW else EMA_ALPHA
        self.ema_id = None
        self.last = None
        self.last_bearing = None
        self.prev_bearing = None
        self.bearing_rate = 0.0
        self.last_yaw_cmd = 0
        self.m0_ever = False                               # 第二段: 進來之後有沒有看過 marker 0
        self.m3_since = self.m3_last = None                # 第二段: marker 0 不在 + marker 3 在 的開始時間
        self.ema3 = getattr(self, "ema3", None) or Ema()
        self.m0_back = 0                                   # S3_ALIGN: 連續看到 marker 0 的 frame 數
        self.lat_i, self.lat_t = 0.0, None                 # S3_ALIGN 左右積分 (抵抗一直飄)
        self.last_seen = now
        self.lost_since = None
        self.notes = set()
        self.alt0 = read_alt(self.drone)
        if state == S3_UNDER:
            self.hold_alt = self.alt0
            self.under_time = S3_UNDER_TIME
            v = S3_UNDER_SPEED * SPEED_SCALE_MOVE * S3_UNDER_CM_PER_RC
            z0 = getattr(self, "s3_z0", None)
            if z0 is not None and v > 0:
                self.under_time = (z0 + S3_PAST_CM) / v
                print("  離 marker 3 %.0f cm, 往前飛 %.0f cm (%.1f 秒)" % (z0, z0 + S3_PAST_CM, self.under_time))
            else:
                print("  不知道離 marker 3 多遠, 依時間往前 %.1f 秒" % self.under_time)
        self.s3_z0 = None
        if state == S3_TURN:
            yaw = read_yaw(self.drone)
            base = self.heading_ref if self.heading_ok else yaw
            self.turn_target = None if base is None else wrap180(base + S3_TURN_DEG)
        if state in (S3_ALIGN, S3_ALIGN4, S5_ALIGN) and prev != state:
            self.heading_ok = False                    # 對正時用 marker 角度轉向, 不做航向保持
        if state in (S4_LEFT, S5_ALIGN) and prev not in (S4_LEFT, S4_WAIT5):
            self.tof_floor = None                          # 重新記錄地板高度
        if getattr(self, "run_t0", None) is not None and not getattr(self, "timer_done", False):
            for key in SPLIT_STATES:
                if state == key and (key == S3_ALIGN or key not in self.splits):
                    self.splits[key] = now                 # S3_ALIGN 取最後一次 (會從 S3_ALIGN 回 S2)
            if state == FINISHED:
                self.splits[FINISHED] = now
                self.timer_done = True
                self.report_time(now)
        self.s2_hold_until = now + S2_START_HOLD if (state == S2_FOLLOW and prev in (S1_LEFT, S1_WAIT0)) else 0.0
        if state == FINISHED:
            print("全部完成, 降落")
            self.want_land = True

    def elapsed(self, now):
        """按 g 之後經過幾秒 (沒按過 g 回傳 None)"""
        t0 = getattr(self, "run_t0", None)
        if t0 is None:
            return None
        end = self.splits.get(FINISHED) if getattr(self, "timer_done", False) else None
        return (end if end is not None else now) - t0

    def report_time(self, now):
        """印出總時間與各段時間 / 佔比 (按 g 開始計時)"""
        if getattr(self, "run_t0", None) is None:
            return
        total = self.elapsed(now)
        marks = [("g", self.run_t0)] + [(k, self.splits.get(k)) for k in SPLIT_STATES] + \
                [("降落", self.splits.get(FINISHED))]
        names = ["第1段 g -> S2_FOLLOW", "第2段 S2_FOLLOW -> S3_ALIGN", "第3段 S3_ALIGN -> S4",
                 "第4段 S4 -> S5_ALIGN", "第5段 S5_ALIGN -> 降落"]
        log("[計時] 總時間 %.1f 秒%s" % (total, "" if FINISHED in self.splits else " (還沒完成)"))
        for i, name in enumerate(names):
            a, b = marks[i][1], marks[i + 1][1]
            if a is None or b is None:
                log("[計時]   %-28s   -" % name)
            else:
                log("[計時]   %-28s %6.1f 秒  %5.1f%%" % (name, b - a, 100.0 * (b - a) / max(total, 1e-6)))

    def note(self, key, msg):
        if key not in self.notes:
            self.notes.add(key)
            print("  " + msg)

    def set_heading_ref(self):
        self.heading_ref = read_yaw(self.drone)
        self.heading_ok = self.heading_ref is not None
        if not self.heading_ok:
            print("  (讀不到 yaw, 不做航向保持)")

    def begin(self, state, now):
        """從手動直接進入某一段 (g h j k l)"""
        print("進入 %s" % state)
        if state == S1_APPROACH:                           # 按 g: 從這裡開始計時
            self.run_t0 = now
            self.splits = {}
            self.timer_done = False
            log("[計時] 開始")
        self.set_heading_ref()                             # 直線段的方向 = 按下按鍵時的方向
        self.enter(state, now)

    def abort(self, now, why):
        if self.state != MANUAL:
            print("[%.1f] 切回手動 (%s)" % (now, why))
        self.state = MANUAL

    # ---------------- 量測 ----------------
    def track(self, markers, mid, now):
        """目標 marker 的平滑量測 (x, y, z, ang); 看不到回傳 None"""
        m = markers.get(mid)
        if m is None:
            if self.lost_since is None:
                self.lost_since = now
            return None
        self.lost_since = None
        if self.ema_id != mid or now - self.ema_seen > 0.5:
            self.ema.reset()
            self.ema_id = mid
        self.ema_seen = now
        self.last_seen = now
        x, y, z = m["t"]
        self.last = self.ema.update((x, y, z, m["ang"]))
        return self.last

    # ---------------- 航向 ----------------
    def heading_err(self, target):
        yaw = read_yaw(self.drone)
        return None if yaw is None or target is None else wrap180(yaw - target)

    def heading_cmd(self):
        if not (HOLD_HEADING and self.heading_ok):
            return 0
        err = self.heading_err(self.heading_ref)
        if err is None:
            return 0
        if abs(err) > HEADING_ABORT:
            print("  航向偏離 %.0f 度, 停止航向修正 (檢查 HEADING_SIGN)" % err)
            self.heading_ok = False
            return 0
        cmd = p_ctrl(-HEADING_SIGN * err, HEADING_KP, HEADING_DEAD, HEADING_MAX, HEADING_MIN)
        if DEBUG_HEADING and time.time() - self.last_dbg >= 1.0:
            self.last_dbg = time.time()
            print("  [%s] err=%.0f -> %d" % (self.state, err, cmd))
        return cmd

    @staticmethod
    def lat(x):
        return p_ctrl(x, LAT_KP, LAT_DEAD, LAT_MAX)          # marker 在右 -> 往右

    @staticmethod
    def vert(y):
        return p_ctrl(-y, UD_KP, UD_DEAD, UD_MAX)            # marker 在下 (y>0) -> 往下

    @staticmethod
    def fwd(err, dead=FB_DEAD, cap=FB_MAX):
        return p_ctrl(err, FB_KP, dead, cap, FB_MIN)         # 太遠 -> 往前; 太近 -> 往後

    def alt_hold(self, target):
        alt = read_alt(self.drone)
        if alt is None or target is None:
            return 0
        return p_ctrl(target - alt, ALT_KP, ALT_DEAD, ALT_MAX)

    def arrived(self, cond):
        self.arrive = self.arrive + 1 if cond else 0
        return self.arrive >= ARRIVE_FRAMES

    @staticmethod
    def clear3(y, z):
        """marker 3 下緣比鏡頭高多少 (cm, 真正的垂直方向): 補償鏡頭往下傾 CAM_PITCH_DEG"""
        a = math.radians(CAM_PITCH_DEG)
        down = y * math.cos(a) + z * math.sin(a)          # marker 中心在鏡頭 "正下方" 多少 (負 = 在上方)
        return -(down + MARKER_SIZE / 2)

    @staticmethod
    def low_enough(alt):
        return S3_PASS_ALT is None or alt is None or alt <= S3_PASS_ALT

    def seek(self, mid, now):
        """對正中的 marker 跑出畫面: 依最後看到的位置往那個方向慢慢移動找回來; 沒有資料或超時就懸停"""
        if self.last is None or now - self.lost_since > SEEK_TIME:
            self.note("lost%d" % mid, "看不到 marker %d, 懸停等待" % mid)
            return (0, 0, 0, 0)
        x, y, z = self.last[0], self.last[1], max(self.last[2], 1.0)
        # 只有 marker 消失前靠近畫面邊緣才往那邊找; 在畫面中間消失 = 只是漏抓 (晃動 / 模糊 / Wi-Fi), 原地等
        # (原本只要偏離中心 6 cm 就全速往旁邊 + 往上飛, 實飛第五段每漏抓一次就被甩開一次)
        waited = now - self.lost_since > SEEK_WAIT          # 原地等了一陣子還是沒有: 不是漏抓, 真的跑掉了
        edge_x = waited and abs(x) > LAT_DEAD or abs(math.degrees(math.atan2(x, z))) > SEEK_EDGE_X_DEG
        edge_y = waited and abs(y) > LAT_DEAD or abs(math.degrees(math.atan2(y, z))) > SEEK_EDGE_Y_DEG
        lr = (SEEK_SPEED if x > 0 else -SEEK_SPEED) if edge_x else 0
        ud = (-SEEK_SPEED if y > 0 else SEEK_SPEED) if edge_y else 0   # y < 0: marker 在上方
        if not lr and not ud:
            self.note("seek%d" % mid, "暫時沒抓到 marker %d (在畫面中間), 原地等" % mid)
        else:
            self.note("seek%d" % mid, "marker %d 跑出畫面, 往%s找" % (
                mid, "".join(t for t, on in (("右" if x > 0 else "左", lr), ("下" if y > 0 else "上", ud)) if on)))
        return (lr, 0, ud, 0)

    def align(self, p, dist, tol, use_ud):
        """對正 marker: 左右置中、距離 dist、(可選) 高度置中、轉向正對。回傳 (rc, 是否對正)"""
        x, y, z, ang = p
        if ALIGN_YAW:
            yaw = p_ctrl(SIGN_ALIGN_YAW * ang, ALIGN_YAW_KP, ALIGN_YAW_DEAD, ALIGN_YAW_MAX, ALIGN_YAW_MIN)
        else:
            yaw = 0
        ud = p_ctrl(-y, UD_KP, ALIGN_UD_DEAD, UD_MAX, ALIGN_UD_MIN) if use_ud else 0
        rc = (self.lat(x), self.fwd(z - dist, dead=tol), ud, yaw)
        ok = (abs(x) < LAT_DEAD + 2 and abs(z - dist) <= tol
              and (not use_ud or abs(y) < ALIGN_UD_DEAD + 2)
              and (not ALIGN_YAW or abs(ang) < ALIGN_YAW_DEAD + 1))
        return rc, ok

    # ---------------- 每個週期 ----------------
    def step(self, now, markers):
        st = self.state
        el = now - self.t0

        if st in (S4_LEFT, S4_WAIT5, S5_ALIGN, S5_TABLE):   # 記錄地板的 TOF (取最大 = 下面是地板時)
            t = read_tof(self.drone)
            if t is not None:
                self.tof_floor = t if getattr(self, "tof_floor", None) is None else max(self.tof_floor, t)

        if st in PAUSES:                                   # 動作之間停一下, 保持航向
            if el >= PAUSE_TIME:
                self.enter(PAUSES[st], now)
            return (0, 0, 0, self.heading_cmd())

        if st in (S1_WAIT0, S3_WAIT4, S4_WAIT5):           # 停下來等目標 marker 出現
            want = {S1_WAIT0: (M0_ID, S2_FOLLOW), S3_WAIT4: (M4_ID, S3_ALIGN4),
                    S4_WAIT5: (M5_ID, S5_ALIGN)}[st]
            if self.arrived(want[0] in markers):
                self.enter(want[1], now)
                return (0, 0, 0, 0)
            return (0, 0, 0, self.heading_cmd())

        # ======================= 第一段 =======================
        if st == S1_APPROACH:
            if el > APPROACH_MAX_TIME:
                print("  接近 marker 1 超過 %.0f 秒, 直接右移" % APPROACH_MAX_TIME)
                self.enter(S1_PAUSE1, now)
                return (0, 0, 0, 0)
            last = self.last
            p = self.track(markers, M1_ID, now)
            if p is None:
                if last is not None and last[2] <= APPROACH_DIST + NEAR_LOST_EXTRA:
                    print("  靠太近看不到 marker 1 (最後 %.0f cm), 視為抵達" % last[2])
                    self.enter(S1_PAUSE1, now)
                else:
                    self.note("lost1", "看不到 marker 1, 懸停等待")
                return (0, 0, 0, self.heading_cmd())
            x, y, z, _ = p
            if self.arrived(z <= APPROACH_DIST + FB_DEAD):
                print("  到達 marker 1 前 %.0f cm" % z)
                self.enter(S1_PAUSE1, now)
                return (0, 0, 0, 0)
            fb = self.fwd(z - APPROACH_DIST)
            if z < SAFE_MIN_DIST:
                fb = min(fb, 0)
            return (self.lat(x), fb, self.vert(y), self.heading_cmd())

        if st == S1_RIGHT:
            p = self.track(markers, M2_ID, now)
            done = False
            if SIDE_MODE == "marker2":
                if p is not None:
                    self.note("see2", "右移中看到 marker 2, x = %.0f cm" % p[0])
                    done = el >= SIDE_MIN_TIME and p[0] <= SIDE_ALIGN_CM
                elif el >= SIDE_TIME and "see2" not in self.notes:
                    print("  右移 %.1f 秒沒看到 marker 2, 依時間停止" % SIDE_TIME)
                    done = True
            else:
                done = el >= SIDE_TIME
            if el >= SIDE_MAX_TIME:
                print("  右移達上限 %.1f 秒" % SIDE_MAX_TIME)
                done = True
            if done:
                self.enter(S1_PAUSE2, now)
                return (0, 0, 0, 0)
            return (SIDE_SPEED, 0, 0, self.heading_cmd())

        if st == S1_FORWARD:
            p = self.track(markers, M2_ID, now)
            if p is None:
                if el >= FWD_MAX_TIME:
                    self.note("nofind2", "往前 %.0f 秒仍沒看到 marker 2, 懸停 (可手動調整後再按 g)" % FWD_MAX_TIME)
                    return (0, 0, 0, self.heading_cmd())
                return (0, FWD_SPEED, 0, self.heading_cmd())
            x, y, z, _ = p
            err = z - M2_STOP_DIST
            if self.arrived(abs(err) <= M2_TOL):
                print("  到達 marker 2 前 %.0f cm" % z)
                self.enter(S1_PAUSE3, now)
                return (0, 0, 0, 0)
            # 死區 = 到位範圍, 否則會卡在 "不動但也還沒到位" 的距離
            return (self.lat(x), self.fwd(err, dead=M2_TOL), self.vert(y), self.heading_cmd())

        if st == S1_LEFT:
            if el >= LEFT_MIN_TIME and self.arrived(M0_ID in markers):
                print("  看到手持的 marker 0 (%.0f cm), 進入第二段" % markers[M0_ID]["d"])
                self.enter(S2_FOLLOW, now)
                return (0, 0, 0, 0)
            if el >= LEFT_MAX_TIME:
                print("  左移 %.0f 秒仍沒看到 marker 0, 原地等待" % LEFT_MAX_TIME)
                self.enter(S1_WAIT0, now)
                return (0, 0, 0, 0)
            return (-LEFT_SPEED, 0, 0, self.heading_cmd())

        # ======================= 第二段 =======================
        if st == S2_FOLLOW:
            p = self.track(markers, M0_ID, now)
            if M0_ID in markers:                           # 最後一次在畫面的左右 (原始值): 跑出畫面時往這邊轉
                rx, _, rz = markers[M0_ID]["t"]
                self.m0_side = 1 if rx > 0 else -1
                self.m0_side_bearing = math.degrees(math.atan2(rx, rz))
            # 自動進第三段: 之前看過 marker 0, 現在 marker 0 不在畫面但 marker 3 在 -> 停止找 marker 0, 對準 marker 3;
            # 這個狀態持續 S2_TO_S3_TIME 秒就進 S3_ALIGN (marker 0 又出現就回去跟)
            if M0_ID in markers:
                self.m0_ever = True
                self.m3_since = None
            if now < self.s2_hold_until:
                # 第一段左移剛看到 marker 0: 先原地懸停一下 (把左移的速度停掉), 再開始跟
                self.note("s2hold", "看到 marker 0, 先懸停 %.1f 秒再開始跟" % S2_START_HOLD)
                return (0, 0, 0, 0)
            if S2_AUTO_S3 and self.m0_ever and p is None:
                if M3_ID in markers:
                    self.m3_last = now
                    if self.m3_since is None:
                        self.m3_since = now
                        self.ema3.reset()
                        print("  marker 0 不見了, 看到 marker 3: 對準 marker 3, %.1f 秒後進第三段" % S2_TO_S3_TIME)
                    if now - self.m3_since >= S2_TO_S3_TIME:
                        print("  看到 marker 3 %.1f 秒, 進第三段" % (now - self.m3_since))
                        self.enter(S3_ALIGN, now)
                        return (0, 0, 0, 0)
                    m3 = markers[M3_ID]
                    p3 = self.ema3.update((m3["t"][0], m3["t"][1], m3["t"][2], m3["ang"]))
                    rc3, _ = self.align(p3, S3_DIST, S3_DIST_TOL, use_ud=False)
                    self.last_yaw_cmd = 0
                    return rc3
                if self.m3_since is not None and now - self.m3_last < S2_TO_S3_M3_GAP:
                    return (0, 0, 0, 0)                    # marker 3 只漏抓一下: 原地等, 不要開始轉
                self.m3_since = None
            if p is not None:
                x, y, z, _ = p
                bearing = math.degrees(math.atan2(x, z))   # marker 0 偏離正前方的角度, 右為正
                # 估計 marker 0 繞著無人機移動的角速度 = 畫面裡角度的變化 + 自己轉掉的角度
                if self.prev_bearing is not None and now - self.prev_bt > 0.5:
                    self.prev_bearing = None               # 中間斷掉太久: 重新開始估
                if self.prev_bearing is not None and now > self.prev_bt:
                    rate = (bearing - self.prev_bearing) / (now - self.prev_bt) + self.last_yaw_cmd * SPEED_SCALE_TURN * YAW_RC_TO_DEG
                    self.bearing_rate += FOLLOW_RATE_ALPHA * (rate - self.bearing_rate)
                else:
                    self.bearing_rate = 0.0
                self.prev_bearing, self.prev_bt = bearing, now
                self.last_bearing = bearing
                # 預測: 往 FOLLOW_LEAD 秒後 marker 0 會在的方向轉
                lead = max(-FOLLOW_PRED_MAX, min(FOLLOW_PRED_MAX, FOLLOW_LEAD * self.bearing_rate))
                target = bearing + lead
                # 轉向 + 左右平移一起把 marker 0 拉回畫面中心 (只靠轉向會跟不上)
                yaw = p_ctrl(target, FOLLOW_YAW_KP, FOLLOW_YAW_DEAD, FOLLOW_YAW_MAX, FOLLOW_YAW_MIN)
                self.last_yaw_cmd = yaw
                lr = p_ctrl(x, FOLLOW_LAT_KP, LAT_DEAD, FOLLOW_LAT_MAX)
                fb = p_ctrl(z - FOLLOW_DIST, FOLLOW_FB_KP, FOLLOW_DEAD, FOLLOW_FB_MAX, FB_MIN)
                # marker 0 偏離中心時先轉、少往前: 偏 FOLLOW_TURN_FIRST_DEG 度以上時前進只剩 30%
                # (實飛: 往前 58 + 轉向 -20 時看起來只有在往前, 沒在轉)
                if fb > 0 and FOLLOW_TURN_FIRST_DEG:
                    fb = int(fb * max(0.3, 1.0 - abs(bearing) / FOLLOW_TURN_FIRST_DEG))
                return (lr, fb, self.vert(y), yaw)
            # 只漏抓一下 (晃動 / 模糊 / Wi-Fi): 不要馬上全速轉去找 (實飛 log: 漏抓 1 張就以 55 轉, 轉太快畫面模糊更抓不到),
            # 先照剛剛的轉向繼續轉, 停止前後左右
            if self.last_bearing is not None and now - self.lost_since < FOLLOW_LOST_HOLD:
                return (0, 0, 0, self.last_yaw_cmd)
            # 找不到 marker 0: 一直往左轉找, 直到找到 (最多 FOLLOW_SEARCH_LEFT_MAX 秒, 約轉一圈)
            if FOLLOW_SEARCH_DIR == "left":
                if not self.m0_ever:
                    # 剛從第一段 (看到 m2 後左移) 進來, 這一段還沒看到過 marker 0: 不要自轉, 原地等它出現
                    self.last_yaw_cmd = 0
                    self.note("wait0first", "還沒看到 marker 0, 原地等 (看到一次之後不見才會往左轉找)")
                    return (0, 0, 0, 0)
                if self.lost_since is not None and now - self.lost_since < FOLLOW_LOST_HOLD + FOLLOW_SEARCH_LEFT_MAX:
                    self.note("search0", "找不到 marker 0, 往左轉找")
                    self.last_yaw_cmd = -FOLLOW_SEARCH_YAW
                    return (0, 0, 0, self.last_yaw_cmd)
                self.last_yaw_cmd = 0
                self.note("lost0", "往左轉 %.0f 秒還是找不到 marker 0, 懸停等待" % FOLLOW_SEARCH_LEFT_MAX)
                return (0, 0, 0, 0)
            # (FOLLOW_SEARCH_DIR = "last") marker 0 剛跑出畫面: 往它最後出現的那一側轉一下找回來
            if self.last_bearing is not None and now - self.lost_since < FOLLOW_LOST_HOLD + FOLLOW_SEARCH_TIME:
                side = getattr(self, "m0_side", 1 if self.last_bearing > 0 else -1)
                if FOLLOW_LEAD:                            # (有開速度預測時才用)
                    side = 1 if self.last_bearing + FOLLOW_LEAD * self.bearing_rate > 0 else -1
                self.note("search0", "marker 0 跑出畫面 (最後在畫面%s %.0f 度), 往%s轉找回來" % (
                    "右" if side > 0 else "左", abs(getattr(self, "m0_side_bearing", self.last_bearing)),
                    "右" if side > 0 else "左"))
                self.last_yaw_cmd = FOLLOW_SEARCH_YAW * side
                return (0, 0, 0, self.last_yaw_cmd)
            self.last_yaw_cmd = 0
            self.note("lost0", "看不到 marker 0, 懸停等待")
            return (0, 0, 0, 0)

        # ======================= 第三段: 鑽桌子 =======================
        if st == S3_ALIGN:
            # 還在對正 marker 3 時 marker 0 又出現: 人還沒走完, 回去跟 (之後要重新看到 marker 0 + 3 才會再自動切)
            if S3_BACK_TO_S2:
                self.m0_back = self.m0_back + 1 if M0_ID in markers else 0
                if self.m0_back >= S3_BACK_FRAMES:
                    print("  對正 marker 3 時又看到 marker 0, 回到第二段跟隨 (marker 0 不見 + 看到 marker 3 %.1f 秒就會再進第三段)"
                          % S2_TO_S3_TIME)
                    self.enter(S2_FOLLOW, now)
                    self.m0_ever = True                        # 剛剛有看到 marker 0
                    return (0, 0, 0, 0)
            p = self.track(markers, M3_ID, now)
            if p is None:
                return self.seek(M3_ID, now)
            rc, ok = self.align(p, S3_DIST, S3_DIST_TOL, use_ud=False)
            # 角度: 平均幾張再用, 而且要 < S3_YAW_TOL (原本 ±6 度就算正; 鑽桌子要往前飛 ~300 cm, 歪 5 度就偏 26 cm)
            if M3_ID in markers:
                self.ang3 = float(self.ema3a.update([markers[M3_ID]["ang"]])[0])
            ang3 = self.ang3 if self.ang3 is not None else p[3]
            yaw = (p_ctrl(SIGN_ALIGN_YAW * ang3, ALIGN_YAW_KP, S3_YAW_DEAD, ALIGN_YAW_MAX, ALIGN_YAW_MIN)
                   if ALIGN_YAW else 0)
            yaw_ok = not ALIGN_YAW or abs(ang3) < S3_YAW_TOL
            face_ok = yaw_ok and abs(p[0]) < S3_LAT_TOL           # 正對: 左右置中 + 角度正
            dt = 0.0 if self.lat_t is None else min(0.2, now - self.lat_t)
            self.lat_t = now
            if self.lat_i * p[0] < 0 and abs(p[0]) > LAT_DEAD:
                self.lat_i = 0.0                           # 已經偏到另一邊: 積分歸零 (實飛: 積分 +12 一直推, 推過頭往右偏)
            if ALIGN_YAW and abs(ang3) >= S3_YAW_TOL + 2:
                # 角度差很多: 先轉正, 不修左右 (邊轉邊修, 每轉一點 marker 在畫面的左右就跳, 積分會越推越歪); 積分先凍結
                rc = (0, rc[1], 0, yaw)
            else:
                if abs(p[0]) > LAT_DEAD:
                    self.lat_i = max(-S3_LAT_I_MAX, min(S3_LAT_I_MAX, self.lat_i + S3_LAT_KI * p[0] * dt))
                rc = (limit(rc[0] + self.lat_i, LAT_MAX + S3_LAT_I_MAX), rc[1], 0, yaw)
            ok = abs(p[2] - S3_DIST) <= S3_DIST_TOL and face_ok
            if now - getattr(self, "_s3_dbg", 0.0) >= 1.0:   # 每秒印一次, 看 log 才知道角度有沒有在收斂
                self._s3_dbg = now
                print("  [S3_ALIGN] x %+.0f  z %.0f  角度 %+.0f (平均 %+.1f)  rc %s  (左右積分 %+.0f)" % (p[0], p[2], p[3], ang3, rc, self.lat_i))
            if not self.arrived(ok) and el > ALIGN_MAX_TIME:
                if not face_ok:
                    # 還沒正對: 不可以往前 (會斜斜 / 偏一邊鑽過去), 距離先不管, 只修左右和角度;
                    # 再修 S3_YAW_EXTRA_TIME 秒還不行就懸停
                    if el > ALIGN_MAX_TIME + S3_YAW_EXTRA_TIME:
                        self.note("s3yaw", "對正 marker 3 一直對不正 (左右 %+.0f cm, 角度 %+.0f), 懸停: 請手動對正後再按 j"
                                  % (p[0], p[3]))
                        return (0, 0, 0, 0)
                    self.note("s3yawonly", "對正 marker 3 超過 %.0f 秒, 距離先不管, 只修左右和角度 (左右 %+.0f cm, 角度 %+.0f)"
                              % (ALIGN_MAX_TIME, p[0], p[3]))
                    lr = p_ctrl(p[0], LAT_KP, LAT_DEAD, LAT_MAX, S3_LAT_MIN) + self.lat_i
                    return (limit(lr, LAT_MAX + S3_LAT_I_MAX), 0, 0, rc[3])
                print("  對正 marker 3 超過 %.0f 秒, 已正對 (左右 %+.0f cm, 角度 %+.0f), 距離 %.0f cm 就從這裡繼續"
                      % (ALIGN_MAX_TIME, p[0], p[3], p[2]))
                ok = True
            if ok and (self.arrive >= ARRIVE_FRAMES or el > ALIGN_MAX_TIME):
                print("  已正對 marker 3 (%.0f cm, 左右 %+.0f cm, 角度 %+.1f (平均)), 開始下降" % (p[2], p[0], ang3))
                self.set_heading_ref()                     # 鑽桌子的方向 = 正對 marker 3 的方向
                self.enter(S3_DOWN, now)
                return (0, 0, 0, 0)
            return rc

        if st == S3_DOWN:
            last = self.last
            p = self.track(markers, M3_ID, now)
            alt = read_tof(self.drone)
            if p is not None:
                x, y, z, _ = p
                clear = self.clear3(y, z)                  # marker 下緣比鏡頭高多少 (cm)
                if self.arrived(clear >= S3_CLEAR and self.low_enough(alt)):
                    print("  marker 3 下緣高於鏡頭 %.0f cm (高度 %s), 開始鑽桌子" %
                          (clear, "%.0f cm" % alt if alt is not None else "?"))
                    self.s3_z0 = z
                    self.enter(S3_UNDER, now)
                    return (0, 0, 0, 0)
                lr = self.lat(x)
                fb = self.fwd(z - S3_DIST, dead=S3_DIST_TOL)
            else:
                # 下降時 marker 3 從畫面上緣跑掉: 它已經在上方, 再多降一點保險後就鑽
                if last is not None and self.clear3(last[1], last[2]) >= 0:
                    if alt is not None and S3_PASS_ALT is not None:
                        if alt <= S3_PASS_ALT:             # 有 TOF: 降到 S3_PASS_ALT 以下才鑽
                            print("  marker 3 已移出畫面上方, 高度 %.0f cm, 開始鑽桌子" % alt)
                            self.s3_z0 = last[2]
                            self.enter(S3_UNDER, now)
                            return (0, 0, 0, 0)
                    elif now - self.lost_since >= S3_LOST_EXTRA:
                        print("  marker 3 已移出畫面上方, 多降 %.1f 秒後開始鑽桌子" % S3_LOST_EXTRA)
                        self.s3_z0 = last[2]
                        self.enter(S3_UNDER, now)
                        return (0, 0, 0, 0)
                else:
                    # 偶爾一兩張 frame 沒抓到 (晃動 / Wi-Fi): 不要停下來懸停 (實飛會一直停停走走直到超時),
                    # 只停止前後左右修正, 繼續往下; 最低高度 MIN_ALT 仍會擋住
                    self.note("lost3d", "下降時暫時看不到 marker 3, 繼續往下")
                lr = fb = 0
            if alt is not None and alt <= MIN_ALT:
                self.note("minalt", "已到最低高度 %.0f cm, marker 3 仍不夠高, 不往前鑽 (請手動處理)" % alt)
                return (0, 0, 0, self.heading_cmd())
            if el > S3_DOWN_MAX_TIME:
                self.note("downmax", "下降超過 %.0f 秒仍不夠低, 懸停" % S3_DOWN_MAX_TIME)
                return (0, 0, 0, self.heading_cmd())
            return (lr, fb, -S3_DOWN_SPEED, self.heading_cmd())

        if st == S3_UNDER:
            if el >= self.under_time:
                print("  已鑽過桌子")
                self.enter(S3_PAUSE1, now)
                return (0, 0, 0, 0)
            # 還看得到 marker 3 時再檢查一次: 下緣不夠高就先停下來往下 (絕不往前撞)
            p = self.track(markers, M3_ID, now)
            if p is not None and self.clear3(p[1], p[2]) < S3_CLEAR * 0.5:
                self.note("lowclear", "鑽桌子時 marker 3 太低, 先停下來下降")
                self.t0 += CONTROL_PERIOD                  # 這段時間不算在往前的秒數裡
                return (0, 0, -S3_DOWN_SPEED, self.heading_cmd())
            ud = self.alt_hold(self.hold_alt)
            alt = read_alt(self.drone)
            if ud > 0 and not (alt is not None and alt < MIN_ALT):
                ud = 0                                     # 桌子底下不准往上修 (除非快貼地)
            return (0, S3_UNDER_SPEED, ud, self.heading_cmd())

        if st == S3_TURN:
            err = self.heading_err(self.turn_target)
            if err is None:                                # 讀不到 yaw: 用內建指令轉
                print("  讀不到 yaw, 改用 rotate_clockwise(%d)" % S3_TURN_DEG)
                try:
                    self.drone.rotate_clockwise(int(S3_TURN_DEG))
                except Exception as e:
                    print("  rotate failed:", e)
                self.enter(S3_UP, now)
                return (0, 0, 0, 0)
            if self.arrived(abs(err) <= TURN_TOL) or el > TURN_MAX_TIME:
                print("  已轉 %.0f 度 (誤差 %.0f)" % (S3_TURN_DEG, err))
                self.set_heading_ref()
                self.enter(S3_UP, now)
                return (0, 0, 0, 0)
            return (0, 0, 0, p_ctrl(-HEADING_SIGN * err, HEADING_KP, TURN_TOL, TURN_MAX, HEADING_MIN))

        if st == S3_UP:
            if self.arrived(M4_ID in markers):
                print("  看到 marker 4")
                self.enter(S3_ALIGN4, now)
                return (0, 0, 0, 0)
            alt = read_tof(self.drone)
            high = S3_UP_ALT is not None and alt is not None and alt >= S3_UP_ALT
            if high or el >= S3_UP_MAX_TIME:
                print("  往上到 %s 仍沒看到 marker 4, 往後退 + 左右找" % ("%.0f cm" % alt if alt is not None else "%.0f 秒" % el))
                self.enter(S3_FIND4, now)
                return (0, 0, 0, 0)
            return (0, 0, S3_UP_SPEED, self.heading_cmd())

        if st == S3_FIND4:
            # 不靠鑽了多遠: 停在 marker 4 的高度, 先後退讓視野變寬, 再往左 / 往右平移找 marker 4
            if self.arrived(M4_ID in markers):
                print("  找到 marker 4 (找了 %.1f 秒)" % el)
                self.enter(S3_ALIGN4, now)
                return (0, 0, 0, 0)
            ud = self.alt_hold(self.alt0)                  # 維持剛升到的高度
            t1 = S3_FIND4_BACK
            t2 = t1 + S3_FIND4_TIME
            t3 = t2 + 2 * S3_FIND4_TIME
            if el < t1:
                return (0, -S3_FIND4_SPEED, ud, self.heading_cmd())
            if el < t2:
                self.note("find4l", "往左找 marker 4")
                return (-S3_FIND4_SPEED, 0, ud, self.heading_cmd())
            if el < t3:
                self.note("find4r", "往右找 marker 4")
                return (S3_FIND4_SPEED, 0, ud, self.heading_cmd())
            print("  左右都找不到 marker 4, 原地等待 (可手動移到看得到 marker 4 後按 k)")
            self.enter(S3_WAIT4, now)
            return (0, 0, 0, 0)

        if st == S3_ALIGN4:
            p = self.track(markers, M4_ID, now)
            if p is None:                                  # 剛升上來時 marker 4 常只在畫面邊緣
                if self.last is None and el > S4_FIND_M4_TIME:
                    # 按 k 進來但一直沒看到 marker 4: 照舊以目前方向進第四段
                    print("  %.0f 秒沒看到 marker 4, 以目前方向進第四段" % S4_FIND_M4_TIME)
                    self.set_heading_ref()
                    self.enter(S4_CLIMB, now)
                    return (0, 0, 0, 0)
                return self.seek(M4_ID, now)
            # 對正 marker 4: 前後 / 上下用比較快的速度, 容許誤差放寬一點 (實飛常要 5~8 秒: 剛找到時離牆太近要退很多,
            # 高度只用 8~11 的速度慢慢爬, 而且要對到 ±7 cm)
            x, y, z, ang = p
            yaw = (p_ctrl(SIGN_ALIGN_YAW * ang, ALIGN_YAW_KP, ALIGN_YAW_DEAD, ALIGN_YAW_MAX, ALIGN_YAW_MIN)
                   if ALIGN_YAW else 0)
            rc = (self.lat(x),
                  p_ctrl(z - M4_DIST, FB_KP, S4A_DIST_TOL / 2, S4A_FB_MAX, FB_MIN),
                  p_ctrl(-y, S4A_UD_KP, ALIGN_UD_DEAD, S4A_UD_MAX, ALIGN_UD_MIN),
                  yaw)
            yaw_ok = not ALIGN_YAW or abs(ang) < ALIGN_YAW_DEAD + 1
            ok = (abs(x) < S4A_LAT_TOL and abs(z - M4_DIST) <= S4A_DIST_TOL and abs(y) < S4A_UD_TOL and yaw_ok)
            if now - getattr(self, "_s4_dbg", 0.0) >= 1.0:
                self._s4_dbg = now
                print("  [S3_ALIGN4] x %+.0f  y %+.0f  z %.0f  角度 %+.0f  rc %s" % (p[0], p[1], p[2], p[3], rc))
            if not self.arrived(ok) and el > ALIGN_MAX_TIME:
                if not yaw_ok:
                    # 沒正對 marker 4 就往左, 第四段會斜著飛, 第五段降落就不準: 先只轉正
                    if el > ALIGN_MAX_TIME + S3_YAW_EXTRA_TIME:
                        self.note("s4yaw", "對正 marker 4 一直轉不正 (角度 %+.0f), 懸停: 請手動轉正後再按 k" % p[3])
                        return (0, 0, 0, 0)
                    self.note("s4yawonly", "對正 marker 4 超過 %.0f 秒, 位置先不管, 只把角度轉正 (現在 %+.0f 度)"
                              % (ALIGN_MAX_TIME, p[3]))
                    return (rc[0], 0, rc[2], rc[3])
                ok = True
            if ok and (self.arrive >= ARRIVE_FRAMES or el > ALIGN_MAX_TIME):
                print("  第三段完成: 已對正 marker 4 (%.0f cm, 角度 %+.0f)%s" %
                      (p[2], p[3], "" if self.arrive >= ARRIVE_FRAMES else " (超時, 角度已正, 以目前位置繼續)"))
                self.set_heading_ref()                     # 沿牆平移的方向 = 正對 marker 4 的方向
                self.enter(S4_CLIMB, now)
                return (0, 0, 0, 0)
            return rc

        # ======================= 第四段: 越過板子 =======================
        if st == S4_CLIMB:
            if S4_CLIMB_CM <= 0:                           # 不用爬升: 直接以目前 (marker 4 的) 高度向左
                self.enter(S4_LEFT, now)
                self.alt_left = self.alt0
                return (0, 0, 0, 0)
            alt = read_alt(self.drone)
            climbed = None if (alt is None or self.alt0 is None) else alt - self.alt0
            if (climbed is not None and climbed >= S4_CLIMB_CM) or el >= S4_CLIMB_MAX_TIME:
                print("  爬升完成 (%s)" % ("+%.0f cm" % climbed if climbed is not None else "依時間"))
                base = self.alt0
                self.enter(S4_LEFT, now)
                self.alt_left = self.alt0                  # 向左時要維持的高度 (爬升後)
                self.alt0 = base                           # 記住爬升前的高度, 過板子後降回來
                return (0, 0, 0, 0)
            return (0, 0, S4_CLIMB_SPEED, self.heading_cmd())

        if st == S4_LEFT:
            if self.arrived(M5_ID in markers):
                print("  第四段完成: 看到 marker 5")
                self.enter(S5_ALIGN, now)
                return (0, 0, 0, 0)
            if el >= S4_MAX_TIME:
                print("  向左 %.0f 秒仍沒看到 marker 5, 原地等待" % S4_MAX_TIME)
                self.enter(S4_WAIT5, now)
                return (0, 0, 0, 0)
            if S4_CLIMB_CM > 0 and S4_DESCEND_AFTER and el >= S4_CROSS_TIME:
                ud = min(self.alt_hold(self.alt0), 0)      # 已越過板子: 降回爬升前的高度
            else:
                ud = self.alt_hold(self.alt_left)          # 維持高度, 不要往下掉到板子
            return (-S4_LEFT_SPEED, 0, ud, self.heading_cmd())

        # ======================= 第五段: 降落在桌上 =======================
        if st == S5_TABLE:
            # 用 TOF 確認 / 找桌子: 下面是桌子時 TOF 會比地板小一個桌子的高度
            p = self.track(markers, M5_ID, now)
            tof = read_tof(self.drone)
            if p is None:
                if self.lost_since is not None and now - self.lost_since > S5_TABLE_LOST:
                    print("  找桌子時 %.0f 秒看不到 marker 5, 直接降落" % S5_TABLE_LOST)
                    self.enter(FINISHED, now)
                return (0, 0, 0, 0)
            moved = self.tbl_x0 - p[0]                     # 無人機往右移了多少 (cm): 往右 -> marker 在畫面往左
            fb = self.fwd(p[2] - M5_DIST, dead=M5_TOL)
            on = tof is not None and self.tof_floor is not None and tof < self.tof_floor - S5_TABLE_DROP
            self.tbl_on = self.tbl_on + 1 if on else 0
            if now - getattr(self, "_s5t_dbg", 0.0) >= 1.0:
                self._s5t_dbg = now
                print("  [S5_TABLE] 移了 %+.0f cm  tof %s  地板 %s  %s" % (moved, "%.0f" % tof if tof is not None else "?",
                      "%.0f" % self.tof_floor if self.tof_floor is not None else "?", "桌子" if on else "地板"))
            if self.tbl_found is None:
                if self.tbl_on >= 3:
                    if not getattr(self, "tbl_scanned", False):
                        print("  下面就是桌子 (tof %.0f, 地板 %.0f), 降落" % (tof, self.tof_floor))
                        self.enter(FINISHED, now)
                        return (0, 0, 0, 0)
                    self.tbl_found = moved
                    print("  找到桌子邊緣 (往%s %.0f cm), 再往裡面 %.0f cm" % ("右" if moved > 0 else "左", abs(moved), S5_TABLE_IN_CM))
                elif self.tof_floor is None or tof is None:
                    print("  讀不到 TOF, 無法確認桌子, 直接降落")
                    self.enter(FINISHED, now)
                    return (0, 0, 0, 0)
                elif self.tbl_on > 0:
                    return (0, fb, 0, 0)                   # TOF 剛變小: 先停下來確認是不是桌子
                else:
                    if self.tbl_dir * moved >= S5_SCAN_CM:
                        if self.tbl_dir == S5_SCAN_FIRST:
                            self.tbl_dir = -self.tbl_dir
                            print("  往%s %.0f cm 沒找到桌子, 換邊找" % ("右" if moved > 0 else "左", S5_SCAN_CM))
                        else:
                            print("  左右都沒找到桌子 (TOF 沒有變小), 回到對準 marker 5 的位置降落")
                            self.tbl_found = 0.0
                            self.tbl_dir = 0
                    self.tbl_scanned = True
                    self.note("s5scan", "下面不是桌子 (tof %s, 地板 %.0f), 往%s找桌子" % (
                        "%.0f" % tof, self.tof_floor, "右" if self.tbl_dir > 0 else "左"))
                    return (self.tbl_dir * S5_SCAN_SPEED, fb, 0, 0)
            target = self.tbl_found + self.tbl_dir * S5_TABLE_IN_CM
            err = target - moved
            if abs(err) < 4:
                if self.tbl_ok_t is None:
                    self.tbl_ok_t = now
                if now - self.tbl_ok_t >= S5_SETTLE:
                    print("  在桌子上方 (tof %s), 降落" % ("%.0f" % tof if tof is not None else "?"))
                    self.enter(FINISHED, now)
                return (0, 0, 0, 0)
            self.tbl_ok_t = None
            return (p_ctrl(err, S5_LAT_KP, 2, S5_SCAN_SPEED, 8), fb, 0, 0)

        if st == S5_ALIGN:
            p = self.track(markers, M5_ID, now)
            if p is None:                                  # 剛移過來時 marker 5 常只在畫面邊緣
                self.s5_n = 0
                return self.seek(M5_ID, now)
            # 左右要對得比其他段準 (桌子只有 50 cm 寬): 死區 / 容許誤差都比較小, 而且有最小速度推得動
            # 機頭要先正對牆, 畫面裡的左右才等於沿牆的左右: 機頭歪 7 度時畫面置中, 170 cm 外沿牆會差約 20 cm
            # (實飛就是這樣落在桌子左邊)。單張的角度雜訊有 ±5 度, 所以角度先平均很多張再轉, 死區也收小
            if M5_ID in markers:
                self.ang5 = float(self.ema5.update([markers[M5_ID]["ang"]])[0])
            ang5 = self.ang5 if self.ang5 is not None else p[3]
            ex = p[0] + S5_LAT_TRIM                        # marker 在畫面右邊 (x>0) -> 往右; 加 TRIM 讓降落點往右偏
            zz = p[2]
            yaw = (p_ctrl(SIGN_ALIGN_YAW * ang5, ALIGN_YAW_KP, S5_YAW_DEAD, ALIGN_YAW_MAX, S5_YAW_MIN)
                   if ALIGN_YAW else 0)
            if S5_SIMPLE:
                # 簡單版: 連續 S5_ARRIVE_FRAMES 張都看到 marker 5、而且在畫面中心 + 距離對 -> 降落
                lr = p_ctrl(ex, S5_LAT_KP, S5_LAT_DEAD, S5_LAT_MAX, S5_LAT_MIN)
                fb = self.fwd(zz - M5_DIST, dead=M5_TOL)
                ok = abs(ex) < S5_LAT_TOL and abs(zz - M5_DIST) <= M5_TOL
                self.s5_n = self.s5_n + 1 if ok else 0
                if self.s5_n >= S5_ARRIVE_FRAMES:
                    print("  連續 %d 張 marker 5 在中心 (離 %.0f cm, 左右偏 %+.0f cm, 角度 %+.1f), 降落"
                          % (self.s5_n, zz, p[0], ang5))
                    self.enter(FINISHED, now)
                    return (0, 0, 0, 0)
                return (lr, fb, 0, yaw)
            # 剛看到 marker 5 時還帶著第四段向左的速度: 先只煞車懸停 S5_SETTLE 秒, 不要邊滑邊對
            if el < S5_SETTLE:
                self.note("s5settle", "看到 marker 5, 先煞車懸停 %.1f 秒" % S5_SETTLE)
                return (0, 0, 0, 0)
            lr = p_ctrl(ex, S5_LAT_KP, S5_LAT_DEAD, S5_LAT_MAX, S5_LAT_MIN)
            fb = self.fwd(zz - M5_DIST, dead=M5_TOL)
            if ALIGN_YAW and abs(ang5) >= S5_YAW_TOL:
                # 先轉正再對左右: 邊轉邊對, 每轉一點畫面裡的左右就跳, 會一直左右來回
                self.s5_hist = []
                self.note("s5yaw", "先轉正 (角度 %+.1f)" % ang5)
                return (0, fb, 0, yaw)
            if abs(ex) < S5_LAT_AIM:
                lr = 0                                     # 已經夠準: 不要再推 (最小速度會推過頭), 讓它停下來
            ok = (abs(ex) < S5_LAT_TOL and abs(zz - M5_DIST) <= M5_TOL
                  and (not ALIGN_YAW or abs(ang5) < S5_YAW_TOL))
            rough = abs(ex) < 1.5 * S5_LAT_TOL and abs(zz - M5_DIST) <= 1.5 * M5_TOL
            # 只在 "到位 (或超時後大致到位) 而且真的停住了" 才降落: 位置在 S5_STILL_TIME 秒內晃動不超過 S5_STILL_CM
            if ok or (el > S5_MAX_TIME and rough):
                self.s5_hist.append((now, ex, zz))
                while self.s5_hist and now - self.s5_hist[0][0] > S5_STILL_TIME:
                    self.s5_hist.pop(0)
                span = now - self.s5_hist[0][0]
                xs = [h[1] for h in self.s5_hist]
                zs = [h[2] for h in self.s5_hist]
                still = max(xs) - min(xs) <= S5_STILL_CM and max(zs) - min(zs) <= 2 * S5_STILL_CM
                if span >= S5_STILL_TIME * 0.9 and still:
                    print("  對準 marker 5 且停穩 (離 marker 5 %.0f cm, 左右偏 %+.0f cm, 角度 %+.1f (平均))%s" %
                          (zz, p[0], ang5, "" if ok else " (超時, 大致到位)"))
                    if S5_TABLE_CHECK:
                        self.tbl_x0, self.tbl_dir, self.tbl_found, self.tbl_on, self.tbl_ok_t = p[0], S5_SCAN_FIRST, None, 0, None
                        self.tbl_scanned = False
                        self.enter(S5_TABLE, now)
                    else:
                        self.enter(FINISHED, now)
                    return (0, 0, 0, 0)
                if ok:
                    return (lr, 0, 0, yaw)                 # 到位了: 不再前後推, 等它停穩
            else:
                self.s5_hist = []
            return (lr, fb, 0, yaw)

        return (0, 0, 0, 0)


# ============================================================================
def main():
    K, dist = load_calib()
    detect = make_detector()

    half = MARKER_SIZE / 2
    obj_pts = np.array([[-half,  half, 0],
                        [ half,  half, 0],
                        [ half, -half, 0],
                        [-half, -half, 0]], dtype=np.float32)

    drone = Tello()
    drone.connect()
    print("Battery:", drone.get_battery(), "%")
    drone.streamon()
    frame_read = drone.get_frame_read()

    apply_speed_scale()
    if (SPEED_SCALE_MOVE, SPEED_SCALE_UPDOWN, SPEED_SCALE_TURN) != (1.0, 1.0, 1.0):
        print("速度倍率: 平移 x%.2f, 上下 x%.2f, 旋轉 x%.2f" % (SPEED_SCALE_MOVE, SPEED_SCALE_UPDOWN, SPEED_SCALE_TURN))
    last_sent = [time.time()]
    _raw_send = drone.send_rc_control

    def _tracked_send(*a):
        last_sent[0] = time.time()
        _raw_send(*a)

    drone.send_rc_control = _tracked_send                  # 記錄最後一次送指令的時間 (給 keepalive 用)
    course = Course(drone)
    manual_drone = LimitedDrone(drone, MANUAL_MAX)
    key_expire = 0.0
    interrupted = False
    battery = None
    last_bat = 0.0
    temp = None                                       # (templ, temph) °C
    last_mon = last_log = 0.0
    ground_since = None                               # 開始讀到高度 0 的時間 (偵測自己降落)
    self_landed = temp_warned = False
    max_temp, min_bat = None, None
    last_raw = None
    seen_raw = None
    n_new = n_done = 0                                # 畫面張數統計 (算 fps)
    fps_t = time.time()
    last_ctrl = 0.0

    last_loop = time.time()
    try:
        while True:
            now = time.time()
            if now - last_loop > LOOP_WARN:
                print("  [主迴圈卡住 %.1f 秒: 這段時間按鍵不會有反應]" % (now - last_loop))
            last_loop = now

            # ---- 保持連線: 太久沒送指令就補一個懸停, 避免 Tello 15 秒自動降落 ----
            if getattr(drone, "is_flying", False) and not key_expire and now - last_sent[0] >= KEEPALIVE:
                drone.send_rc_control(0, 0, 0, 0)

            # ---- 鍵盤: 最高優先權 ------------------------------------------
            key = cv2.waitKey(1)
            if key != -1:
                key &= 0xFF
            if key in (27, ord('q')):
                break
            if key in START_KEYS:
                if course.state == S2_FOLLOW and key == ord('j'):
                    drone.send_rc_control(0, 0, 0, 0)      # 第二段跟隨中按 j: 切到第三段
                    course.begin(S3_ALIGN, time.time())
                elif course.state != MANUAL and course.state not in WAIT_STATES:
                    pass                                   # 自動飛行進行中, 忽略
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                    key_expire = 0.0
                    if not getattr(drone, "is_flying", False):
                        print("還沒起飛: 自動起飛")
                        try:
                            drone.takeoff()
                        except Exception as e:
                            print("takeoff failed:", e)
                    now = time.time()
                    if getattr(drone, "is_flying", False):
                        course.begin(START_KEYS[key], now)
                    else:
                        print("沒有起飛成功, 維持手動")
            elif key != -1:
                if key in OPERATION_KEYS:
                    if course.state != MANUAL:
                        course.abort(now, "key %c" % key)
                        drone.send_rc_control(0, 0, 0, 0)
                try:
                    keyboard(manual_drone, key)
                except Exception as e:
                    print("command failed (key %d): %s" % (key, e))
                if key in (ord('1'), ord('2'), ord('3')):
                    drone.send_rc_control(0, 0, 0, 0)
                    key_expire = 0.0
                elif key not in NON_MOVE_KEYS:
                    key_expire = time.time() + KEY_HOLD

            if key_expire and time.time() >= key_expire:
                drone.send_rc_control(0, 0, 0, 0)
                key_expire = 0.0

            # ---- 狀態監控: 溫度 / 電量 / 自己降落 (影像卡住時也照常跑) ----------
            if now - last_mon >= MONITOR_PERIOD:
                last_mon = now
                t = read_temp(drone)
                if t is not None:
                    temp = t
                    max_temp = t[1] if max_temp is None else max(max_temp, t[1])
                b = read_battery(drone)
                if b is not None:
                    battery = b
                    min_bat = b if min_bat is None else min(min_bat, b)
                h = state_field(drone, "h")
                tof = state_field(drone, "tof")
                in_air = bool(getattr(drone, "is_flying", False))
                desc = "電量 %s%%  溫度 %s  高度 h=%s tof=%s  %s  %s" % (
                    "?" if battery is None else battery,
                    "?" if temp is None else "%.0f~%.0f°C" % temp,
                    "?" if h is None else "%.0f" % h, "?" if tof is None else "%.0f" % tof,
                    course.state, "FLYING" if in_air else "ON GROUND")
                if now - last_log >= STATUS_LOG_PERIOD:
                    last_log = now
                    span = max(now - fps_t, 1e-6)
                    desc += "  畫面 %.1f fps (處理 %.1f fps)" % (n_new / span, n_done / span)
                    n_new = n_done = 0
                    fps_t = now
                    log("[狀態] " + desc)
                if temp is not None and temp[1] >= TEMP_WARN and not temp_warned:
                    temp_warned = True
                    log("[警告] 溫度過高! " + desc)
                elif temp is not None and temp[1] < TEMP_WARN - 3:
                    temp_warned = False
                # 程式以為還在飛 (沒送過 land), 但高度一直是 0 → 多半是 Tello 自己降落 (過熱 / 沒電 / 保護)
                if in_air and h is not None and h <= 0 and (tof is None or tof < 20):
                    if ground_since is None:
                        ground_since = now
                    if not self_landed and now - ground_since >= SELF_LAND_TIME:
                        self_landed = True
                        log("[!!!] 偵測到無人機可能自己降落 (沒有送 land): " + desc)
                else:
                    ground_since = None
                    self_landed = False

            # ---- 取影像 (限制控制頻率) ---------------------------------------
            raw = frame_read.frame
            if raw is not None and raw is not seen_raw:    # 統計: 無人機傳來幾張新畫面 (fps)
                seen_raw = raw
                n_new += 1
            if raw is None or raw is last_raw or now - last_ctrl < CONTROL_PERIOD:
                continue
            last_raw = raw
            last_ctrl = now
            n_done += 1
            # djitellopy 2.5 回傳的是 RGB, OpenCV 用 BGR: 不轉的話紅藍對調, 畫面看起來像負片
            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if FRAME_IS_RGB else raw.copy()

            if now - last_bat >= BATTERY_PERIOD:
                last_bat = now
                b = read_battery(drone)
                if b is not None:
                    battery = b

            markers = detect_markers(frame, detect, obj_pts, K, dist)
            flying = bool(getattr(drone, "is_flying", False))

            # ---- 自動飛行 ----------------------------------------------------
            rc = (0, 0, 0, 0)
            if course.state != MANUAL:
                if not flying:
                    course.abort(now, "not flying")
                else:
                    rc = scale_rc(*course.step(now, markers))
                    drone.send_rc_control(*rc)
                    if course.want_land:
                        course.want_land = False
                        drone.send_rc_control(0, 0, 0, 0)
                        try:
                            drone.land()
                        except Exception as e:
                            print("land failed:", e)
                        course.abort(now, "landed")

            put(frame, "%s  %s  rc=%s" % (course.state, "FLYING" if flying else "ON GROUND", rc),
                (10, 25), (0, 255, 0))
            if battery is not None:
                put(frame, "Battery: %d%%" % battery, (10, 50),
                    (0, 0, 255) if battery < 20 else (0, 255, 0))
            if temp is not None:
                put(frame, "Temp: %.0f-%.0f C" % temp, (200, 50),
                    (0, 0, 255) if temp[1] >= TEMP_WARN else (0, 255, 0))
            el_run = course.elapsed(now)
            if el_run is not None:
                put(frame, "Time: %.1f s" % el_run, (400, 50), (0, 255, 255))
            if course.state == MANUAL:
                put(frame, "g/h/j/k/l: start segment 1-5", (10, 75), (0, 200, 255))
            cv2.imshow(WINDOW, frame)

    except KeyboardInterrupt:                          # Ctrl+C (SIGINT): 立刻降落
        interrupted = True
        print("\nSIGINT: 發送降落指令")
    finally:
        try:
            drone.send_rc_control(0, 0, 0, 0)
            if interrupted or getattr(drone, "is_flying", False):
                drone.land()
        except BaseException as e:
            print("land failed:", e)
        if getattr(course, "run_t0", None) is not None and not getattr(course, "timer_done", False):
            course.report_time(time.time())
        log("[結束] 這次最高溫度 %s, 最低電量 %s" % (
            "?" if max_temp is None else "%.0f°C" % max_temp, "?" if min_bat is None else "%d%%" % min_bat))
        drone.streamoff()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
