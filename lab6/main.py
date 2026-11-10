"""
main.py - 期中賽道 (EAB06): 第一段 ~ 第五段

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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # 讓 core/ 和 keyboard_djitellopy 找得到

from djitellopy import Tello                                        # noqa: E402

from core.app import run                                            # noqa: E402


def main():
    run(Tello())


if __name__ == '__main__':
    main()
