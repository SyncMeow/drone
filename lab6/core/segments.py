"""segments.py - 五段賽道每個狀態的控制邏輯 (Course.step 依狀態分派)

Segments 是 Course 的父類別 (mixin): 這裡的 self 就是 Course, 用到的 track / align / seek / enter 等在 machine.py
每個 step_sN 回傳 rc (左右, 前後, 上下, 旋轉), 還沒套速度倍率
"""
import math

from core.control import limit, p_ctrl
from core.params import P
from core.states import (S1_APPROACH, S1_PAUSE1, S1_RIGHT, S1_PAUSE2, S1_FORWARD, S1_PAUSE3, S1_LEFT,
                         S1_WAIT0, S2_FOLLOW, S3_ALIGN, S3_DOWN, S3_UNDER, S3_PAUSE1, S3_TURN, S3_UP, S3_WAIT4,
                         S3_ALIGN4, S3_FIND4, S4_CLIMB, S4_LEFT, S4_WAIT5, S5_ALIGN, S5_TABLE, FINISHED)
from core.telemetry import read_alt, read_tof


class Segments(object):
    def step_s1(self, st, now, el, markers):
        """第一段: marker 1, 2 -> 左移找 marker 0"""
        if st == S1_APPROACH:
            if el > P.APPROACH_MAX_TIME:
                print("  接近 marker 1 超過 %.0f 秒, 直接右移" % P.APPROACH_MAX_TIME)
                self.enter(S1_PAUSE1, now)
                return (0, 0, 0, 0)
            last = self.last
            p = self.track(markers, P.M1_ID, now)
            if p is None:
                if last is not None and last[2] <= P.APPROACH_DIST + P.NEAR_LOST_EXTRA:
                    print("  靠太近看不到 marker 1 (最後 %.0f cm), 視為抵達" % last[2])
                    self.enter(S1_PAUSE1, now)
                else:
                    self.note("lost1", "看不到 marker 1, 懸停等待")
                return (0, 0, 0, self.heading_cmd())
            x, y, z, _ = p
            if self.arrived(z <= P.APPROACH_DIST + P.FB_DEAD):
                print("  到達 marker 1 前 %.0f cm" % z)
                self.enter(S1_PAUSE1, now)
                return (0, 0, 0, 0)
            fb = self.fwd(z - P.APPROACH_DIST)
            if z < P.SAFE_MIN_DIST:
                fb = min(fb, 0)
            return (self.lat(x), fb, self.vert(y), self.heading_cmd())

        if st == S1_RIGHT:
            p = self.track(markers, P.M2_ID, now)
            done = False
            if P.SIDE_MODE == "marker2":
                if p is not None:
                    self.note("see2", "右移中看到 marker 2, x = %.0f cm" % p[0])
                    done = el >= P.SIDE_MIN_TIME and p[0] <= P.SIDE_ALIGN_CM
                elif el >= P.SIDE_TIME and "see2" not in self.notes:
                    print("  右移 %.1f 秒沒看到 marker 2, 依時間停止" % P.SIDE_TIME)
                    done = True
            else:
                done = el >= P.SIDE_TIME
            if el >= P.SIDE_MAX_TIME:
                print("  右移達上限 %.1f 秒" % P.SIDE_MAX_TIME)
                done = True
            if done:
                self.enter(S1_PAUSE2, now)
                return (0, 0, 0, 0)
            return (P.SIDE_SPEED, 0, 0, self.heading_cmd())

        if st == S1_FORWARD:
            p = self.track(markers, P.M2_ID, now)
            if p is None:
                if el >= P.FWD_MAX_TIME:
                    self.note("nofind2", "往前 %.0f 秒仍沒看到 marker 2, 懸停 (可手動調整後再按 g)" % P.FWD_MAX_TIME)
                    return (0, 0, 0, self.heading_cmd())
                return (0, P.FWD_SPEED, 0, self.heading_cmd())
            x, y, z, _ = p
            err = z - P.M2_STOP_DIST
            if self.arrived(abs(err) <= P.M2_TOL):
                print("  到達 marker 2 前 %.0f cm" % z)
                self.enter(S1_PAUSE3, now)
                return (0, 0, 0, 0)
            # 死區 = 到位範圍, 否則會卡在 "不動但也還沒到位" 的距離
            return (self.lat(x), self.fwd(err, dead=P.M2_TOL), self.vert(y), self.heading_cmd())

        if st == S1_LEFT:
            if el >= P.LEFT_MIN_TIME and self.arrived(P.M0_ID in markers):
                print("  看到手持的 marker 0 (%.0f cm), 進入第二段" % markers[P.M0_ID]["d"])
                self.enter(S2_FOLLOW, now)
                return (0, 0, 0, 0)
            if el >= P.LEFT_MAX_TIME:
                print("  左移 %.0f 秒仍沒看到 marker 0, 原地等待" % P.LEFT_MAX_TIME)
                self.enter(S1_WAIT0, now)
                return (0, 0, 0, 0)
            return (-P.LEFT_SPEED, 0, 0, self.heading_cmd())
        return (0, 0, 0, 0)

    def step_s2(self, st, now, el, markers):
        """第二段: 跟著手持 marker 0"""
        if st == S2_FOLLOW:
            p = self.track(markers, P.M0_ID, now)
            if P.M0_ID in markers:                           # 最後一次在畫面的左右 (原始值): 跑出畫面時往這邊轉
                rx, _, rz = markers[P.M0_ID]["t"]
                self.m0_side = 1 if rx > 0 else -1
                self.m0_side_bearing = math.degrees(math.atan2(rx, rz))
            # 自動進第三段: 之前看過 marker 0, 現在 marker 0 不在畫面但 marker 3 在 -> 停止找 marker 0, 對準 marker 3;
            # 這個狀態持續 S2_TO_S3_TIME 秒就進 S3_ALIGN (marker 0 又出現就回去跟)
            if P.M0_ID in markers:
                self.m0_ever = True
                self.m3_since = None
            if now < self.s2_hold_until:
                # 第一段左移剛看到 marker 0: 先原地懸停一下 (把左移的速度停掉), 再開始跟
                self.note("s2hold", "看到 marker 0, 先懸停 %.1f 秒再開始跟" % P.S2_START_HOLD)
                return (0, 0, 0, 0)
            if P.S2_AUTO_S3 and self.m0_ever and p is None:
                if P.M3_ID in markers:
                    self.m3_last = now
                    if self.m3_since is None:
                        self.m3_since = now
                        self.ema3.reset()
                        print("  marker 0 不見了, 看到 marker 3: 對準 marker 3, %.1f 秒後進第三段" % P.S2_TO_S3_TIME)
                    if now - self.m3_since >= P.S2_TO_S3_TIME:
                        print("  看到 marker 3 %.1f 秒, 進第三段" % (now - self.m3_since))
                        self.enter(S3_ALIGN, now)
                        return (0, 0, 0, 0)
                    m3 = markers[P.M3_ID]
                    p3 = self.ema3.update((m3["t"][0], m3["t"][1], m3["t"][2], m3["ang"]))
                    rc3, _ = self.align(p3, P.S3_DIST, P.S3_DIST_TOL, use_ud=False)
                    self.last_yaw_cmd = 0
                    return rc3
                if self.m3_since is not None and now - self.m3_last < P.S2_TO_S3_M3_GAP:
                    return (0, 0, 0, 0)                    # marker 3 只漏抓一下: 原地等, 不要開始轉
                self.m3_since = None
            if p is not None:
                x, y, z, _ = p
                bearing = math.degrees(math.atan2(x, z))   # marker 0 偏離正前方的角度, 右為正
                # 估計 marker 0 繞著無人機移動的角速度 = 畫面裡角度的變化 + 自己轉掉的角度
                if self.prev_bearing is not None and now - self.prev_bt > 0.5:
                    self.prev_bearing = None               # 中間斷掉太久: 重新開始估
                if self.prev_bearing is not None and now > self.prev_bt:
                    rate = (bearing - self.prev_bearing) / (now - self.prev_bt) + self.last_yaw_cmd * P.SPEED_SCALE_TURN * P.YAW_RC_TO_DEG
                    self.bearing_rate += P.FOLLOW_RATE_ALPHA * (rate - self.bearing_rate)
                else:
                    self.bearing_rate = 0.0
                self.prev_bearing, self.prev_bt = bearing, now
                self.last_bearing = bearing
                # 預測: 往 FOLLOW_LEAD 秒後 marker 0 會在的方向轉
                lead = max(-P.FOLLOW_PRED_MAX, min(P.FOLLOW_PRED_MAX, P.FOLLOW_LEAD * self.bearing_rate))
                target = bearing + lead
                # 轉向 + 左右平移一起把 marker 0 拉回畫面中心 (只靠轉向會跟不上)
                yaw = p_ctrl(target, P.FOLLOW_YAW_KP, P.FOLLOW_YAW_DEAD, P.FOLLOW_YAW_MAX, P.FOLLOW_YAW_MIN)
                self.last_yaw_cmd = yaw
                lr = p_ctrl(x, P.FOLLOW_LAT_KP, P.LAT_DEAD, P.FOLLOW_LAT_MAX)
                fb = p_ctrl(z - P.FOLLOW_DIST, P.FOLLOW_FB_KP, P.FOLLOW_DEAD, P.FOLLOW_FB_MAX, P.FB_MIN)
                # marker 0 偏離中心時先轉、少往前: 偏 FOLLOW_TURN_FIRST_DEG 度以上時前進只剩 30%
                # (實飛: 往前 58 + 轉向 -20 時看起來只有在往前, 沒在轉)
                if fb > 0 and P.FOLLOW_TURN_FIRST_DEG:
                    fb = int(fb * max(0.3, 1.0 - abs(bearing) / P.FOLLOW_TURN_FIRST_DEG))
                return (lr, fb, self.vert(y), yaw)
            # 只漏抓一下 (晃動 / 模糊 / Wi-Fi): 不要馬上全速轉去找 (實飛 log: 漏抓 1 張就以 55 轉, 轉太快畫面模糊更抓不到),
            # 先照剛剛的轉向繼續轉, 停止前後左右
            if self.last_bearing is not None and now - self.lost_since < P.FOLLOW_LOST_HOLD:
                return (0, 0, 0, self.last_yaw_cmd)
            # 找不到 marker 0: 一直往左轉找, 直到找到 (最多 FOLLOW_SEARCH_LEFT_MAX 秒, 約轉一圈)
            if P.FOLLOW_SEARCH_DIR == "left":
                if not self.m0_ever:
                    # 剛從第一段 (看到 m2 後左移) 進來, 這一段還沒看到過 marker 0: 不要自轉, 原地等它出現
                    self.last_yaw_cmd = 0
                    self.note("wait0first", "還沒看到 marker 0, 原地等 (看到一次之後不見才會往左轉找)")
                    return (0, 0, 0, 0)
                if self.lost_since is not None and now - self.lost_since < P.FOLLOW_LOST_HOLD + P.FOLLOW_SEARCH_LEFT_MAX:
                    self.note("search0", "找不到 marker 0, 往左轉找")
                    self.last_yaw_cmd = -P.FOLLOW_SEARCH_YAW
                    return (0, 0, 0, self.last_yaw_cmd)
                self.last_yaw_cmd = 0
                self.note("lost0", "往左轉 %.0f 秒還是找不到 marker 0, 懸停等待" % P.FOLLOW_SEARCH_LEFT_MAX)
                return (0, 0, 0, 0)
            # (FOLLOW_SEARCH_DIR = "last") marker 0 剛跑出畫面: 往它最後出現的那一側轉一下找回來
            if self.last_bearing is not None and now - self.lost_since < P.FOLLOW_LOST_HOLD + P.FOLLOW_SEARCH_TIME:
                side = getattr(self, "m0_side", 1 if self.last_bearing > 0 else -1)
                if P.FOLLOW_LEAD:                            # (有開速度預測時才用)
                    side = 1 if self.last_bearing + P.FOLLOW_LEAD * self.bearing_rate > 0 else -1
                self.note("search0", "marker 0 跑出畫面 (最後在畫面%s %.0f 度), 往%s轉找回來" % (
                    "右" if side > 0 else "左", abs(getattr(self, "m0_side_bearing", self.last_bearing)),
                    "右" if side > 0 else "左"))
                self.last_yaw_cmd = P.FOLLOW_SEARCH_YAW * side
                return (0, 0, 0, self.last_yaw_cmd)
            self.last_yaw_cmd = 0
            self.note("lost0", "看不到 marker 0, 懸停等待")
            return (0, 0, 0, 0)
        return (0, 0, 0, 0)

    def step_s3(self, st, now, el, markers):
        """第三段: 對正 marker 3 -> 下降 -> 鑽桌子 -> 轉 90 度 -> 上升對正 marker 4"""
        if st == S3_ALIGN:
            # 還在對正 marker 3 時 marker 0 又出現: 人還沒走完, 回去跟 (之後要重新看到 marker 0 + 3 才會再自動切)
            if P.S3_BACK_TO_S2:
                self.m0_back = self.m0_back + 1 if P.M0_ID in markers else 0
                if self.m0_back >= P.S3_BACK_FRAMES:
                    print("  對正 marker 3 時又看到 marker 0, 回到第二段跟隨 (marker 0 不見 + 看到 marker 3 %.1f 秒就會再進第三段)"
                          % P.S2_TO_S3_TIME)
                    self.enter(S2_FOLLOW, now)
                    self.m0_ever = True                        # 剛剛有看到 marker 0
                    return (0, 0, 0, 0)
            p = self.track(markers, P.M3_ID, now)
            if p is None:
                return self.seek(P.M3_ID, now)
            rc, ok = self.align(p, P.S3_DIST, P.S3_DIST_TOL, use_ud=False)
            # 角度: 平均幾張再用, 而且要 < S3_YAW_TOL (原本 ±6 度就算正; 鑽桌子要往前飛 ~300 cm, 歪 5 度就偏 26 cm)
            if P.M3_ID in markers:
                self.ang3 = float(self.ema3a.update([markers[P.M3_ID]["ang"]])[0])
            ang3 = self.ang3 if self.ang3 is not None else p[3]
            yaw = (p_ctrl(P.SIGN_ALIGN_YAW * ang3, P.ALIGN_YAW_KP, P.S3_YAW_DEAD, P.ALIGN_YAW_MAX, P.ALIGN_YAW_MIN)
                   if P.ALIGN_YAW else 0)
            yaw_ok = not P.ALIGN_YAW or abs(ang3) < P.S3_YAW_TOL
            face_ok = yaw_ok and abs(p[0]) < P.S3_LAT_TOL           # 正對: 左右置中 + 角度正
            dt = 0.0 if self.lat_t is None else min(0.2, now - self.lat_t)
            self.lat_t = now
            if self.lat_i * p[0] < 0 and abs(p[0]) > P.LAT_DEAD:
                self.lat_i = 0.0                           # 已經偏到另一邊: 積分歸零 (實飛: 積分 +12 一直推, 推過頭往右偏)
            if P.ALIGN_YAW and abs(ang3) >= P.S3_YAW_TOL + 2:
                # 角度差很多: 先轉正, 不修左右 (邊轉邊修, 每轉一點 marker 在畫面的左右就跳, 積分會越推越歪); 積分先凍結
                rc = (0, rc[1], 0, yaw)
            else:
                if abs(p[0]) > P.LAT_DEAD:
                    self.lat_i = max(-P.S3_LAT_I_MAX, min(P.S3_LAT_I_MAX, self.lat_i + P.S3_LAT_KI * p[0] * dt))
                rc = (limit(rc[0] + self.lat_i, P.LAT_MAX + P.S3_LAT_I_MAX), rc[1], 0, yaw)
            ok = abs(p[2] - P.S3_DIST) <= P.S3_DIST_TOL and face_ok
            if now - getattr(self, "_s3_dbg", 0.0) >= 1.0:   # 每秒印一次, 看 log 才知道角度有沒有在收斂
                self._s3_dbg = now
                print("  [S3_ALIGN] x %+.0f  z %.0f  角度 %+.0f (平均 %+.1f)  rc %s  (左右積分 %+.0f)" % (p[0], p[2], p[3], ang3, rc, self.lat_i))
            if not self.arrived(ok) and el > P.ALIGN_MAX_TIME:
                if not face_ok:
                    # 還沒正對: 不可以往前 (會斜斜 / 偏一邊鑽過去), 距離先不管, 只修左右和角度;
                    # 再修 S3_YAW_EXTRA_TIME 秒還不行就懸停
                    if el > P.ALIGN_MAX_TIME + P.S3_YAW_EXTRA_TIME:
                        self.note("s3yaw", "對正 marker 3 一直對不正 (左右 %+.0f cm, 角度 %+.0f), 懸停: 請手動對正後再按 j"
                                  % (p[0], p[3]))
                        return (0, 0, 0, 0)
                    self.note("s3yawonly", "對正 marker 3 超過 %.0f 秒, 距離先不管, 只修左右和角度 (左右 %+.0f cm, 角度 %+.0f)"
                              % (P.ALIGN_MAX_TIME, p[0], p[3]))
                    lr = p_ctrl(p[0], P.LAT_KP, P.LAT_DEAD, P.LAT_MAX, P.S3_LAT_MIN) + self.lat_i
                    return (limit(lr, P.LAT_MAX + P.S3_LAT_I_MAX), 0, 0, rc[3])
                print("  對正 marker 3 超過 %.0f 秒, 已正對 (左右 %+.0f cm, 角度 %+.0f), 距離 %.0f cm 就從這裡繼續"
                      % (P.ALIGN_MAX_TIME, p[0], p[3], p[2]))
                ok = True
            if ok and (self.arrive >= P.ARRIVE_FRAMES or el > P.ALIGN_MAX_TIME):
                print("  已正對 marker 3 (%.0f cm, 左右 %+.0f cm, 角度 %+.1f (平均)), 開始下降" % (p[2], p[0], ang3))
                self.set_heading_ref()                     # 鑽桌子的方向 = 正對 marker 3 的方向
                self.enter(S3_DOWN, now)
                return (0, 0, 0, 0)
            return rc

        if st == S3_DOWN:
            last = self.last
            p = self.track(markers, P.M3_ID, now)
            alt = read_tof(self.drone)
            if p is not None:
                x, y, z, _ = p
                clear = self.clear3(y, z)                  # marker 下緣比鏡頭高多少 (cm)
                if self.arrived(clear >= P.S3_CLEAR and self.low_enough(alt)):
                    print("  marker 3 下緣高於鏡頭 %.0f cm (高度 %s), 開始鑽桌子" %
                          (clear, "%.0f cm" % alt if alt is not None else "?"))
                    self.s3_z0 = z
                    self.enter(S3_UNDER, now)
                    return (0, 0, 0, 0)
                lr = self.lat(x)
                fb = self.fwd(z - P.S3_DIST, dead=P.S3_DIST_TOL)
            else:
                # 下降時 marker 3 從畫面上緣跑掉: 它已經在上方, 再多降一點保險後就鑽
                if last is not None and self.clear3(last[1], last[2]) >= 0:
                    if alt is not None and P.S3_PASS_ALT is not None:
                        if alt <= P.S3_PASS_ALT:             # 有 TOF: 降到 S3_PASS_ALT 以下才鑽
                            print("  marker 3 已移出畫面上方, 高度 %.0f cm, 開始鑽桌子" % alt)
                            self.s3_z0 = last[2]
                            self.enter(S3_UNDER, now)
                            return (0, 0, 0, 0)
                    elif now - self.lost_since >= P.S3_LOST_EXTRA:
                        print("  marker 3 已移出畫面上方, 多降 %.1f 秒後開始鑽桌子" % P.S3_LOST_EXTRA)
                        self.s3_z0 = last[2]
                        self.enter(S3_UNDER, now)
                        return (0, 0, 0, 0)
                else:
                    # 偶爾一兩張 frame 沒抓到 (晃動 / Wi-Fi): 不要停下來懸停 (實飛會一直停停走走直到超時),
                    # 只停止前後左右修正, 繼續往下; 最低高度 MIN_ALT 仍會擋住
                    self.note("lost3d", "下降時暫時看不到 marker 3, 繼續往下")
                lr = fb = 0
            if alt is not None and alt <= P.MIN_ALT:
                self.note("minalt", "已到最低高度 %.0f cm, marker 3 仍不夠高, 不往前鑽 (請手動處理)" % alt)
                return (0, 0, 0, self.heading_cmd())
            if el > P.S3_DOWN_MAX_TIME:
                self.note("downmax", "下降超過 %.0f 秒仍不夠低, 懸停" % P.S3_DOWN_MAX_TIME)
                return (0, 0, 0, self.heading_cmd())
            return (lr, fb, -P.S3_DOWN_SPEED, self.heading_cmd())

        if st == S3_UNDER:
            if el >= self.under_time:
                print("  已鑽過桌子")
                self.enter(S3_PAUSE1, now)
                return (0, 0, 0, 0)
            # 還看得到 marker 3 時再檢查一次: 下緣不夠高就先停下來往下 (絕不往前撞)
            p = self.track(markers, P.M3_ID, now)
            if p is not None and self.clear3(p[1], p[2]) < P.S3_CLEAR * 0.5:
                self.note("lowclear", "鑽桌子時 marker 3 太低, 先停下來下降")
                self.t0 += P.CONTROL_PERIOD                  # 這段時間不算在往前的秒數裡
                return (0, 0, -P.S3_DOWN_SPEED, self.heading_cmd())
            ud = self.alt_hold(self.hold_alt)
            alt = read_alt(self.drone)
            if ud > 0 and not (alt is not None and alt < P.MIN_ALT):
                ud = 0                                     # 桌子底下不准往上修 (除非快貼地)
            return (0, P.S3_UNDER_SPEED, ud, self.heading_cmd())

        if st == S3_TURN:
            err = self.heading_err(self.turn_target)
            if err is None:                                # 讀不到 yaw: 用內建指令轉
                print("  讀不到 yaw, 改用 rotate_clockwise(%d)" % P.S3_TURN_DEG)
                try:
                    self.drone.rotate_clockwise(int(P.S3_TURN_DEG))
                except Exception as e:
                    print("  rotate failed:", e)
                self.enter(S3_UP, now)
                return (0, 0, 0, 0)
            if self.arrived(abs(err) <= P.TURN_TOL) or el > P.TURN_MAX_TIME:
                print("  已轉 %.0f 度 (誤差 %.0f)" % (P.S3_TURN_DEG, err))
                self.set_heading_ref()
                self.enter(S3_UP, now)
                return (0, 0, 0, 0)
            return (0, 0, 0, p_ctrl(-P.HEADING_SIGN * err, P.HEADING_KP, P.TURN_TOL, P.TURN_MAX, P.HEADING_MIN))

        if st == S3_UP:
            if self.arrived(P.M4_ID in markers):
                print("  看到 marker 4")
                self.enter(S3_ALIGN4, now)
                return (0, 0, 0, 0)
            alt = read_tof(self.drone)
            high = P.S3_UP_ALT is not None and alt is not None and alt >= P.S3_UP_ALT
            if high or el >= P.S3_UP_MAX_TIME:
                print("  往上到 %s 仍沒看到 marker 4, 往後退 + 左右找" % ("%.0f cm" % alt if alt is not None else "%.0f 秒" % el))
                self.enter(S3_FIND4, now)
                return (0, 0, 0, 0)
            return (0, 0, P.S3_UP_SPEED, self.heading_cmd())

        if st == S3_FIND4:
            # 不靠鑽了多遠: 停在 marker 4 的高度, 先後退讓視野變寬, 再往左 / 往右平移找 marker 4
            if self.arrived(P.M4_ID in markers):
                print("  找到 marker 4 (找了 %.1f 秒)" % el)
                self.enter(S3_ALIGN4, now)
                return (0, 0, 0, 0)
            ud = self.alt_hold(self.alt0)                  # 維持剛升到的高度
            t1 = P.S3_FIND4_BACK
            t2 = t1 + P.S3_FIND4_TIME
            t3 = t2 + 2 * P.S3_FIND4_TIME
            if el < t1:
                return (0, -P.S3_FIND4_SPEED, ud, self.heading_cmd())
            if el < t2:
                self.note("find4l", "往左找 marker 4")
                return (-P.S3_FIND4_SPEED, 0, ud, self.heading_cmd())
            if el < t3:
                self.note("find4r", "往右找 marker 4")
                return (P.S3_FIND4_SPEED, 0, ud, self.heading_cmd())
            print("  左右都找不到 marker 4, 原地等待 (可手動移到看得到 marker 4 後按 k)")
            self.enter(S3_WAIT4, now)
            return (0, 0, 0, 0)

        if st == S3_ALIGN4:
            p = self.track(markers, P.M4_ID, now)
            if p is None:                                  # 剛升上來時 marker 4 常只在畫面邊緣
                if self.last is None and el > P.S4_FIND_M4_TIME:
                    # 按 k 進來但一直沒看到 marker 4: 照舊以目前方向進第四段
                    print("  %.0f 秒沒看到 marker 4, 以目前方向進第四段" % P.S4_FIND_M4_TIME)
                    self.set_heading_ref()
                    self.enter(S4_CLIMB, now)
                    return (0, 0, 0, 0)
                return self.seek(P.M4_ID, now)
            # 對正 marker 4: 前後 / 上下用比較快的速度, 容許誤差放寬一點 (實飛常要 5~8 秒: 剛找到時離牆太近要退很多,
            # 高度只用 8~11 的速度慢慢爬, 而且要對到 ±7 cm)
            x, y, z, ang = p
            yaw = (p_ctrl(P.SIGN_ALIGN_YAW * ang, P.ALIGN_YAW_KP, P.ALIGN_YAW_DEAD, P.ALIGN_YAW_MAX, P.ALIGN_YAW_MIN)
                   if P.ALIGN_YAW else 0)
            rc = (self.lat(x),
                  p_ctrl(z - P.M4_DIST, P.FB_KP, P.S4A_DIST_TOL / 2, P.S4A_FB_MAX, P.FB_MIN),
                  p_ctrl(-y, P.S4A_UD_KP, P.ALIGN_UD_DEAD, P.S4A_UD_MAX, P.ALIGN_UD_MIN),
                  yaw)
            yaw_ok = not P.ALIGN_YAW or abs(ang) < P.ALIGN_YAW_DEAD + 1
            ok = (abs(x) < P.S4A_LAT_TOL and abs(z - P.M4_DIST) <= P.S4A_DIST_TOL and abs(y) < P.S4A_UD_TOL and yaw_ok)
            if now - getattr(self, "_s4_dbg", 0.0) >= 1.0:
                self._s4_dbg = now
                print("  [S3_ALIGN4] x %+.0f  y %+.0f  z %.0f  角度 %+.0f  rc %s" % (p[0], p[1], p[2], p[3], rc))
            if not self.arrived(ok) and el > P.ALIGN_MAX_TIME:
                if not yaw_ok:
                    # 沒正對 marker 4 就往左, 第四段會斜著飛, 第五段降落就不準: 先只轉正
                    if el > P.ALIGN_MAX_TIME + P.S3_YAW_EXTRA_TIME:
                        self.note("s4yaw", "對正 marker 4 一直轉不正 (角度 %+.0f), 懸停: 請手動轉正後再按 k" % p[3])
                        return (0, 0, 0, 0)
                    self.note("s4yawonly", "對正 marker 4 超過 %.0f 秒, 位置先不管, 只把角度轉正 (現在 %+.0f 度)"
                              % (P.ALIGN_MAX_TIME, p[3]))
                    return (rc[0], 0, rc[2], rc[3])
                ok = True
            if ok and (self.arrive >= P.ARRIVE_FRAMES or el > P.ALIGN_MAX_TIME):
                print("  第三段完成: 已對正 marker 4 (%.0f cm, 角度 %+.0f)%s" %
                      (p[2], p[3], "" if self.arrive >= P.ARRIVE_FRAMES else " (超時, 角度已正, 以目前位置繼續)"))
                self.set_heading_ref()                     # 沿牆平移的方向 = 正對 marker 4 的方向
                self.enter(S4_CLIMB, now)
                return (0, 0, 0, 0)
            return rc
        return (0, 0, 0, 0)

    def step_s4(self, st, now, el, markers):
        """第四段: 維持高度向左越過板子, 直到看到 marker 5"""
        if st == S4_CLIMB:
            if P.S4_CLIMB_CM <= 0:                           # 不用爬升: 直接以目前 (marker 4 的) 高度向左
                self.enter(S4_LEFT, now)
                self.alt_left = self.alt0
                return (0, 0, 0, 0)
            alt = read_alt(self.drone)
            climbed = None if (alt is None or self.alt0 is None) else alt - self.alt0
            if (climbed is not None and climbed >= P.S4_CLIMB_CM) or el >= P.S4_CLIMB_MAX_TIME:
                print("  爬升完成 (%s)" % ("+%.0f cm" % climbed if climbed is not None else "依時間"))
                base = self.alt0
                self.enter(S4_LEFT, now)
                self.alt_left = self.alt0                  # 向左時要維持的高度 (爬升後)
                self.alt0 = base                           # 記住爬升前的高度, 過板子後降回來
                return (0, 0, 0, 0)
            return (0, 0, P.S4_CLIMB_SPEED, self.heading_cmd())

        if st == S4_LEFT:
            if self.arrived(P.M5_ID in markers):
                print("  第四段完成: 看到 marker 5")
                self.enter(S5_ALIGN, now)
                return (0, 0, 0, 0)
            if el >= P.S4_MAX_TIME:
                print("  向左 %.0f 秒仍沒看到 marker 5, 原地等待" % P.S4_MAX_TIME)
                self.enter(S4_WAIT5, now)
                return (0, 0, 0, 0)
            if P.S4_CLIMB_CM > 0 and P.S4_DESCEND_AFTER and el >= P.S4_CROSS_TIME:
                ud = min(self.alt_hold(self.alt0), 0)      # 已越過板子: 降回爬升前的高度
            else:
                ud = self.alt_hold(self.alt_left)          # 維持高度, 不要往下掉到板子
            return (-P.S4_LEFT_SPEED, 0, ud, self.heading_cmd())
        return (0, 0, 0, 0)

    def step_s5(self, st, now, el, markers):
        """第五段: 對準 marker 5 -> (確認桌子) -> 降落"""
        if st == S5_TABLE:
            # 用 TOF 確認 / 找桌子: 下面是桌子時 TOF 會比地板小一個桌子的高度
            p = self.track(markers, P.M5_ID, now)
            tof = read_tof(self.drone)
            if p is None:
                if self.lost_since is not None and now - self.lost_since > P.S5_TABLE_LOST:
                    print("  找桌子時 %.0f 秒看不到 marker 5, 直接降落" % P.S5_TABLE_LOST)
                    self.enter(FINISHED, now)
                return (0, 0, 0, 0)
            moved = self.tbl_x0 - p[0]                     # 無人機往右移了多少 (cm): 往右 -> marker 在畫面往左
            fb = self.fwd(p[2] - P.M5_DIST, dead=P.M5_TOL)
            on = tof is not None and self.tof_floor is not None and tof < self.tof_floor - P.S5_TABLE_DROP
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
                    print("  找到桌子邊緣 (往%s %.0f cm), 再往裡面 %.0f cm" % ("右" if moved > 0 else "左", abs(moved), P.S5_TABLE_IN_CM))
                elif self.tof_floor is None or tof is None:
                    print("  讀不到 TOF, 無法確認桌子, 直接降落")
                    self.enter(FINISHED, now)
                    return (0, 0, 0, 0)
                elif self.tbl_on > 0:
                    return (0, fb, 0, 0)                   # TOF 剛變小: 先停下來確認是不是桌子
                else:
                    if self.tbl_dir * moved >= P.S5_SCAN_CM:
                        if self.tbl_dir == P.S5_SCAN_FIRST:
                            self.tbl_dir = -self.tbl_dir
                            print("  往%s %.0f cm 沒找到桌子, 換邊找" % ("右" if moved > 0 else "左", P.S5_SCAN_CM))
                        else:
                            print("  左右都沒找到桌子 (TOF 沒有變小), 回到對準 marker 5 的位置降落")
                            self.tbl_found = 0.0
                            self.tbl_dir = 0
                    self.tbl_scanned = True
                    self.note("s5scan", "下面不是桌子 (tof %s, 地板 %.0f), 往%s找桌子" % (
                        "%.0f" % tof, self.tof_floor, "右" if self.tbl_dir > 0 else "左"))
                    return (self.tbl_dir * P.S5_SCAN_SPEED, fb, 0, 0)
            target = self.tbl_found + self.tbl_dir * P.S5_TABLE_IN_CM
            err = target - moved
            if abs(err) < 4:
                if self.tbl_ok_t is None:
                    self.tbl_ok_t = now
                if now - self.tbl_ok_t >= P.S5_SETTLE:
                    print("  在桌子上方 (tof %s), 降落" % ("%.0f" % tof if tof is not None else "?"))
                    self.enter(FINISHED, now)
                return (0, 0, 0, 0)
            self.tbl_ok_t = None
            return (p_ctrl(err, P.S5_LAT_KP, 2, P.S5_SCAN_SPEED, 8), fb, 0, 0)

        if st == S5_ALIGN:
            p = self.track(markers, P.M5_ID, now)
            if p is None:                                  # 剛移過來時 marker 5 常只在畫面邊緣
                self.s5_n = 0
                return self.seek(P.M5_ID, now)
            # 左右要對得比其他段準 (桌子只有 50 cm 寬): 死區 / 容許誤差都比較小, 而且有最小速度推得動
            # 機頭要先正對牆, 畫面裡的左右才等於沿牆的左右: 機頭歪 7 度時畫面置中, 170 cm 外沿牆會差約 20 cm
            # (實飛就是這樣落在桌子左邊)。單張的角度雜訊有 ±5 度, 所以角度先平均很多張再轉, 死區也收小
            if P.M5_ID in markers:
                self.ang5 = float(self.ema5.update([markers[P.M5_ID]["ang"]])[0])
            ang5 = self.ang5 if self.ang5 is not None else p[3]
            ex = p[0] + P.S5_LAT_TRIM                        # marker 在畫面右邊 (x>0) -> 往右; 加 TRIM 讓降落點往右偏
            zz = p[2]
            yaw = (p_ctrl(P.SIGN_ALIGN_YAW * ang5, P.ALIGN_YAW_KP, P.S5_YAW_DEAD, P.ALIGN_YAW_MAX, P.S5_YAW_MIN)
                   if P.ALIGN_YAW else 0)
            if P.S5_SIMPLE:
                # 簡單版: 連續 S5_ARRIVE_FRAMES 張都看到 marker 5、而且在畫面中心 + 距離對 -> 降落
                lr = p_ctrl(ex, P.S5_LAT_KP, P.S5_LAT_DEAD, P.S5_LAT_MAX, P.S5_LAT_MIN)
                fb = self.fwd(zz - P.M5_DIST, dead=P.M5_TOL)
                ok = abs(ex) < P.S5_LAT_TOL and abs(zz - P.M5_DIST) <= P.M5_TOL
                self.s5_n = self.s5_n + 1 if ok else 0
                if self.s5_n >= P.S5_ARRIVE_FRAMES:
                    print("  連續 %d 張 marker 5 在中心 (離 %.0f cm, 左右偏 %+.0f cm, 角度 %+.1f), 降落"
                          % (self.s5_n, zz, p[0], ang5))
                    self.enter(FINISHED, now)
                    return (0, 0, 0, 0)
                return (lr, fb, 0, yaw)
            # 剛看到 marker 5 時還帶著第四段向左的速度: 先只煞車懸停 S5_SETTLE 秒, 不要邊滑邊對
            if el < P.S5_SETTLE:
                self.note("s5settle", "看到 marker 5, 先煞車懸停 %.1f 秒" % P.S5_SETTLE)
                return (0, 0, 0, 0)
            lr = p_ctrl(ex, P.S5_LAT_KP, P.S5_LAT_DEAD, P.S5_LAT_MAX, P.S5_LAT_MIN)
            fb = self.fwd(zz - P.M5_DIST, dead=P.M5_TOL)
            if P.ALIGN_YAW and abs(ang5) >= P.S5_YAW_TOL:
                # 先轉正再對左右: 邊轉邊對, 每轉一點畫面裡的左右就跳, 會一直左右來回
                self.s5_hist = []
                self.note("s5yaw", "先轉正 (角度 %+.1f)" % ang5)
                return (0, fb, 0, yaw)
            if abs(ex) < P.S5_LAT_AIM:
                lr = 0                                     # 已經夠準: 不要再推 (最小速度會推過頭), 讓它停下來
            ok = (abs(ex) < P.S5_LAT_TOL and abs(zz - P.M5_DIST) <= P.M5_TOL
                  and (not P.ALIGN_YAW or abs(ang5) < P.S5_YAW_TOL))
            rough = abs(ex) < 1.5 * P.S5_LAT_TOL and abs(zz - P.M5_DIST) <= 1.5 * P.M5_TOL
            # 只在 "到位 (或超時後大致到位) 而且真的停住了" 才降落: 位置在 S5_STILL_TIME 秒內晃動不超過 S5_STILL_CM
            if ok or (el > P.S5_MAX_TIME and rough):
                self.s5_hist.append((now, ex, zz))
                while self.s5_hist and now - self.s5_hist[0][0] > P.S5_STILL_TIME:
                    self.s5_hist.pop(0)
                span = now - self.s5_hist[0][0]
                xs = [h[1] for h in self.s5_hist]
                zs = [h[2] for h in self.s5_hist]
                still = max(xs) - min(xs) <= P.S5_STILL_CM and max(zs) - min(zs) <= 2 * P.S5_STILL_CM
                if span >= P.S5_STILL_TIME * 0.9 and still:
                    print("  對準 marker 5 且停穩 (離 marker 5 %.0f cm, 左右偏 %+.0f cm, 角度 %+.1f (平均))%s" %
                          (zz, p[0], ang5, "" if ok else " (超時, 大致到位)"))
                    if P.S5_TABLE_CHECK:
                        self.tbl_x0, self.tbl_dir, self.tbl_found, self.tbl_on, self.tbl_ok_t = p[0], P.S5_SCAN_FIRST, None, 0, None
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
