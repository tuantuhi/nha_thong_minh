
import random
import math
import sqlite3
import csv
import os
import sys
import argparse
import uuid
import json
import hashlib
import time
from datetime import datetime, timedelta
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database import Database

# ─────────────────────────────────────────────────────────────────
#  HẰNG SỐ PRESENCE TYPE
# ─────────────────────────────────────────────────────────────────
PT_EMPTY          = "empty"
PT_SINGLE_KNOWN   = "single_known"
PT_SINGLE_UNKNOWN = "single_unknown"
PT_MULTI_USER     = "multi_user"
PT_ANIMAL         = "animal"

# ─────────────────────────────────────────────────────────────────
#  ĐỒNG BỘ KHUÔN MẶT VÀO DATABASE
# ─────────────────────────────────────────────────────────────────
def auto_sync_faces_to_db(db_instance):
    # [FIX BUG-PATH] Fallback linh hoạt nếu config.py không khai báo FACES_DIR
    faces_dir = getattr(config, 'FACES_DIR', os.path.join(os.path.dirname(config.DATABASE_PATH), 'faces') if hasattr(config, 'DATABASE_PATH') else 'faces')
    encodings_path = os.path.join(faces_dir, "encodings.json")
    
    if os.path.exists(encodings_path):
        try:
            with open(encodings_path, "r", encoding="utf-8") as f:
                face_data = json.load(f)
                for user_id in face_data.keys():
                    display_name = user_id.replace("_", " ").title()
                    db_instance.upsert_face_profile(user_id, display_name)
        except Exception as e:
            print(f"  [!] Lỗi tự động đồng bộ: {e}")

# ─────────────────────────────────────────────────────────────────
#  CẤU TRÚC DỮ LIỆU ĐẦU VÀO
# ─────────────────────────────────────────────────────────────────
@dataclass
class EnvSample:
    timestamp:        str
    hour:             int
    minute:           int
    day_of_week:      int
    temperature:      float
    humidity:         float
    light_level:      float
    motion_detected:  int
    user_id:          str
    light_state:      int
    fan_state:        int
    session_id:       str
    presence_type:    str
    face_count:       int
    scenario_type:    str = ""
    data_source:      str = "simulated"
    override_flag:    int = 0
    override_reason:  str = ""

# ─────────────────────────────────────────────────────────────────
#  TRÍCH XUẤT NGƯỜI DÙNG TỪ DATABASE
# ─────────────────────────────────────────────────────────────────
# [FIX BUG-F1/C6] Bắt buộc sử dụng db_instance thay vì tự connect
def get_real_users_from_db(db_instance):
    users = []
    try:
        with db_instance.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT user_id, display_name FROM face_profiles"
            ).fetchall()
            for r in rows:
                user_id  = r["user_id"]
                hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                u_type   = 'a' if hash_val % 2 == 0 else 'b'

                random.seed(hash_val)
                base_temp = random.uniform(26.5, 28.0)
                base_lux  = random.uniform(100, 150)
                random.seed()

                users.append({
                    "user_id":        user_id,
                    "type":           u_type,
                    "base_temp_pref": base_temp,
                    "base_lux_pref":  base_lux,
                })

        if not users:
            print("  [!] Không tìm thấy user nào, dùng user giả định A và B...")
            return _default_users()
        return users

    except Exception as e:
        print(f"  [!] Lỗi đọc Database ({e}), dùng user mặc định A và B.")
        return _default_users()


def _default_users():
    return [
        {"user_id": "user_a", "type": "a", "base_temp_pref": 27.5, "base_lux_pref": 120},
        {"user_id": "user_b", "type": "b", "base_temp_pref": 26.5, "base_lux_pref": 150},
    ]


def _ensure_min_users(users: list) -> list:
    if len(users) >= 2:
        return users

    real   = users[0]
    s_type = 'b' if real['type'] == 'a' else 'a'

    random.seed(42)
    s_temp = round(real['base_temp_pref'] - random.uniform(0.5, 1.5), 1)
    s_lux  = round(real['base_lux_pref']  + random.uniform(20, 50),   1)
    random.seed()

    synth = {
        "user_id":        f"synth_{real['user_id']}",
        "type":           s_type,
        "base_temp_pref": s_temp,
        "base_lux_pref":  s_lux,
        "is_synthetic":   True,
    }
    print(
        f"  [FIX-7] DB chỉ có 1 user thực → thêm synthetic co-user "
        f"'{synth['user_id']}' (temp_pref={s_temp}°C) cho kịch bản multi_user."
    )
    return users + [synth]


# ─────────────────────────────────────────────────────────────────
#  CỐT LÕI GENERATOR
# ─────────────────────────────────────────────────────────────────
class SmartHomeDataGenerator:
    def __init__(self, start_date: datetime, db_instance: Database, is_realtime: bool = False):
        self.start_date  = start_date
        self.is_realtime = is_realtime
        
        # [FIX BUG-F1/C6] Lấy user từ DB dùng chung
        self.users       = _ensure_min_users(get_real_users_from_db(db_instance))
        self.t_threshold = config.ENV_THRESHOLDS["temp_high"]
        self.l_threshold = config.ENV_THRESHOLDS["light_dark"]
        self.bright_threshold = config.ENV_THRESHOLDS.get("light_bright", self.l_threshold + 150)

        # ── Trạng thái session chính ─────────────────────────────
        self.current_session_id   = None
        self.current_user         = "unknown"
        self.session_time_left    = 0
        self.empty_room_time_left = 0

        self.post_session_gap: int = 0
        self.post_animal_gap: int = 0
        self._is_sleep_session: bool = False
        self._fan_hys: int = 0
        self.animal_event_left: int = 0

        # ── Trạng thái đa người dùng (co-users) ─────────
        self.co_users: list[str] = []
        self.co_user_time_left: dict[str, int] = {}

        # ── Thời tiết & Nhiệt độ (AR-1) ────────────────────────
        self._weather_cloudy    = False
        self._weather_day_stamp = None
        self._daily_temp_offset: float = 0.0
        self._temp_day_stamp            = None

        # ── Tích lũy nhiệt ban ngày ─────────────────────
        self._heat_accumulation: float = 0.0
        self._last_heat_hour: int      = -1

        # ── Kịch bản mô phỏng phục vụ báo cáo ─────────────
        self._scenario_cycle = [
            "sequence_behavior",
            "context_exception",
            "manual_override",
            "background_noise",
        ]
        self._user_profiles = getattr(config, "USER_HYSTERESIS_PROFILES", {}) or {}
        self._fan_on_temp_default = getattr(config, "FAN_HYSTERESIS_MIN_ON_TEMP", self.t_threshold)
        self._fan_off_temp_default = getattr(config, "FAN_HYSTERESIS_MIN_OFF_TEMP", self.t_threshold - 1.0)
        self._scenario_day_cache = {}
        self._override_active = False
        self._override_remaining = 0
        self._override_target = None
        self._override_reason = ""
        self._last_prediction = {"light_state": None, "fan_state": None}
        self._sensor_delay_queue: list[dict] = []
        # Các tham số này được tăng nhẹ để dữ liệu mô phỏng sát thực hơn
        # và làm giảm xu hướng false positive cho `light`.
        self._sensor_delay_prob = 0.14
        self._sensor_delay_max = 5
        self._sensor_noise_prob = 0.18
        self._sensor_noise_strength = 0.24
        self._abnormal_behavior_prob = 0.075
        self._near_miss_prob = 0.18
        self._light_conservative_bias = 0.82
        self._light_false_positive_guard = 0.68
        self._light_threshold_floor = 14.0

    def _refresh_weather_if_new_day(self, current_dt: datetime):
        today = current_dt.date()
        if self._weather_day_stamp != today:
            if self._weather_day_stamp is None:
                self._weather_cloudy = random.random() < 0.30
            else:
                p_cloud = 0.55 if self._weather_cloudy else 0.20
                self._weather_cloudy = random.random() < p_cloud
            self._weather_day_stamp = today

    def _refresh_temp_offset_if_new_day(self, current_dt: datetime):
        today = current_dt.date()
        if self._temp_day_stamp != today:
            noise = random.gauss(0, 0.8)
            self._daily_temp_offset = max(-3.0, min(3.0,
                0.7 * self._daily_temp_offset + noise
            ))
            self._temp_day_stamp = today

    def _update_heat_accumulation(self, h: int, fan_was_on: bool):
        if h == 6:
            self._heat_accumulation = 0.0
            self._last_heat_hour = h
            return

        if h >= 13 and h != self._last_heat_hour:
            if not fan_was_on:
                self._heat_accumulation = min(2.0, self._heat_accumulation + 0.1)
            else:
                self._heat_accumulation = max(0.0, self._heat_accumulation - 0.05)
            self._last_heat_hour = h

    def _is_sleeping_hour(self, hour: int, dow: int = -1) -> bool:
        if hour <= 5 or hour >= 23:
            return True
        if dow in (5, 6) and 6 <= hour <= 8:
            return True
        return False

    def _get_active_user(self, hour: int, dow: int) -> str:
        if not self.users:
            return "unknown"

        is_weekend      = (dow >= 5)
        is_friday_night = (dow == 4 and 18 <= hour <= 22)

        if 0 <= hour <= 5:
            weights, unknown_w = [10.0 for _ in self.users], 0.05
        elif 6 <= hour <= 7:
            weights, unknown_w = [6.0 for _ in self.users], 0.5
        elif 8 <= hour <= 17:
            if is_weekend:
                weights, unknown_w = [9.0 for _ in self.users], 1.5
            else:
                weights   = [9.0 if u['type'] == 'b' else 2.5 for u in self.users]
                unknown_w = 1.5
        elif 18 <= hour <= 21:
            weights, unknown_w = [10.0 for _ in self.users], 0.3
        else:
            weights, unknown_w = [8.0 for _ in self.users], 0.2

        if is_friday_night:
            unknown_w = min(unknown_w * 1.5, 3.0)

        choices = [u['user_id'] for u in self.users] + ["unknown"]
        weights = weights + [unknown_w]
        return random.choices(choices, weights=weights, k=1)[0]

    def _get_session_duration(self, hour: int, user_type: str, is_unknown_visitor: bool = False) -> int:
        if is_unknown_visitor:
            return random.randint(5, 15)

        if 0 <= hour <= 5:
            return random.randint(120, 240)
        elif 6 <= hour <= 7:
            return random.randint(10, 30)
        elif 8 <= hour <= 11:
            return random.randint(30, 90) if user_type == 'b' else random.randint(15, 45)
        elif 12 <= hour <= 13:
            return random.randint(20, 60)
        elif 14 <= hour <= 17:
            return random.randint(25, 75)
        elif 18 <= hour <= 21:
            return random.randint(40, 120)
        else:
            return random.randint(30, 90)

    def _empty_room_probability(self, hour: int, dow: int) -> float:
        is_weekend = (dow >= 5)
        if 0 <= hour <= 5:      return 0.05
        elif 6 <= hour <= 7:    return 0.45
        elif 8 <= hour <= 17:   return 0.35 if is_weekend else 0.55
        elif 18 <= hour <= 21:  return 0.20
        else:                   return 0.30

    def _maybe_start_animal_event(self, hour: int) -> bool:
        if (self.session_time_left > 0
                or self.post_session_gap > 0
                or self.empty_room_time_left > 0
                or self.post_animal_gap > 0): 
            return False

        if 4 <= hour <= 7 or 17 <= hour <= 20:
            prob = 0.030
        elif hour <= 3 or hour >= 21:
            prob = 0.020
        else:
            prob = 0.010

        return random.random() < prob

    def _maybe_add_co_user(self, hour: int, dow: int):
        if self.current_user == "unknown":
            return
        if len(self.co_users) >= len(self.users) - 1:
            return

        available = [
            u['user_id'] for u in self.users
            if u['user_id'] != self.current_user
            and u['user_id'] not in self.co_users
        ]
        if not available:
            return

        is_weekend = (dow >= 5)
        is_friday  = (dow == 4)

        if 18 <= hour <= 22:
            prob = 0.012 if is_friday else 0.010
        elif is_weekend and 8 <= hour <= 22:
            prob = 0.006
        else:
            prob = 0.002

        if random.random() < prob:
            co_uid = random.choice(available)
            self.co_users.append(co_uid)
            self.co_user_time_left[co_uid] = random.randint(10, 60)


    def _tick_co_users(self):
        to_remove = [
            uid for uid in self.co_users
            if self.co_user_time_left.get(uid, 0) <= 1
        ]
        for uid in to_remove:
            self.co_users.remove(uid)
            self.co_user_time_left.pop(uid, None)
        for uid in self.co_users:
            self.co_user_time_left[uid] -= 1

    def _maybe_handoff_to_co_user(self) -> bool:
        if not self.co_users:
            return False

        best_uid  = max(self.co_users, key=lambda uid: self.co_user_time_left.get(uid, 0))
        best_time = self.co_user_time_left.get(best_uid, 0)

        if best_time <= 0:
            return False

        self.co_users.remove(best_uid)
        self.co_user_time_left.pop(best_uid, None)

        self.current_user      = best_uid
        self.session_time_left = best_time
        self.current_session_id = f"sess_{str(uuid.uuid4())[:8]}_{best_uid}_handoff"

        return True

    def _effective_temp(self, temp: float, hum: float) -> float:
        humidity_penalty = max(0.0, (hum - 60.0) * 0.02)
        return round(temp + humidity_penalty, 2)

    def _unknown_face_count(self) -> int:
        return 1 if random.random() < 0.40 else 0

    def _multi_user_face_count(self, total_people: int) -> int:
        raw = total_people
        if raw >= 2 and random.random() < 0.15:
            raw -= 1
        return max(1, raw)

    def _nighttime_light(self, h: int, m: int) -> float:
        if 0 <= h < 4:
            return round(random.uniform(0.0, 2.0), 1)
        elif 4 <= h < 6:
            progress = (h - 4) + m / 60.0  
            lux_base = 1.0 + progress * 7.0
            return round(max(0.0, lux_base + random.uniform(-1, 1)), 1)
        elif 19 <= h < 21:
            return round(random.uniform(3.0, 10.0), 1)
        elif 21 <= h < 23:
            progress = (h - 21) + m / 60.0  
            lux_base = 8.0 - progress * 3.0
            return round(max(1.0, lux_base + random.uniform(-1, 1)), 1)
        else:
            return round(random.uniform(1.0, 4.0), 1)

    def _nighttime_humidity_offset(self, h: int, m: int) -> float:
        if 23 <= h or h < 3:
            if h >= 23:
                progress = (h - 23) + m / 60.0  
            else:
                progress = (h + 1) + m / 60.0    
            return min(8.0, progress * 2.0)
        elif 3 <= h < 6:
            progress = (h - 3) + m / 60.0       
            return max(0.0, 8.0 - progress * 2.67)
        return 0.0

    def _scenario_for_date(self, current_dt: datetime) -> str:
        day_key = current_dt.date().isoformat()
        if day_key not in self._scenario_day_cache:
            # Chọn scenario theo ngày để tạo nhịp dữ liệu tự nhiên hơn,
            # đồng thời bám theo tỉ lệ cân bằng mục tiêu của dataset.
            weights = [config.SCENARIO_WEIGHTS.get(name, 0.25) for name in self._scenario_cycle]
            self._scenario_day_cache[day_key] = random.choices(self._scenario_cycle, weights=weights, k=1)[0]
        return self._scenario_day_cache[day_key]

    def _queue_sensor_delay(self, payload: dict):
        delay = random.randint(1, self._sensor_delay_max)
        self._sensor_delay_queue.append({"delay": delay, "payload": payload})

    def _drain_sensor_delay_queue(self) -> list[dict]:
        ready: list[dict] = []
        next_queue: list[dict] = []
        for item in self._sensor_delay_queue:
            item["delay"] -= 1
            if item["delay"] <= 0:
                ready.append(item["payload"])
            else:
                next_queue.append(item)
        self._sensor_delay_queue = next_queue
        return ready

    def _apply_sensor_noise(self, value: float, scale: float = 1.0) -> float:
        if random.random() >= self._sensor_noise_prob:
            return value
        factor = 1.0 + random.uniform(-self._sensor_noise_strength, self._sensor_noise_strength) * scale
        jitter = random.uniform(-2.0, 2.0) * scale
        return value * factor + jitter

    def _maybe_delay_sensor(self, current_sample: dict) -> list[dict]:
        emitted = self._drain_sensor_delay_queue()
        if random.random() < self._sensor_delay_prob:
            self._queue_sensor_delay(current_sample.copy())
            return emitted
        emitted.append(current_sample)
        return emitted

    def _light_target_from_context(self, light: float, user_pref: float, scenario: str, hour: int) -> int:
        # Ưu tiên giảm false positive cho đèn: ban ngày chỉ bật khi thật sự tối,
        # ban đêm dùng hysteresis rộng hơn để tránh dao động.
        if 22 <= hour or hour <= 5:
            return 1 if light <= max(self._light_threshold_floor, user_pref * 0.33) else 0

        if 6 <= hour <= 7 or 18 <= hour <= 21:
            threshold = user_pref * (0.78 if self._weather_cloudy else 0.86)
            if scenario == "context_exception":
                threshold *= 0.90
            return 1 if light < threshold else 0

        threshold = user_pref * (0.62 if self._weather_cloudy else 0.74)
        if scenario == "context_exception":
            threshold *= 0.90
        return 1 if light < threshold else 0

    def _fan_target_from_context(self, eff_temp: float, composite_temp_pref: float, scenario: str, hour: int, humidity: float, last_fan: int = 0) -> int:
        fan_on_at = composite_temp_pref + 0.6
        fan_off_at = composite_temp_pref - 0.6

        if scenario == "context_exception":
            if humidity >= 85 and eff_temp < composite_temp_pref + 0.8:
                return 0
            if 22 <= hour or hour <= 5:
                if last_fan == 1:
                    return 0 if eff_temp <= fan_off_at else 1
                return 1 if eff_temp >= fan_on_at else 0
        if last_fan == 1:
            return 0 if eff_temp <= fan_off_at else 1
        return 1 if eff_temp >= fan_on_at else 0

    def _user_profile(self, user_id: str) -> dict:
        key = (user_id or "unknown").strip().lower()
        if key in self._user_profiles:
            return self._user_profiles[key]
        return self._user_profiles.get("unknown", {"fan_on_delta": 1.0, "fan_off_delta": 0.5, "light_off_delta": 22.0})

    def _light_target_hysteresis(self, light: float, motion: bool, user_lux_threshold: float, last_light: int, profile: dict) -> int:
        light_on_at = max(5.0, user_lux_threshold * 0.9)
        light_off_at = max(light_on_at + 5.0, user_lux_threshold + float(profile.get("light_off_delta", 20.0)))
        if last_light == 1:
            return 1 if (motion or light < light_off_at) else 0
        if motion and light <= light_on_at:
            return 1
        return 1 if light <= user_lux_threshold * 0.6 else 0

    def _fan_target_hysteresis(self, eff_temp: float, humidity: float, composite_temp_pref: float, scenario: str, hour: int, last_fan: int, profile: dict) -> int:
        fan_on_at = composite_temp_pref + float(profile.get("fan_on_delta", 0.8))
        fan_off_at = composite_temp_pref - float(profile.get("fan_off_delta", 0.6))

        if scenario == "context_exception" and humidity >= 85 and eff_temp < composite_temp_pref + 0.8:
            return 0

        if 22 <= hour or hour <= 5:
            fan_on_at -= 0.2
            fan_off_at -= 0.1

        if last_fan == 1:
            return 0 if eff_temp <= fan_off_at else 1
        return 1 if eff_temp >= fan_on_at else 0

    def _abnormal_presence_pattern(self, hour: int, dow: int, presence_type: str, motion: int, face_count: int, light_state: int, fan_state: int) -> tuple[int, int, int, int, str]:
        if random.random() >= self._abnormal_behavior_prob:
            return motion, face_count, light_state, fan_state, presence_type

        if presence_type == PT_SINGLE_KNOWN and 7 <= hour <= 11:
            if random.random() < 0.55:
                face_count = 0
                motion = 1
                presence_type = PT_SINGLE_UNKNOWN
        elif presence_type == PT_SINGLE_UNKNOWN and random.random() < 0.35:
            face_count = 1
        elif presence_type == PT_MULTI_USER and random.random() < 0.30:
            face_count = max(1, face_count - 1)

        if presence_type not in (PT_EMPTY, PT_ANIMAL):
            if random.random() < 0.20:
                light_state = 1 - light_state if random.random() < 0.50 else light_state
            if random.random() < 0.15:
                fan_state = 1 - fan_state if random.random() < 0.50 else fan_state

        return motion, face_count, light_state, fan_state, presence_type

    def _near_miss_override(self, light: float, motion: int, last_light: int, user_lux_threshold: float, light_state: int) -> int:
        if random.random() >= self._near_miss_prob:
            return light_state

        if motion == 1 and user_lux_threshold * 0.55 <= light <= user_lux_threshold * 1.05:
            return 1 if last_light == 1 else 0

        if user_lux_threshold * 0.90 <= light <= user_lux_threshold * 1.08:
            return 0

        if light <= user_lux_threshold * 0.72:
            return 1

        return light_state

    def _inject_override_sequence(self, hour: int, current_user: str, base_light: int, base_fan: int) -> tuple[int, int, bool, str]:
        if self._override_active and self._override_remaining > 0 and self._override_target:
            self._override_remaining -= 1
            if self._override_remaining <= 0:
                self._override_active = False
            return self._override_target[0], self._override_target[1], True, self._override_reason

        if current_user != "unknown":
            manual_probs = [
                (0 <= hour <= 5, 0.006),
                (6 <= hour <= 8, 0.018),
                (9 <= hour <= 11, 0.014),
                (12 <= hour <= 14, 0.028),
                (15 <= hour <= 17, 0.022),
                (18 <= hour <= 22, 0.026),
                (hour >= 23, 0.010),
            ]
            prob = next((p for cond, p in manual_probs if cond), 0.014)
            if random.random() < prob:
                if base_light == 1:
                    manual_light = 0 if random.random() < 0.72 else 1
                else:
                    manual_light = 1 if random.random() < 0.82 else 0

                if base_fan == 1:
                    manual_fan = 0 if random.random() < 0.62 else 1
                else:
                    manual_fan = 1 if random.random() < 0.76 else 0

                self._override_active = True
                self._override_remaining = random.randint(3, 6)
                self._override_target = (manual_light, manual_fan)
                self._override_reason = "manual_override_demo"
                return manual_light, manual_fan, True, self._override_reason

        return base_light, base_fan, False, ""

    def _maybe_perturb_light_for_balance(self, light_state: int, light: float, user_lux_threshold: float, motion: int, scenario: str, hour: int, last_light: int) -> int:
        if scenario == "manual_override":
            return light_state
        if light_state == 1 and random.random() < 0.10 and light > user_lux_threshold * self._light_conservative_bias:
            return 0
        if light_state == 0 and random.random() < 0.08 and motion == 1 and light < user_lux_threshold * self._light_false_positive_guard:
            return 1
        if self._near_miss_prob > 0:
            return self._near_miss_override(light, motion, last_light, user_lux_threshold, light_state)
        return light_state

    def _maybe_start_sequence_behavior(self, hour: int, dow: int):
        if self.session_time_left > 0 or self.post_session_gap > 0 or self.empty_room_time_left > 0 or self.post_animal_gap > 0:
            return
        if 17 <= hour <= 20 and dow < 5 and random.random() < 0.10:
            self.current_user = random.choice([u["user_id"] for u in self.users if u["user_id"] != "unknown"] or ["unknown"])
            self.current_session_id = f"sess_{str(uuid.uuid4())[:8]}_{self.current_user}_seq"
            self.session_time_left = random.randint(12, 25)
            self._is_sleep_session = False
            self._scenario_day_cache["active"] = "sequence_behavior"

    # ─────────────────────────────────────────────────────────────
    #  HÀM GENERATE CHÍNH
    # ─────────────────────────────────────────────────────────────
    def generate(self, num_samples: int = 1440) -> tuple[list[EnvSample], list[str]]:
        dataset      = []
        sessions_log = []
        current_time = self.start_date

        _prev_fan_state = 0
        
        _last_recorded_light = -1
        _last_recorded_fan = -1
        _last_recorded_presence = ""

        for step in range(num_samples):
            h, m, dow = current_time.hour, current_time.minute, current_time.weekday()

            self._refresh_weather_if_new_day(current_time)
            self._refresh_temp_offset_if_new_day(current_time) 

            if m == 0:
                self._update_heat_accumulation(h, _prev_fan_state == 1)

            cloud_temp_reduction = random.uniform(1.0, 2.0) if self._weather_cloudy else 0.0
            t_base = (
                28.0
                + 4.0 * math.sin((h - 8) * math.pi / 12)
                - cloud_temp_reduction
                + self._daily_temp_offset
                + self._heat_accumulation
            )
            temp = round(t_base + random.uniform(-0.5, 0.5), 1)

            hum_base   = 70.0 + 15.0 * math.cos((h - 6) * math.pi / 12) + random.uniform(-2, 2)
            hum_offset = self._nighttime_humidity_offset(h, m) 
            hum = round(hum_base + hum_offset, 1)
            hum = max(40.0, min(95.0, hum))

            if 6 <= h <= 18:
                sun_intensity = math.sin((h - 6) * math.pi / 12)
                cloud_factor  = random.uniform(0.15, 0.40) if self._weather_cloudy else 1.0
                light = round(sun_intensity * 400 * cloud_factor + random.uniform(-10, 10), 1)
                light = max(0.0, light)
            else:
                light = self._nighttime_light(h, m)

            eff_temp = self._effective_temp(temp, hum)

            presence_type = PT_EMPTY
            motion        = 0
            face_count    = 0
            override_flag = False
            override_reason = ""

            if self.animal_event_left > 0:
                self.animal_event_left -= 1
                presence_type           = PT_ANIMAL
                motion                  = 1
                face_count              = 0 
                self.current_session_id = None
                self.current_user       = "unknown"
                if self.animal_event_left == 0:
                    self.post_animal_gap = random.randint(1, 5)

            elif self.post_animal_gap > 0:
                self.post_animal_gap -= 1
                presence_type         = PT_EMPTY
                motion                = 0
                self._fan_hys         = 0

            elif self.session_time_left > 0:
                self.session_time_left -= 1

                if not self._is_sleep_session and self._is_sleeping_hour(h, dow):
                    self._is_sleep_session = True

                if self._is_sleep_session:
                    motion = 1 if random.random() < 0.08 else 0
                else:
                    motion = 1 if random.random() < 0.85 else 0

                self._tick_co_users()
                self._maybe_add_co_user(h, dow)

                if self.co_users:
                    presence_type = PT_MULTI_USER
                    total_people  = 1 + len(self.co_users)
                    face_count    = self._multi_user_face_count(total_people)
                elif self.current_user != "unknown":
                    presence_type = PT_SINGLE_KNOWN
                    face_count    = 1
                else:
                    presence_type = PT_SINGLE_UNKNOWN
                    face_count    = self._unknown_face_count()

                if self.session_time_left == 0:
                    if not self._maybe_handoff_to_co_user():
                        self.post_session_gap = random.randint(1, 3)
                    else:
                        sessions_log.append(self.current_session_id)

            elif self.post_session_gap > 0:
                self.post_session_gap -= 1
                self.current_session_id = None
                self.current_user       = "unknown"
                self.co_users           = []
                self.co_user_time_left  = {}
                self._is_sleep_session  = False
                if eff_temp < self.t_threshold - 0.5:
                    self._fan_hys = 0
                presence_type           = PT_EMPTY
                motion                  = 0

            elif self.empty_room_time_left > 0:
                self.empty_room_time_left -= 1
                motion                    = 0
                self.current_user         = "unknown"
                self.current_session_id   = None
                self.co_users             = []
                self.co_user_time_left    = {}
                if eff_temp < self.t_threshold - 0.5:
                    self._fan_hys = 0
                presence_type             = PT_EMPTY

            else:
                self._maybe_start_sequence_behavior(h, dow)
                self.co_users          = []
                self.co_user_time_left = {}

                if self._maybe_start_animal_event(h):
                    self.animal_event_left  = random.randint(2, 15)
                    self.current_session_id = None
                    self.current_user       = "unknown"
                    presence_type           = PT_ANIMAL
                    motion                  = 1
                    face_count              = 0

                elif random.random() < self._empty_room_probability(h, dow):
                    self.empty_room_time_left = random.randint(15, 120)
                    motion                    = 0
                    self.current_user         = "unknown"
                    self.current_session_id   = None
                    if eff_temp < self.t_threshold - 0.5:
                        self._fan_hys = 0
                    presence_type             = PT_EMPTY

                else:
                    new_user = self._get_active_user(h, dow)
                    u_prof   = next(
                        (u for u in self.users if u['user_id'] == new_user), None
                    )
                    u_type = u_prof['type'] if u_prof else 'b'

                    is_visitor = (
                        new_user == "unknown"
                        and not (dow >= 5)
                        and 8 <= h <= 18
                        and random.random() < 0.55
                    )

                    dur      = self._get_session_duration(h, u_type, is_unknown_visitor=is_visitor)
                    is_sleep = self._is_sleeping_hour(h, dow)

                    self.current_user         = new_user
                    self.session_time_left    = dur
                    self.current_session_id   = (
                        f"sess_{str(uuid.uuid4())[:8]}_{new_user}"
                    )
                    self._is_sleep_session    = is_sleep
                    sessions_log.append(self.current_session_id)

                    motion = 0 if is_sleep else 1

                    if new_user != "unknown":
                        presence_type = PT_SINGLE_KNOWN
                        face_count    = 1
                    else:
                        presence_type = PT_SINGLE_UNKNOWN
                        face_count    = self._unknown_face_count()

            scenario = self._scenario_for_date(current_time)
            override_flag = False
            override_reason = ""

            if presence_type not in (PT_EMPTY, PT_ANIMAL):
                u_profile = next((u for u in self.users if u['user_id'] == self.current_user), None)
                all_profiles = [u_profile] if u_profile else []
                for co_uid in self.co_users:
                    co_p = next((u for u in self.users if u['user_id'] == co_uid), None)
                    if co_p:
                        all_profiles.append(co_p)

                composite_temp_pref = min(p['base_temp_pref'] for p in all_profiles) if all_profiles else self.t_threshold
                user_lux_threshold = u_profile['base_lux_pref'] if u_profile else self.l_threshold
                profile = self._user_profile(self.current_user)
                last_light = self._last_prediction.get("light_state") if self._last_prediction else 0
                last_fan = self._last_prediction.get("fan_state") if self._last_prediction else 0

                if scenario == "sequence_behavior":
                    if 17 <= h <= 21:
                        if m < 8:
                            light_state = 1
                            fan_state = self._fan_target_hysteresis(eff_temp, hum, composite_temp_pref, scenario, h, last_fan, profile)
                        elif m < 20:
                            light_state = 1 if light <= user_lux_threshold * 0.76 else 0
                            fan_state = 1 if eff_temp >= composite_temp_pref - 0.1 else 0
                        else:
                            light_state = 1 if light < user_lux_threshold * 0.80 else 0
                            fan_state = self._fan_target_hysteresis(eff_temp, hum, composite_temp_pref, scenario, h, last_fan, profile)
                    elif 6 <= h <= 8:
                        light_state = 1 if light < user_lux_threshold * 0.76 else 0
                        fan_state = self._fan_target_hysteresis(eff_temp, hum, composite_temp_pref, scenario, h, last_fan, profile)
                    else:
                        light_state = 1 if light < user_lux_threshold * 0.82 else 0
                        fan_state = self._fan_target_hysteresis(eff_temp, hum, composite_temp_pref, scenario, h, last_fan, profile)
                elif scenario == "context_exception":
                    if 22 <= h or h <= 5:
                        light_state = 0 if light > 28 else (1 if light < user_lux_threshold * 0.56 else 0)
                    elif 12 <= h <= 14 and eff_temp >= self.t_threshold:
                        light_state = 0 if light > 100 else (1 if light < user_lux_threshold * 0.70 else 0)
                    else:
                        light_state = 1 if light < user_lux_threshold * 0.68 else 0
                    fan_state = self._fan_target_hysteresis(eff_temp, hum, composite_temp_pref, scenario, h, last_fan, profile)
                elif scenario == "manual_override":
                    if random.random() < 0.52:
                        # Mẫu “gần đúng nhưng không đúng hoàn toàn” để tăng tính thực tế.
                        if light <= user_lux_threshold * 0.85:
                            light_state = 0 if random.random() < 0.58 else 1
                        else:
                            light_state = 1 if random.random() < 0.22 else 0
                        if eff_temp >= composite_temp_pref - 0.2:
                            fan_state = 1 if random.random() < 0.64 else 0
                        else:
                            fan_state = 0 if random.random() < 0.68 else 1
                    else:
                        light_state = self._light_target_hysteresis(light, motion, user_lux_threshold, last_light, profile)
                        fan_state = self._fan_target_hysteresis(eff_temp, hum, composite_temp_pref, scenario, h, last_fan, profile)
                else:
                    if self._weather_cloudy and 9 <= h <= 16:
                        light_state = 1 if light < user_lux_threshold * 0.75 else 0
                    else:
                        light_state = 1 if light < user_lux_threshold * 0.86 else 0
                    fan_state = self._fan_target_hysteresis(eff_temp, hum, composite_temp_pref, scenario, h, last_fan, profile)

                if self._is_sleep_session:
                    light_state = 0
                    fan_state = 0

                if presence_type == PT_SINGLE_KNOWN:
                    light_state = self._maybe_perturb_light_for_balance(
                        light_state, light, user_lux_threshold, motion, scenario, h, last_light
                    )

                motion, face_count, light_state, fan_state, presence_type = self._abnormal_presence_pattern(
                    h, dow, presence_type, motion, face_count, light_state, fan_state
                )

                light_state, fan_state, override_flag, override_reason = self._inject_override_sequence(
                    h, self.current_user, light_state, fan_state
                )
                self._last_prediction = {"light_state": light_state, "fan_state": fan_state}
            else:
                light_state = 0
                fan_state   = 0
                self._fan_hys = 0
                self._last_prediction = {"light_state": 0, "fan_state": 0}

            _prev_fan_state = fan_state

            # ──────────────────────────────────────────────────────
            # 4. GHI SAMPLE VÀ CẬP NHẬT TRẠNG THÁI (FIX BUG-12)
            # ──────────────────────────────────────────────────────
            is_interval = (m % 2 == 0)
            is_event = (light_state != _last_recorded_light) or \
                       (fan_state != _last_recorded_fan) or \
                       (presence_type != _last_recorded_presence)

            scenario_type = scenario if presence_type not in (PT_EMPTY, PT_ANIMAL) else "background_noise"
            if override_flag:
                scenario_type = "manual_override"
            elif presence_type == PT_EMPTY:
                # Giảm tỷ lệ background_noise thuần, tăng phần ngoại lệ theo ngữ cảnh.
                if 22 <= h or h <= 5 or self._weather_cloudy or random.random() < 0.35:
                    scenario_type = "context_exception"

            if presence_type in (PT_EMPTY, PT_ANIMAL):
                light_state = 0
                fan_state = 0
                face_count = 0
                motion = 0

            if is_interval or is_event:
                dataset.append(EnvSample(
                    timestamp       = current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    hour            = h,
                    minute          = m,
                    day_of_week     = dow,
                    temperature     = temp,
                    humidity        = hum,
                    light_level     = light,
                    motion_detected = motion,
                    user_id         = self.current_user,
                    light_state     = light_state,
                    fan_state       = fan_state,
                    session_id      = self.current_session_id,
                    presence_type   = presence_type,
                    face_count      = face_count,
                    scenario_type   = scenario_type,
                    data_source     = "simulated",
                    override_flag   = 1 if override_flag else 0,
                    override_reason = override_reason,
                ))
            
            # [FIX BUG-12]: Rút biến cập nhật tracker ra khỏi khối if 
            # để đảm bảo luôn theo dõi trạng thái phút liền trước một cách chính xác
            _last_recorded_light = light_state
            _last_recorded_fan = fan_state
            _last_recorded_presence = presence_type
            
            current_time += timedelta(minutes=1)

        return dataset, list(set(sessions_log))


# ─────────────────────────────────────────────────────────────────
#  [FIX BUG-TRAIN-01] CHIA NHỎ TRANSACTION VÀ YIELD CPU KHI LƯU DB
# ─────────────────────────────────────────────────────────────────
def save_to_sqlite(dataset: list[EnvSample], db_instance: Database):
    try:
        print("  Đang dọn dẹp dữ liệu mô phỏng cũ...")
        with db_instance.get_connection() as conn:
            cur = conn.execute(
                "SELECT env_log_id FROM device_states "
                "WHERE source = 'simulated' AND env_log_id IS NOT NULL"
            )
            simulated_env_ids = [row[0] for row in cur.fetchall()]

        # [FIX BUG DB-LOCK]: Áp dụng chunking để xóa `device_states` trước thay vì xóa một cục
        if simulated_env_ids:
            chunk_size = 900
            for i in range(0, len(simulated_env_ids), chunk_size):
                chunk = simulated_env_ids[i : i + chunk_size]
                placeholders = ','.join(['?'] * len(chunk))
                with db_instance.get_connection() as conn:
                    # Xóa bảng con trước
                    conn.execute(f"DELETE FROM device_states WHERE env_log_id IN ({placeholders}) AND source = 'simulated'", chunk)
                    # Xóa bảng cha
                    conn.execute(f"DELETE FROM environmental_logs WHERE id IN ({placeholders})", chunk)
                time.sleep(0.05) # Yield cho các truy vấn khác

        if not dataset:
            print("  ⚠️ Không có dữ liệu nào để lưu.")
            return

        print("  Đang chèn dữ liệu mới...")
        BATCH_SIZE = 500
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i+BATCH_SIZE]
            with db_instance.get_connection() as conn:
                for s in batch:
                    cursor = conn.execute(
                        """
                        INSERT INTO environmental_logs
                        (timestamp, hour, minute, day_of_week,
                         temperature, humidity, light_level, motion_detected)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (s.timestamp, s.hour, s.minute, s.day_of_week,
                         s.temperature, s.humidity, s.light_level, s.motion_detected),
                    )
                    env_log_id = cursor.lastrowid
                    conn.execute(
                        """
                        INSERT INTO device_states
                        (timestamp, env_log_id, user_id, light_state, fan_state,
                         source, data_source, scenario_type, override_flag, override_reason,
                         session_id, presence_type, face_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            s.timestamp, env_log_id, s.user_id, s.light_state, s.fan_state,
                            'simulated', (s.data_source or 'simulated').strip().lower(),
                            (s.scenario_type or 'background_noise').strip().lower(),
                            int(s.override_flag or 0),
                            s.override_reason, s.session_id, s.presence_type, s.face_count,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO device_training_logs
                        (env_log_id, user_id, light_state, fan_state, source, data_source,
                         scenario_type, override_flag, override_reason, session_id,
                         presence_type, face_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            env_log_id, s.user_id, s.light_state, s.fan_state,
                            'simulated', (s.data_source or 'simulated').strip().lower(),
                            (s.scenario_type or 'background_noise').strip().lower(),
                            int(s.override_flag or 0),
                            s.override_reason, s.session_id, s.presence_type, s.face_count,
                        ),
                    )
            time.sleep(0.05)  # Yield CPU
            print(f"  Đã lưu {min(i+BATCH_SIZE, len(dataset))}/{len(dataset)}...")

        print(f"  ✅ Đã lưu {len(dataset)} mẫu vào SQLite thành công.")

    except Exception as e:
        print(f"  ❌ Lỗi khi lưu vào SQLite: {e}")


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db = Database()
    auto_sync_faces_to_db(db)

    parser = argparse.ArgumentParser(
        description="Sinh dữ liệu huấn luyện cho Smart Home AI (v7.5 - Fully Patched)"
    )
    parser.add_argument("--days",     type=int, default=14, help="Số ngày mô phỏng (Offline)")
    parser.add_argument("--realtime", action="store_true",  help="Sinh dữ liệu lùi từ hiện tại")
    args = parser.parse_args()

    n_samples = max(1, args.days) * 24 * 60

    if args.realtime:
        start_date = datetime.now() - timedelta(minutes=n_samples)
        print(
            f"Bắt đầu sinh dữ liệu realtime từ "
            f"{start_date.strftime('%Y-%m-%d %H:%M:%S')} tới hiện tại..."
        )
        try:
            with db.get_connection() as conn:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM device_states "
                    "WHERE source != 'simulated' AND timestamp >= ?",
                    (start_date.strftime('%Y-%m-%d %H:%M:%S'),),
                )
                real_count = cur.fetchone()[0]
                if real_count > 0:
                    print(f"\n  ⚠️  CẢNH BÁO: Đã có {real_count} bản ghi THỰC trong khoảng mô phỏng.")
                    print("      Tiếp tục sau 3 giây...\n")
                    time.sleep(3)
        except Exception:
            pass
    else:
        start_date = datetime(2025, 6, 1, 0, 0, 0)
        print(
            f"Bắt đầu sinh dữ liệu chuẩn từ "
            f"{start_date.strftime('%Y-%m-%d %H:%M:%S')} ({max(1, args.days)} ngày)..."
        )

    # Truyền db_instance vào Generator
    generator = SmartHomeDataGenerator(start_date=start_date, db_instance=db, is_realtime=args.realtime)
    dataset, sessions = generator.generate(num_samples=n_samples)

    if not dataset:
        print("  ❌ Lỗi: Tập dữ liệu sinh ra rỗng!")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────
    # BÁO CÁO CHẤT LƯỢNG
    # ─────────────────────────────────────────────────────────────
    n = len(dataset)

    pt_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    users_cnt: dict[str, int] = {}
    lights_on = fans_on = 0
    animal_device_on = empty_device_on = sleep_light_on = 0
    lux_mid = lux_mid_lit = 0
    back_to_back = 0
    handoff_sessions = 0
    prev_session_id = None

    for s in dataset:
        pt_counts[s.presence_type] = pt_counts.get(s.presence_type, 0) + 1
        scenario_counts[s.scenario_type] = scenario_counts.get(s.scenario_type, 0) + 1
        users_cnt[s.user_id]       = users_cnt.get(s.user_id, 0) + 1
        if s.light_state: lights_on += 1
        if s.fan_state:   fans_on   += 1

        if s.presence_type == PT_ANIMAL and (s.light_state or s.fan_state):
            animal_device_on += 1
        if s.presence_type == PT_EMPTY and (s.light_state or s.fan_state):
            empty_device_on += 1
        if (s.hour <= 5 or s.hour >= 23) and s.light_state:
            sleep_light_on += 1
        if 180 < s.light_level <= 300:
            lux_mid += 1
            if s.light_state: lux_mid_lit += 1
        if s.session_id and "_handoff" in s.session_id:
            handoff_sessions += 1

        if (s.session_id and s.session_id != prev_session_id
                and prev_session_id is not None
                and s.presence_type != PT_EMPTY
                and s.presence_type != PT_ANIMAL):
            back_to_back += 1
        prev_session_id = s.session_id

    print("\n  ─── Phân phối Presence Type ───")
    pt_order = [PT_SINGLE_KNOWN, PT_MULTI_USER, PT_SINGLE_UNKNOWN, PT_ANIMAL, PT_EMPTY]
    for pt in pt_order:
        cnt = pt_counts.get(pt, 0)
        bar = "█" * int(cnt / n * 40)
        print(f"    {pt:<20}: {cnt:6d} ({cnt / n * 100:.1f}%)  {bar}")

    print("\n  ─── Phân phối Người dùng ───")
    for uid, cnt in sorted(users_cnt.items(), key=lambda x: -x[1]):
        print(f"    {uid:<20}: {cnt:6d} ({cnt / n * 100:.1f}%)")

    print(f"\n  Tổng số phiên     : {len(sessions)}")
    print(f"  Trung bình mỗi ph.: {n / max(len(sessions), 1):.0f} phút")
    if handoff_sessions > 0:
        print(f"  Co-user handoff   : {handoff_sessions} mẫu [V7-3]")

    print(f"\n  ─── Phân phối scenario ───")
    for sc, cnt in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        print(f"    {sc:<20}: {cnt:6d} ({cnt / n * 100:.1f}%)")


    print(f"\n  ─── Nhãn thiết bị ───")
    print(f"    Đèn BẬT : {lights_on:6d} ({lights_on / n * 100:.1f}%)")
    print(f"    Quạt BẬT: {fans_on:6d}   ({fans_on / n * 100:.1f}%)")

    print(f"\n  ─── Kiểm tra bất biến (phải = 0) ───")
    ok = lambda v: "✅" if v == 0 else "❌"
    print(f"    {ok(animal_device_on)} [I2] animal + thiết bị BẬT    : {animal_device_on}")
    print(f"    {ok(empty_device_on)}  [I1] empty  + thiết bị BẬT    : {empty_device_on}")
    print(f"    {ok(sleep_light_on)}  [I3] giờ ngủ + đèn BẬT        : {sleep_light_on}")
    print(f"    {ok(back_to_back)}  [V6-2] Back-to-back sessions  : {back_to_back}")
    if lux_mid > 0:
        print(f"    [INFO] Lux 180-300 → đèn BẬT  : {lux_mid_lit}/{lux_mid} ({lux_mid_lit/lux_mid*100:.1f}%)")

    multi_cnt    = pt_counts.get(PT_MULTI_USER, 0)
    multi_fan_on = sum(1 for s in dataset if s.presence_type == PT_MULTI_USER and s.fan_state)
    if multi_cnt > 0:
        print(f"\n  ─── Thống kê Multi-User ───")
        print(f"    Tổng mẫu multi_user    : {multi_cnt} ({multi_cnt / n * 100:.1f}%)")
        print(f"    multi_user + quạt BẬT  : {multi_fan_on} ({multi_fan_on/multi_cnt*100:.1f}%)")

    unk_cnt   = pt_counts.get(PT_SINGLE_UNKNOWN, 0)
    unk_face1 = sum(1 for s in dataset if s.presence_type == PT_SINGLE_UNKNOWN and s.face_count == 1)
    unk_face0 = sum(1 for s in dataset if s.presence_type == PT_SINGLE_UNKNOWN and s.face_count == 0)
    if unk_cnt > 0:
        print(f"\n  ─── Thống kê Single Unknown [V6-4] ───")
        print(f"    face_count=1 (unrecognized) : {unk_face1} ({unk_face1/unk_cnt*100:.1f}%)")
        print(f"    face_count=0 (PIR only)     : {unk_face0} ({unk_face0/unk_cnt*100:.1f}%)")

    temps = [s.temperature for s in dataset]
    print(f"\n  ─── Kiểm tra môi trường [V7] ───")
    print(f"    Nhiệt độ min/max/avg : {min(temps):.1f} / {max(temps):.1f} / {sum(temps)/len(temps):.1f}°C")
    hums = [s.humidity for s in dataset]
    print(f"    Độ ẩm min/max/avg    : {min(hums):.1f} / {max(hums):.1f} / {sum(hums)/len(hums):.1f}%")

    print("\n  Đang lưu trữ dữ liệu...")
    save_to_sqlite(dataset, db) # Truyền object db vào đây


    # [FIX BUG-PATH] Tự động tạo thư mục nếu chưa tồn tại
    data_dir = getattr(config, 'DATA_DIR', os.path.dirname(config.DATABASE_PATH) if hasattr(config, 'DATABASE_PATH') else 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "training_data.csv")
    
    try:
        with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "hour", "minute", "day_of_week",
                "temperature", "humidity", "light_level", "motion_detected",
                "user_id", "light_state", "fan_state", "session_id",
                "presence_type", "face_count",
                "scenario_type", "override_flag", "override_reason",
            ])
            for s in dataset:
                writer.writerow([
                    s.timestamp, s.hour, s.minute, s.day_of_week,
                    s.temperature, s.humidity, s.light_level, s.motion_detected,
                    s.user_id, s.light_state, s.fan_state, s.session_id,
                    s.presence_type, s.face_count, s.scenario_type,
                    s.override_flag, s.override_reason,
                ])
        print(f"  ✅ Đã xuất {len(dataset)} mẫu ra CSV: {csv_path}")
    except Exception as e:
        print(f"  ❌ Lỗi khi xuất CSV: {e}")