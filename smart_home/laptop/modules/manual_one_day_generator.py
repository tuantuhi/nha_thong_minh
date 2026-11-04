

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

DEFAULT_ACTIVE_USERS = "[]"
DEFAULT_WEATHER = {
    "temperature": 31.0,
    "humidity": 68.0,
    "light_level": 800.0,
}

HOST_PROFILES = {
    "host_a": {
        "user_id": "user_a",
        "presence_type": "human",
        "face_count": 1,
        "light_bias": 1,
        "fan_bias": 1,
        "night_light_pref": 1,
        "night_fan_pref": 1,
    },
    "host_b": {
        "user_id": "user_b",
        "presence_type": "human",
        "face_count": 1,
        "light_bias": 0,
        "fan_bias": 0,
        "night_light_pref": 1,
        "night_fan_pref": 0,
    },
}

WEATHER_PATTERNS = {
    "hot": {"temperature": (30.5, 34.5), "humidity": (55.0, 75.0), "light_level": (500.0, 1000.0)},
    "rainy": {"temperature": (24.0, 28.5), "humidity": (80.0, 95.0), "light_level": (60.0, 250.0)},
    "cold": {"temperature": (20.0, 24.5), "humidity": (45.0, 70.0), "light_level": (100.0, 500.0)},
    "default": {"temperature": (27.0, 31.0), "humidity": (60.0, 78.0), "light_level": (300.0, 800.0)},
}

RNG = random.Random(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from database import Database
from modules.priority_policy import DECISION_MODE_FALLBACK, DECISION_MODE_MANUAL_OVERRIDE, DECISION_MODE_MULTI_USER, DECISION_MODE_SINGLE_USER, normalize_decision_mode


@dataclass
class MinuteRecord:
    timestamp: str
    hour: int
    minute: int
    day_of_week: int
    temperature: float
    humidity: float
    light_level: float
    motion_detected: int
    user_id: str
    dominant_user_id: str
    presence_type: str
    face_count: int
    light_state: int
    fan_state: int
    confidence: float
    source: str = "simulated"
    scenario_type: str = "background_noise"
    sequence_phase: str = "background_noise"
    override_flag: int = 0
    override_reason: str = ""
    session_id: str = ""
    decision_mode: str = "fallback"
    data_source: str = "simulated"
    


def _ts(base: datetime, minute_offset: int) -> datetime:
    return base + timedelta(minutes=minute_offset)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _jitter(value: float, amount: float) -> float:
    return value + RNG.uniform(-amount, amount)


def _smooth_step(value: float, target: float, step: float) -> float:
    if value < target:
        return min(value + step, target)
    return max(value - step, target)


def _make_session(prefix: str, user_id: str, start_dt: datetime) -> str:
    return f"{prefix}_{start_dt.strftime('%Y%m%d%H%M')}_{str(uuid.uuid4())[:8]}_{user_id}"


def _pick_weather_profile(rng: random.Random, kind: str | None = None) -> tuple[str, dict]:
    if kind in WEATHER_PATTERNS:
        return kind, WEATHER_PATTERNS[kind]
    kind = rng.choices(["hot", "rainy", "cold", "default"], weights=[0.38, 0.24, 0.18, 0.20], k=1)[0]
    return kind, WEATHER_PATTERNS[kind]


def _shift_timestamp_fields(dt: datetime, rng: random.Random, max_minutes: int = 1) -> datetime:
    if max_minutes <= 0:
        return dt
    return dt + timedelta(minutes=rng.randint(-max_minutes, max_minutes))


def _pick_host_profile(rng: random.Random, host_key: str | None = None) -> tuple[str, dict]:
    if host_key in HOST_PROFILES:
        return host_key, HOST_PROFILES[host_key]
    host_key = rng.choice(list(HOST_PROFILES.keys()))
    return host_key, HOST_PROFILES[host_key]



def _day_seed(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d"))


def _day_rng(dt: datetime) -> random.Random:
    return random.Random(42 + _day_seed(dt))


def _apply_day_variation(value: float, rng: random.Random, amount: float) -> float:
    return value + rng.uniform(-amount, amount)



def _sequence_phase_for(dt: datetime, scenario_type: str, event_type: str, user_id: str, presence_type: str) -> str:
    hour = dt.hour
    minute = dt.minute
    total_minutes = hour * 60 + minute

    if event_type in {"arrive_home", "arrive_light_on"}:
        return "arrive_light_on"
    if event_type in {"settle_down", "settle_before_fan"}:
        return "settle_before_fan"
    if event_type in {"turn_fan_on", "fan_on_after_delay"}:
        return "fan_on_after_delay"
    if event_type in {"watch_movie", "manual_adjust"}:
        return "manual_adjust_override"
    if event_type == "sleeping":
        return "sleep_background"

    if scenario_type == "sequence_behavior":
        if 5 * 60 <= total_minutes < 6 * 60:
            return "wake_up"
        if 6 * 60 <= total_minutes < 7 * 60:
            return "morning_routine"
        if 7 * 60 <= total_minutes < 9 * 60:
            return "arrival_sequence" if presence_type != "empty" else "sequence_tail"
        if 11 * 60 <= total_minutes < 13 * 60:
            return "midday_routine"
        if 13 * 60 <= total_minutes < 17 * 60:
            return "afternoon_routine"
        if 17 * 60 <= total_minutes < 20 * 60:
            return "evening_arrival"
        if 21 * 60 <= total_minutes < 22 * 60 + 30:
            return "winddown"
        return "sequence_idle"
    if scenario_type == "context_exception":
        if 11 * 60 <= total_minutes < 13 * 60:
            return "temperature_spike" if minute % 4 < 2 else "humidity_spike"
        if 14 * 60 <= total_minutes < 17 * 60:
            return "settle_before_fan" if minute % 4 < 2 else "fan_on_after_delay"
        if hour >= 22 or hour <= 5:
            return "night_context_noise"
        return "context_shift"
    if scenario_type == "manual_override":
        if presence_type == "empty":
            return "away_override"
        if "watch_movie" in event_type:
            return "watch_movie_override"
        if "manual_adjust" in event_type:
            return "manual_adjust_override"
        return "settle_before_fan"
    if scenario_type == "profile_personalization":
        if "," in user_id:
            return "multi_user_priority"
        if user_id == "user_b":
            return "user_b_preference"
        if user_id == "user_a":
            return "user_a_preference"
        return "arrival_sequence"
    if scenario_type == "background_noise":
        if presence_type == "empty":
            if hour <= 5:
                return "sleep_background"
            if hour < 9:
                return "morning_idle"
            if hour < 12:
                return "midmorning_idle"
            if hour < 18:
                return "away_noise"
            if hour < 23:
                return "winddown_idle"
            return "sleep_background"
        if hour < 9:
            return "occupied_morning_noise"
        if hour < 18:
            return "occupied_day_noise"
        return "occupied_evening_noise"
    return "sequence_idle"


def _record(
    dt: datetime,
    temperature: float,
    humidity: float,
    light_level: float,
    motion: int,
    user_id: str,
    light_state: int,
    fan_state: int,
    session_id: str | None,
    presence_type: str,
    face_count: int,
    scenario_type: str,
    override_flag: int = 0,
    override_reason: str = "",
    confidence: float = 0.0,
    dominant_user_id: str | None = None,
    decision_mode: str = DECISION_MODE_FALLBACK,
    data_source: str = "real",
    source: str = "real",
) -> MinuteRecord:
    temp = _clamp(_jitter(temperature, 0.25), 18.0, 38.0)
    hum = _clamp(_jitter(humidity, 0.8), 20.0, 95.0)
    light = _clamp(_jitter(light_level, 4.0), 0.0, 1200.0)
    dominant = dominant_user_id or user_id
    return MinuteRecord(
        timestamp=dt.strftime("%Y-%m-%d %H:%M:%S"),
        hour=dt.hour,
        minute=dt.minute,
        day_of_week=dt.weekday(),
        temperature=round(temp, 1),
        humidity=round(hum, 1),
        light_level=round(light, 1),
        motion_detected=int(motion),
        user_id=user_id,
        dominant_user_id=dominant,
        presence_type=presence_type,
        face_count=int(face_count),
        light_state=int(light_state),
        fan_state=int(fan_state),
        confidence=float(confidence),
        source=source,
        scenario_type=scenario_type,
        sequence_phase=_sequence_phase_for(dt, scenario_type, source, user_id, presence_type),
        override_flag=int(override_flag),
        override_reason=override_reason,
        session_id=session_id or "",
        decision_mode=normalize_decision_mode(decision_mode),
        data_source=data_source,
    )


def build_one_day_schedule(start_date: datetime) -> list[MinuteRecord]:
    rows: list[MinuteRecord] = []
    day_rng = _day_rng(start_date)
    weather_kind, weather_profile = _pick_weather_profile(day_rng)
    host_key, host_profile = _pick_host_profile(day_rng)

    # Mỗi mẫu cách nhau 2 phút để tạo đúng 720 mẫu/ngày.
    sample_interval_minutes = 2
    total_slots = 24 * 60 // sample_interval_minutes

    def add_series(start_minute: int, count: int, builder):
        for i in range(count):
            rows.append(builder(i, _ts(start_date, start_minute + i * sample_interval_minutes)))

    env_temp = day_rng.uniform(*weather_profile["temperature"])
    env_hum = day_rng.uniform(*weather_profile["humidity"])
    env_light = day_rng.uniform(*weather_profile["light_level"])

    def env(dt: datetime, base_temp: float, base_hum: float, base_light: float, motion: int, user_id: str,
            light_state: int, fan_state: int, session_id: str | None, presence_type: str, face_count: int,
            scenario_type: str, override_flag: int = 0, override_reason: str = "",
            source: str = "real",
            host_variant: str | None = None, dominant_user_id: str | None = None,
            decision_mode: str = DECISION_MODE_FALLBACK, data_source: str = "real"):
        nonlocal env_temp, env_hum, env_light
        hour = dt.hour + dt.minute / 60.0
        day_wave = 1.8 * __import__("math").sin((hour - 6.0) / 24.0 * 2.0 * __import__("math").pi)
        env_temp = _smooth_step(env_temp, _apply_day_variation(base_temp + day_wave, day_rng, 0.35), 0.18)
        env_hum = _smooth_step(env_hum, _apply_day_variation(base_hum - day_wave * 0.8, day_rng, 0.7), 0.22)
        env_light = _smooth_step(env_light, _apply_day_variation(base_light, day_rng, 12.0), 6.5)
        dt = _shift_timestamp_fields(dt, day_rng, max_minutes=1)
        host_variant = host_variant or host_key
        host_bias = HOST_PROFILES.get(host_variant, host_profile)
        if user_id == host_bias["user_id"]:
            if dt.hour >= 21:
                light_state = max(light_state, int(host_bias.get("night_light_pref", light_state)))
                fan_state = max(fan_state, int(host_bias.get("night_fan_pref", fan_state)))
            if scenario_type == "profile_personalization":
                light_state = max(light_state, int(host_bias.get("light_bias", light_state)))
                fan_state = max(fan_state, int(host_bias.get("fan_bias", fan_state)))

        dominant = dominant_user_id or user_id
        return _record(
            dt,
            env_temp,
            env_hum,
            env_light,
            motion,
            user_id,
            light_state,
            fan_state,
            session_id,
            presence_type,
            face_count,
            scenario_type,
            override_flag,
            override_reason,
            confidence=0.92 if scenario_type != "manual_override" else 0.78,
            decision_mode=decision_mode,
            dominant_user_id=dominant,
            data_source=data_source,
            source=source,
        )

    # 00:00-04:59 sleep/background: rất tối, ít chuyển động, nhiệt giảm dần.
    add_series(0, 120, lambda i, dt: env(dt, 26.4 - 0.004 * i, 77.5 + 0.01 * i, 0.0, 0, "sleeping", 0, 0, None, "empty", 0, "background_noise", 0, "sleep_background", "sleeping"))
    # 05:00-05:59 vẫn là nền nghỉ ngơi nhưng rút ngắn để giảm block empty quá dài.
    add_series(120, 60, lambda i, dt: env(dt, 26.0 - 0.003 * i, 77.0 + 0.008 * i, 1.5, 0, "sleeping", 0, 0, None, "empty", 0, "background_noise", 0, "sleep_background", "sleeping"))
    add_series(180, 36, lambda i, dt: env(dt, 25.8 + 0.01 * i, 76.8 - 0.01 * i, 3.0, 0, "unknown", 0, 0, None, "empty", 0, "background_noise", 0, "morning_idle", "sleeping"))

    # 06:00-07:29 sequence_behavior: người thức dậy, bật đèn trước, quạt sau.
    seq_morning = _make_session("seq", "user_a", _ts(start_date, 360))
    add_series(360, 8, lambda i, dt: env(dt, 25.8 + 0.05 * i, 80.0 - 0.05 * i, 8 + 2.0 * i, 1, host_profile["user_id"], 1, 0, seq_morning, host_profile["presence_type"], host_profile["face_count"], "sequence_behavior", 0, "arrive_light_on", "arrive_home", host_variant=host_key))
    add_series(368, 5, lambda i, dt: env(dt, 26.1 + 0.03 * i, 79.0 - 0.03 * i, 18 + 0.8 * i, 1, host_profile["user_id"], 1, 0, seq_morning, host_profile["presence_type"], host_profile["face_count"], "sequence_behavior", 0, "settle_before_fan", "settle_down", host_variant=host_key))
    add_series(373, 7, lambda i, dt: env(dt, 26.3 + 0.02 * i, 78.5 - 0.02 * i, 24 + 0.8 * i, 1, host_profile["user_id"], 1, 1, seq_morning, host_profile["presence_type"], host_profile["face_count"], "sequence_behavior", 0, "fan_on_after_delay", "turn_fan_on", host_variant=host_key))
    add_series(380, 18, lambda i, dt: env(dt, 26.6 + 0.02 * i, 78.0 - 0.02 * i, 28 + 1.0 * i, 1, "user_a", 1, 1, seq_morning, "single_known", 1, "sequence_behavior"))
    add_series(398, 8, lambda i, dt: env(dt, 27.0 + 0.01 * i, 77.8 - 0.02 * i, 36 + 1.0 * i, 1, "user_a", 1, 1, seq_morning, "single_known", 1, "sequence_behavior"))
    add_series(406, 8, lambda i, dt: env(dt, 27.2 + 0.01 * i, 77.5 - 0.02 * i, 42 + 1.0 * i, 1 if i < 5 else 0, "user_a", 1 if i < 6 else 0, 1, seq_morning, "single_known", 1, "sequence_behavior"))
    add_series(414, 18, lambda i, dt: env(dt, 27.4 + 0.01 * i, 77.0 - 0.02 * i, 55 + 1.1 * i, 1 if i < 6 else 0, "user_a", 1 if i < 10 else 0, 1, seq_morning, "single_known", 1, "sequence_behavior"))
    add_series(432, 10, lambda i, dt: env(dt, 28.0 + 0.02 * i, 76.8 - 0.02 * i, 66 + 1.0 * i, 1, "user_a", 1, 1 if i < 8 else 0, seq_morning, "single_known", 1, "sequence_behavior", 0, "warm_up_fan", "turn_fan_on"))

    # 07:30-09:29 rời nhà, nhà trống, tăng background_noise để giảm lệch class 1.
    add_series(450, 36, lambda i, dt: env(dt, 27.4 + 0.02 * i, 76.5 - 0.02 * i, 150 + 1.2 * i, 0, "unknown", 0, 0, None, "empty", 0, "background_noise", data_source="real"))
    add_series(486, 24, lambda i, dt: env(dt, 27.8 + 0.02 * i, 75.8 - 0.02 * i, 165 + 1.0 * i, 0, "unknown", 0, 0, None, "empty", 0, "background_noise", data_source="real"))
    add_series(510, 24, lambda i, dt: env(dt, 28.2 + 0.01 * i, 74.8 - 0.01 * i, 190 + 0.8 * i, 0, "unknown", 0, 0, None, "empty", 0, "background_noise", data_source="real"))

    # 09:30-11:29 background_noise có dao động tự nhiên của môi trường.
    add_series(534, 36, lambda i, dt: env(dt, 29.0 + 0.01 * i, 73.0 - 0.02 * i, 210 + 0.9 * i, 0 if i % 6 else 1, "unknown", 0 if i % 5 else 1, 0 if i % 4 else 1, None, "empty", 0, "background_noise", data_source="real"))
    add_series(570, 16, lambda i, dt: env(dt, 29.4 + 0.01 * i, 72.5 - 0.02 * i, 225 + 0.8 * i, 1 if i % 8 == 0 else 0, "user_b", 0 if i % 5 else 1, 0, None, "single_known", 1, "sequence_behavior", 0, "late_morning_visit", "arrive_home"))
    add_series(586, 16, lambda i, dt: env(dt, 29.6 + 0.01 * i, 72.0 - 0.02 * i, 235 + 0.7 * i, 1 if i % 7 == 0 else 0, "user_b", 0 if i % 4 else 1, 0 if i % 3 else 1, None, "single_known", 1, "profile_personalization", 0, "preference_check", "manual_adjust"))
    add_series(588, 12, lambda i, dt: env(dt, 30.1 + 0.01 * i, 71.5 - 0.02 * i, 275 + 0.8 * i, 1, "user_b", 1, 1 if i < 8 else 0, None, "single_known", 1, "context_exception", 0, "temperature_shift", "manual_adjust"))
    add_series(600, 36, lambda i, dt: env(dt, 30.0 + 0.01 * i, 71.2 - 0.02 * i, 280 + 0.8 * i, 1 if i % 10 == 0 else 0, "user_b", 1 if i % 6 == 0 else 0, 1 if i < 20 else 0, None, "single_known", 1, "sequence_behavior"))
    add_series(636, 24, lambda i, dt: env(dt, 26.5 - 0.02 * i, 82.0 + 0.01 * i, 250 + 0.6 * i, 1, "user_b", 0, 1 if i % 4 < 3 else 0, None, "single_known", 1, "context_exception"))

    # 11:30-12:59 context_exception: cùng khung giờ nhưng thiết bị đổi theo nóng/lạnh/ẩm.
    # Dùng các cặp đối chứng rõ hơn để XGBoost học "không chỉ nhìn giờ".
    add_series(690, 6, lambda i, dt: env(dt, 32.2 + 0.02 * i, 63.5 - 0.03 * i, 705 - 5 * i, 1, "user_b", 0, 1, None, "single_known", 1, "context_exception"))
    add_series(696, 6, lambda i, dt: env(dt, 31.4 + 0.02 * i, 67.0 - 0.02 * i, 680 - 5 * i, 1, "user_b", 0, 1, None, "single_known", 1, "context_exception"))
    add_series(702, 6, lambda i, dt: env(dt, 27.6 - 0.02 * i, 86.0 + 0.03 * i, 650 - 6 * i, 1, "user_b", 0, 0, None, "single_known", 1, "context_exception"))
    add_series(708, 6, lambda i, dt: env(dt, 27.2 - 0.02 * i, 87.0 + 0.03 * i, 640 - 6 * i, 1, "user_b", 0, 0, None, "single_known", 1, "context_exception"))
    add_series(714, 6, lambda i, dt: env(dt, 31.0 + 0.01 * i, 72.5 - 0.03 * i, 600 - 4 * i, 1 if i < 3 else 0, "user_b", 0, 1 if i < 4 else 0, None, "single_known", 1, "context_exception"))
    add_series(720, 6, lambda i, dt: env(dt, 27.8 - 0.02 * i, 84.5 + 0.03 * i, 565 - 4 * i, 1, "user_b", 0, 0, None, "single_known", 1, "context_exception"))
    add_series(726, 6, lambda i, dt: env(dt, 31.2 + 0.01 * i, 68.5 - 0.02 * i, 530 - 3 * i, 1, "user_b", 0, 1, None, "single_known", 1, "context_exception"))
    add_series(732, 6, lambda i, dt: env(dt, 27.4 - 0.02 * i, 85.5 + 0.03 * i, 510 - 3 * i, 1, "user_b", 0, 0, None, "single_known", 1, "context_exception"))
    add_series(738, 6, lambda i, dt: env(dt, 30.8 + 0.01 * i, 73.0 - 0.02 * i, 495 - 3 * i, 1 if i < 4 else 0, "user_b", 0, 1 if i < 5 else 0, None, "single_known", 1, "context_exception"))

    # 13:00-16:59 background + return visits.
    add_series(750, 18, lambda i, dt: env(dt, 30.2 + 0.02 * i, 68.0 - 0.02 * i, 520 - 3 * i, 0, "unknown", 0, 0, None, "empty", 0, "background_noise"))
    add_series(768, 18, lambda i, dt: env(dt, 30.4 + 0.01 * i, 67.5 - 0.02 * i, 500 + 1.2 * i, 0 if i % 3 else 1, "user_a", 0 if i % 4 else 1, 0 if i < 12 else 1, None, "single_known", 1, "profile_personalization", 0, "afternoon_preference", "manual_adjust"))
    add_series(786, 18, lambda i, dt: env(dt, 30.6 - 0.02 * i, 66.8 - 0.03 * i, 470 + 1.4 * i, 1, "user_a", 1 if i % 4 else 0, 0 if i < 10 else 1, None, "single_known", 1, "sequence_behavior"))
    add_series(804, 16, lambda i, dt: env(dt, 29.8 - 0.02 * i, 66.5 - 0.03 * i, 420 + 1.4 * i, 1, "user_b", 0 if i < 7 else 1, 0 if i < 10 else 1, None, "single_known", 1, "sequence_behavior"))
    seq_afternoon = _make_session("seq", "user_a", _ts(start_date, 820))
    add_series(820, 14, lambda i, dt: env(dt, 30.8 - 0.03 * i, 66.0 - 0.03 * i, 410 + 2 * i, 1, "user_a", 1, 0 if i < 4 else 1, seq_afternoon, "single_known", 1, "sequence_behavior"))
    add_series(834, 10, lambda i, dt: env(dt, 30.9 - 0.02 * i, 65.0 - 0.03 * i, 440 + 2 * i, 1, "user_a", 1, 1, seq_afternoon, "single_known", 1, "sequence_behavior"))
    add_series(844, 14, lambda i, dt: env(dt, 31.0 - 0.02 * i, 65.0 - 0.02 * i, 460 + 2 * i, 1 if i < 5 else 0, "user_a", 1, 1 if i < 8 else 0, seq_afternoon, "single_known", 1, "sequence_behavior"))
    add_series(858, 10, lambda i, dt: env(dt, 31.3 - 0.01 * i, 64.0 - 0.01 * i, 360 - 2 * i, 0 if i % 6 else 1, "unknown", 0, 0, None, "empty", 0, "background_noise"))

    # 17:00-19:00 sequence_behavior: về nhà, bật đèn trước rồi quạt sau 5-10 phút.
    seq_evening = _make_session("seq", "user_a", _ts(start_date, 1020))
    add_series(1020, 6, lambda i, dt: env(dt, 30.9 - 0.04 * i, 61.0 - 0.02 * i, 160 - 2 * i, 1, "user_a", 1, 0, seq_evening, "single_known", 1, "sequence_behavior", 0, "arrive_light_on"))
    add_series(1026, 6, lambda i, dt: env(dt, 30.8 - 0.02 * i, 60.8 - 0.02 * i, 155 - 1.5 * i, 1, "user_a", 1, 1 if i < 4 else 0, seq_evening, "single_known", 1, "sequence_behavior", 0, "settle_before_fan"))
    add_series(1032, 6, lambda i, dt: env(dt, 30.7 - 0.02 * i, 60.6 - 0.01 * i, 150 - 1.2 * i, 1, "user_a", 1, 1 if i < 4 else 0, seq_evening, "single_known", 1, "sequence_behavior", 0, "settle_before_fan"))
    add_series(1038, 8, lambda i, dt: env(dt, 30.5 - 0.02 * i, 60.2 - 0.01 * i, 142 - 1.2 * i, 1, "user_a", 1, 1, seq_evening, "single_known", 1, "sequence_behavior", 0, "fan_on_after_delay"))
    add_series(1046, 10, lambda i, dt: env(dt, 30.3 - 0.02 * i, 59.9 - 0.01 * i, 136 - 1.0 * i, 1, "user_a", 1, 1, seq_evening, "single_known", 1, "sequence_behavior"))
    add_series(1056, 14, lambda i, dt: env(dt, 30.2 - 0.02 * i, 59.8 - 0.01 * i, 130 - 1.0 * i, 1, "user_a", 1, 1, seq_evening, "single_known", 1, "sequence_behavior"))
    add_series(1070, 10, lambda i, dt: env(dt, 30.0 - 0.02 * i, 59.5 - 0.01 * i, 122 - 0.9 * i, 1, "user_a", 1, 1 if i < 9 else 0, seq_evening, "single_known", 1, "sequence_behavior", 0, "late_evening_cooling", "turn_fan_on"))

    # 18:20-19:50 profile_personalization: ưu tiên user B khi cùng hiện diện.
    add_series(1100, 6, lambda i, dt: env(dt, 30.0 - 0.03 * i, 57.5 - 0.02 * i, 100 - 1.0 * i, 1, "user_a", 1, 1, None, "single_known", 1, "profile_personalization", 0, "user_a_preference", "manual_adjust"))
    add_series(1106, 6, lambda i, dt: env(dt, 29.8 - 0.02 * i, 56.5 - 0.01 * i, 92 - 0.8 * i, 1, "user_b", 1, 0, None, "single_known", 1, "profile_personalization", 0, "user_b_preference", "manual_adjust"))
    add_series(1112, 8, lambda i, dt: env(dt, 29.6 - 0.01 * i, 56.0, 84 - 0.5 * i, 1, "user_a,user_b", 1, 0, None, "multi_user", 2, "profile_personalization", 0, "multi_user_priority", "manual_adjust"))
    add_series(1120, 6, lambda i, dt: env(dt, 29.5 - 0.02 * i, 55.5, 76 - 0.4 * i, 1, "user_b", 1, 0, None, "single_known", 1, "profile_personalization", 0, "user_b_preference", "manual_adjust"))
    add_series(1126, 8, lambda i, dt: env(dt, 29.4 - 0.02 * i, 55.0, 68 - 0.4 * i, 1, "user_a,user_b", 1, 0, None, "multi_user", 2, "profile_personalization", 0, "multi_user_priority", "manual_adjust"))
    add_series(1134, 6, lambda i, dt: env(dt, 29.3 - 0.02 * i, 55.0, 65 - 0.4 * i, 1, "user_b", 1, 0, None, "single_known", 1, "profile_personalization", 0, "user_b_preference", "manual_adjust"))
    add_series(1140, 8, lambda i, dt: env(dt, 29.2 - 0.02 * i, 54.8, 62 - 0.3 * i, 1, "user_a,user_b", 1, 0, None, "multi_user", 2, "profile_personalization", 0, "multi_user_priority", "manual_adjust"))

    # 20:00-21:39 manual_override: can thiệp thủ công dày hơn để tăng coverage.
    override_session = _make_session("ovr", "user_a", _ts(start_date, 1200))
    add_series(1200, 8, lambda i, dt: env(dt, 29.4 - 0.02 * i, 54.8, 72 - 0.4 * i, 1, "user_a", 1, 0 if i < 4 else 1, override_session, "single_known", 1, "manual_override", 1, "ai_prefetch", "manual_override"))
    add_series(1208, 8, lambda i, dt: env(dt, 29.2 - 0.02 * i, 54.6, 70 - 0.4 * i, 1, "user_a", 1, 0 if i < 2 else 1, override_session, "single_known", 1, "manual_override", 1, "ai_prefetch", "manual_override"))
    add_series(1216, 10, lambda i, dt: env(dt, 29.0 - 0.02 * i, 54.4, 68 - 0.3 * i, 1, "user_a", 0 if i >= 2 else 1, 0, override_session, "single_known", 1, "manual_override", 1 if i >= 2 else 0, "watch_movie", "manual_adjust"))
    add_series(1226, 10, lambda i, dt: env(dt, 28.9 - 0.02 * i, 54.2, 65 - 0.3 * i, 1 if i < 3 else 0, "user_a", 0 if i < 5 else 1, 0 if i < 6 else 1, override_session, "single_known", 1, "manual_override", 1, "watch_movie", "manual_adjust"))
    add_series(1236, 8, lambda i, dt: env(dt, 28.7 - 0.01 * i, 53.6, 60 - 0.3 * i, 0, "user_a", 0, 1 if i < 5 else 0, None, "profile_personalization", 1, "profile_personalization", 0, "personal_mood", "manual_adjust"))
    add_series(1244, 10, lambda i, dt: env(dt, 28.6 - 0.01 * i, 53.2, 56 - 0.2 * i, 0 if i < 6 else 1, "user_a", 0 if i < 6 else 1, 1 if i < 4 else 0, None, "single_known", 1, "context_exception", 0, "temperature_shift", "manual_adjust"))

    # Tăng thêm một cụm override ngắn vào đầu tối để tránh lệch lớp 1.
    second_override_session = _make_session("ovr", "user_b", _ts(start_date, 1110))
    add_series(1110, 10, lambda i, dt: env(dt, 29.7 - 0.02 * i, 56.0, 88 - 0.5 * i, 1, "user_b", 1, 0 if i < 5 else 1, second_override_session, "single_known", 1, "manual_override", 1, "manual_adjust", "manual_adjust"))
    add_series(1120, 10, lambda i, dt: env(dt, 29.5 - 0.02 * i, 55.8, 84 - 0.4 * i, 1, "user_b", 0 if i >= 2 else 1, 1 if i >= 6 else 0, second_override_session, "single_known", 1, "manual_override", 1 if i >= 2 else 0, "manual_adjust", "manual_adjust"))

    # 21:00-22:29 profile_personalization + context_exception: chuẩn bị ngủ, thi thoảng phòng trống/đổi thói quen.
    add_series(1254, 10, lambda i, dt: env(dt, 27.8 - 0.02 * i, 51.5, 45 - 0.2 * i, 0, "user_b", 0, 0, None, "single_known", 1, "profile_personalization", 0, "user_b_winddown", "manual_adjust"))
    add_series(1264, 10, lambda i, dt: env(dt, 27.5 - 0.01 * i, 50.5, 35 - 0.1 * i, 0, "user_a,user_b", 0, 0, None, "multi_user", 2, "profile_personalization", 0, "multi_user_winddown", "manual_adjust"))
    add_series(1274, 10, lambda i, dt: env(dt, 27.4 - 0.01 * i, 50.2, 32 - 0.1 * i, 0, "user_c", 0, 0, None, "single_known", 1, "context_exception", 0, "context_cooldown", "manual_adjust"))

    # 22:30-23:59 background noise / sleep.
    add_series(1272, 6, lambda i, dt: env(dt, 27.3 - 0.01 * i, 50.0 + 0.03 * i, 25 - 0.1 * i, 0, "sleeping", 0, 0, None, "empty", 0, "background_noise", 0, "background_noise", "sleeping"))
    add_series(1278, 6, lambda i, dt: env(dt, 27.2 - 0.005 * i, 51.0 + 0.01 * i, 20 - 0.05 * i, 0, "sleeping", 0, 0, None, "empty", 0, "background_noise", 0, "background_noise", "sleeping"))
    add_series(1284, 6, lambda i, dt: env(dt, 27.1 - 0.005 * i, 51.5 + 0.01 * i, 18 - 0.05 * i, 0, "sleeping", 0, 0, None, "empty", 0, "background_noise", 0, "background_noise", "sleeping"))
    add_series(1290, 6, lambda i, dt: env(dt, 27.0 - 0.005 * i, 52.0 + 0.01 * i, 16 - 0.05 * i, 0, "sleeping", 0, 0, None, "empty", 0, "background_noise", 0, "background_noise", "sleeping"))

    rows.sort(key=lambda r: r.timestamp)

    if len(rows) > total_slots:
        rows = rows[:total_slots]
    elif len(rows) < total_slots and rows:
        last = rows[-1]
        while len(rows) < total_slots:
            clone = MinuteRecord(
                timestamp=last.timestamp,
                hour=last.hour,
                minute=last.minute,
                day_of_week=last.day_of_week,
                temperature=last.temperature,
                humidity=last.humidity,
                light_level=last.light_level,
                motion_detected=last.motion_detected,
                user_id=last.user_id,
                light_state=last.light_state,
                fan_state=last.fan_state,
                session_id=last.session_id,
                presence_type=last.presence_type,
                face_count=last.face_count,
                scenario_type=last.scenario_type,
                sequence_phase=last.sequence_phase,
                data_source=last.data_source,
                source=last.source,
                override_flag=last.override_flag,
                override_reason=last.override_reason,
                confidence=last.confidence,
            )
            rows.append(clone)

    rows.sort(key=lambda r: r.timestamp)
    return rows


def save_one_day_to_db(db: Database, rows: Iterable[MinuteRecord]) -> None:
    rows = list(rows)
    if not rows:
        return

    # Xóa toàn bộ dữ liệu real cũ nhưng chỉ cho ngày đã sinh để giữ DB sạch.
    # Dữ liệu train chính của generator này nên mang tính real-like thay vì simulated.
    target_date = rows[0].timestamp[:10]
    with db.get_connection() as conn:
        env_ids = [row[0] for row in conn.execute(
            """
            SELECT id
            FROM environmental_logs
            WHERE timestamp LIKE ? || '%'
              AND id IN (
                    SELECT env_log_id
                    FROM device_states
                    WHERE source IN ('real', 'manual_override', 'manual')
              )
            """,
            (target_date,),
        ).fetchall()]

        if env_ids:
            placeholders = ",".join(["?"] * len(env_ids))
            conn.execute(f"DELETE FROM prediction_feedback_queue WHERE env_log_id IN ({placeholders})", env_ids)
            conn.execute(f"DELETE FROM predictions WHERE env_log_id IN ({placeholders})", env_ids)
            conn.execute(f"DELETE FROM device_states WHERE env_log_id IN ({placeholders})", env_ids)
            conn.execute(f"DELETE FROM environmental_logs WHERE id IN ({placeholders})", env_ids)

        for row in rows:
            cur = conn.execute(
                """
                INSERT INTO environmental_logs
                    (timestamp, hour, minute, day_of_week, temperature, humidity, light_level, motion_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row.timestamp, row.hour, row.minute, row.day_of_week, row.temperature, row.humidity, row.light_level, row.motion_detected),
            )
            env_log_id = cur.lastrowid

            conn.execute(
                """
                INSERT INTO device_states
                    (timestamp, env_log_id, dominant_user_id, decision_mode, user_id, light_state, fan_state, source, data_source,
                     scenario_type, sequence_phase, override_flag, override_reason, session_id,
                     presence_type, face_count, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.timestamp,
                    env_log_id,
                    row.dominant_user_id,
                    row.decision_mode,
                    row.user_id,
                    row.light_state,
                    row.fan_state,
                    getattr(row, "source", "real"),
                    getattr(row, "data_source", "real"),
                    getattr(row, "scenario_type", "background_noise"),
                    getattr(row, "sequence_phase", "background_noise"),
                    row.override_flag,
                    row.override_reason,
                    row.session_id,
                    row.presence_type,
                    row.face_count,
                    0.90 if row.scenario_type == "manual_override" else 0.97,
                ),
            )

            # Tạo thêm dòng prediction cho các mẫu có tính học phản hồi hoặc cần thể hiện logic AI.
            confidence = 0.92
            if row.scenario_type == "manual_override":
                confidence = 0.78
            elif row.scenario_type == "context_exception":
                confidence = 0.84
            elif row.scenario_type == "sequence_behavior":
                confidence = 0.89
            elif row.scenario_type == "profile_personalization":
                confidence = 0.87

            actual_light = row.light_state if row.scenario_type == "manual_override" else None
            actual_fan = row.fan_state if row.scenario_type == "manual_override" else None
            overridden = 1 if row.override_flag else 0
            if row.scenario_type in {"manual_override", "sequence_behavior", "context_exception", "profile_personalization"}:
                conn.execute(
                    """
                    INSERT INTO predictions
                        (timestamp, env_log_id, user_id, pred_light, pred_fan, confidence, executed,
                         actual_light, actual_fan, overridden, override_reason, feedback_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.timestamp,
                        env_log_id,
                        row.user_id,
                        row.light_state,
                        row.fan_state,
                        confidence,
                        1,
                        actual_light,
                        actual_fan,
                        overridden,
                        row.override_reason,
                        row.scenario_type,
                    ),
                )
                if row.override_flag:
                    conn.execute(
                        """
                        INSERT INTO prediction_feedback_queue
                            (env_log_id, user_id, source, predicted_light, predicted_fan, actual_light,
                             actual_fan, override_reason, feedback_source, processed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            env_log_id,
                            row.user_id,
                            "manual_override",
                            row.light_state,
                            row.fan_state,
                            row.light_state,
                            row.fan_state,
                            row.override_reason,
                            row.scenario_type,
                            0,
                        ),
                    )


def _normalize_day_rows(
    rows: list[MinuteRecord],
    day_start: datetime,
    total_slots: int,
    sample_interval_minutes: int,
    rng: random.Random,
    max_jitter_minutes: int = 2,
) -> list[MinuteRecord]:
    rows = sorted(rows, key=lambda r: r.timestamp)
    if len(rows) > total_slots:
        rows = rows[:total_slots]
    elif len(rows) < total_slots and rows:
        last = rows[-1]
        while len(rows) < total_slots:
            rows.append(MinuteRecord(
                timestamp=last.timestamp,
                hour=last.hour,
                minute=last.minute,
                day_of_week=last.day_of_week,
                temperature=last.temperature,
                humidity=last.humidity,
                light_level=last.light_level,
                motion_detected=last.motion_detected,
                user_id=last.user_id,
                light_state=last.light_state,
                fan_state=last.fan_state,
                session_id=last.session_id,
                presence_type=last.presence_type,
                face_count=last.face_count,
                scenario_type=last.scenario_type,
                sequence_phase=last.sequence_phase,
                data_source=last.data_source,
                source=last.source,
                override_flag=last.override_flag,
                override_reason=last.override_reason,
                confidence=last.confidence,
            ))

    normalized: list[MinuteRecord] = []
    prev_dt = day_start - timedelta(minutes=sample_interval_minutes)
    for idx, row in enumerate(rows):
        base_dt = day_start + timedelta(minutes=idx * sample_interval_minutes)
        jitter = rng.randint(-max_jitter_minutes, max_jitter_minutes)
        dt = base_dt + timedelta(minutes=jitter)
        min_dt = prev_dt + timedelta(minutes=1)
        max_dt = day_start + timedelta(days=1) - timedelta(minutes=1)
        if dt < min_dt:
            dt = min_dt
        if dt > max_dt:
            dt = max_dt
        row.timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        row.hour = dt.hour
        row.minute = dt.minute
        row.day_of_week = dt.weekday()
        normalized.append(row)
        prev_dt = dt
    return normalized


def _load_summary(rows: list[MinuteRecord]) -> dict:
    scenario_counts: dict[str, int] = {}
    light_counts = {0: 0, 1: 0}
    fan_counts = {0: 0, 1: 0}
    override_rows = 0
    for row in rows:
        scenario_counts[row.scenario_type] = scenario_counts.get(row.scenario_type, 0) + 1
        light_counts[int(row.light_state)] += 1
        fan_counts[int(row.fan_state)] += 1
        override_rows += int(row.override_flag)
    return {
        "total_rows": len(rows),
        "scenario_counts": scenario_counts,
        "light_counts": light_counts,
        "fan_counts": fan_counts,
        "override_rows": override_rows,
    }


def _export_core_csv(rows: list[MinuteRecord], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    import csv as _csv

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "timestamp", "hour", "minute", "day_of_week",
            "temperature", "humidity", "light_level", "motion_detected",
            "user_id", "dominant_user_id", "presence_type", "face_count",
            "light_state", "fan_state", "confidence", "source", 
            "scenario_type", "sequence_phase", "override_flag", 
            "override_reason", "session_id", "decision_mode" 
            
        ])
        for row in rows:
            writer.writerow([
                row.timestamp, row.hour, row.minute, row.day_of_week,
                row.temperature, row.humidity, row.light_level, row.motion_detected,
                row.user_id, row.dominant_user_id, row.presence_type, row.face_count,
                row.light_state, row.fan_state, row.confidence, getattr(row, "source", "simulated"),
                row.scenario_type, getattr(row, "sequence_phase", "background_noise"), row.override_flag, 
                row.override_reason, row.session_id, row.decision_mode
              
            ])


def generate_one_day_dataset(start_date: datetime | None = None, db_path: str | None = None) -> list[MinuteRecord]:
    start_date = start_date or datetime(2026, 5, 3, 0, 0, 0)
    db = Database(db_path=db_path or config.DATABASE_PATH)
    sample_interval_minutes = 2
    total_slots = (24 * 60) // sample_interval_minutes
    rows = _normalize_day_rows(
        build_one_day_schedule(start_date),
        start_date,
        total_slots,
        sample_interval_minutes,
        _day_rng(start_date),
        max_jitter_minutes=1,
    )
    save_one_day_to_db(db, rows)
    summary = _load_summary(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    data_dir = getattr(config, 'DATA_DIR', os.path.dirname(config.DATABASE_PATH) if hasattr(config, 'DATABASE_PATH') else 'data')
    csv_path = os.path.join(data_dir, 'manual_core_data.csv')
    _export_core_csv(rows, csv_path)
    print(f"  ✅ Đã xuất CSV core: {csv_path}")
    return rows


def generate_multiple_days_dataset(
    days: int,
    start_date: datetime | None = None,
    db_path: str | None = None,
    randomize_each_day: bool = True,
    time_noise_minutes: int = 1,
) -> list[MinuteRecord]:
    if days < 1:
        raise ValueError("days phải >= 1")

    all_rows: list[MinuteRecord] = []
    base_date = start_date or datetime(2026, 2, 1, 0, 0, 0)
    db = Database(db_path=db_path or config.DATABASE_PATH)
    prev_end_dt: datetime | None = None

    for i in range(days):
        day_start = base_date + timedelta(days=i)
        day_rows = build_one_day_schedule(day_start)
        daily_rng = _day_rng(day_start)

        varied_rows: list[MinuteRecord] = []
        for row in day_rows:
            row_dt = datetime.strptime(row.timestamp, "%Y-%m-%d %H:%M:%S")
            if randomize_each_day and time_noise_minutes > 0:
                row_dt = row_dt + timedelta(minutes=daily_rng.randint(-time_noise_minutes, time_noise_minutes))
            row.temperature = round(_clamp(row.temperature + (daily_rng.uniform(-0.4, 0.4) if randomize_each_day else 0.0), 18.0, 38.0), 1)
            row.humidity = round(_clamp(row.humidity + (daily_rng.uniform(-0.8, 0.8) if randomize_each_day else 0.0), 20.0, 95.0), 1)
            row.light_level = round(_clamp(row.light_level + (daily_rng.uniform(-10.0, 10.0) if randomize_each_day else 0.0), 0.0, 1200.0), 1)
            row.motion_detected = int(row.motion_detected if (not randomize_each_day or daily_rng.random() > 0.06) else 1 - row.motion_detected)
            row.timestamp = row_dt.strftime("%Y-%m-%d %H:%M:%S")
            row.hour = row_dt.hour
            row.minute = row_dt.minute
            row.day_of_week = row_dt.weekday()
            varied_rows.append(row)

        day_rows = _normalize_day_rows(
            varied_rows,
            day_start,
            (24 * 60) // 2,
            2,
            daily_rng,
            max_jitter_minutes=max(0, time_noise_minutes),
        )

        if prev_end_dt is not None:
            day_start_dt = datetime.strptime(day_rows[0].timestamp, "%Y-%m-%d %H:%M:%S")
            desired_start = prev_end_dt + timedelta(minutes=2)
            shift = desired_start - day_start_dt
            shifted_rows: list[MinuteRecord] = []
            for idx, row in enumerate(day_rows):
                row_dt = datetime.strptime(row.timestamp, "%Y-%m-%d %H:%M:%S") + shift
                row_dt = desired_start + timedelta(minutes=idx * 2)
                row.timestamp = row_dt.strftime("%Y-%m-%d %H:%M:%S")
                row.hour = row_dt.hour
                row.minute = row_dt.minute
                row.day_of_week = row_dt.weekday()
                shifted_rows.append(row)
            day_rows = shifted_rows

        all_rows.extend(day_rows)
        save_one_day_to_db(db, day_rows)
        prev_end_dt = datetime.strptime(day_rows[-1].timestamp, "%Y-%m-%d %H:%M:%S")

    summary = _load_summary(all_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    data_dir = getattr(config, 'DATA_DIR', os.path.dirname(config.DATABASE_PATH) if hasattr(config, 'DATABASE_PATH') else 'data')
    csv_path = os.path.join(data_dir, 'manual_core_data.csv')
    _export_core_csv(all_rows, csv_path)
    print(f"  ✅ Đã xuất CSV core: {csv_path}")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh dữ liệu mô phỏng cho smart_home.db")
    parser.add_argument("--days", type=int, default=1, help="Số ngày muốn sinh dữ liệu")
    parser.add_argument("--db-path", type=str, default=None, help="Đường dẫn database")
    parser.add_argument("--start-date", type=str, default="2026-02-01", help="Ngày bắt đầu YYYY-MM-DD")
    parser.add_argument("--daily-randomize", action="store_true", help="Bật random nhẹ giữa các ngày")
    parser.add_argument("--time-noise-minutes", type=int, default=1, help="Sai số thời gian tối đa theo phút")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.days == 1:
        rows = generate_one_day_dataset(start_date=start_date, db_path=args.db_path)
        print(f"Đã sinh và lưu {len(rows)} mẫu 1 ngày vào smart_home.db")
    else:
        rows = generate_multiple_days_dataset(
            days=args.days,
            start_date=start_date,
            db_path=args.db_path,
            randomize_each_day=args.daily_randomize,
            time_noise_minutes=max(0, args.time_noise_minutes),
        )
        print(f"Đã sinh và lưu {len(rows)} mẫu cho {args.days} ngày vào smart_home.db")
