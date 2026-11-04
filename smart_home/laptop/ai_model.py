
import os
import json
import joblib
import logging
import threading
import random
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report, precision_score, recall_score

AI_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "temperature",
    "humidity",
    "light_level",
    "motion_detected",
    "temp_delta",
    "humidity_delta",
    "light_delta",
    "motion_streak",
    "time_bucket",
    "is_weekend",
    "presence_encoded",
    "scenario_encoded",
    "sequence_phase_encoded",
    "face_count",
    "face_count_delta",
    "override_flag",
    "last_light_state",
    "last_fan_state",
    "user_id_encoded",
]

TRAINING_SCHEMA_VERSION = 2

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from modules.decision_coordinator import DeviceDecision
from modules.rule_engine import RuleEngine

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Embedding, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.hysteresis_controller import HysteresisController

logger = logging.getLogger(__name__)


class SmartHomeAI:
    def __init__(self, db):
        self.db = db

        self.manual_override_lockout = {}
        self.manual_override_cooldown_sec = getattr(config, "MANUAL_OVERRIDE_COOLDOWN_SEC", 1800)
        self.manual_override_shadow_sec = getattr(config, "MANUAL_OVERRIDE_SHADOW_SEC", 600)
        self.sequence_resample_freq_sec = getattr(config, "RESAMPLE_FREQ_SEC", 30)

        self.lstm_model    = None
        self.fe_light      = None
        self.fe_fan        = None

        self.xgb_light     = None
        self.xgb_fan       = None
        self.xgb_scenario   = None
        self.xgb_override   = None

        self.scaler        = None
        self.label_encoder = None
        self.is_trained    = False
        self._lock         = threading.Lock()
        self.hysteresis    = HysteresisController()

        self.TIME_STEPS        = getattr(config, "LSTM_TIME_STEPS",    5)
        self.LSTM_EPOCHS       = getattr(config, "LSTM_EPOCHS",       30)
        self.LSTM_BATCH_SIZE   = getattr(config, "LSTM_BATCH_SIZE",   32)
        self.XGB_N_ESTIMATORS  = getattr(config, "XGB_N_ESTIMATORS", 100)
        self.XGB_MAX_DEPTH     = getattr(config, "XGB_MAX_DEPTH",      5)
        self.XGB_LEARNING_RATE = getattr(config, "XGB_LEARNING_RATE", 0.1)
        self.LIGHT_CONF_THRESH = getattr(config, "LIGHT_CONF_THRESH", 0.55)
        self.FAN_CONF_THRESH   = getattr(config, "FAN_CONF_THRESH",   0.55)
        self.LIGHT_POS_THRESH  = getattr(config, "LIGHT_POS_THRESH",  0.30)

        # Per-user state
        self.user_buffers      = {}   # {user_id: [scaled_feature_vectors]}
        self.last_predict_time = {}   # {user_id: datetime}
        self.last_fan_state    = {}   # {user_id: 0|1}
        self.last_light_state  = {}   # {user_id: 0|1}
        self._user_state_last_seen = {}   # {user_id: monotonic timestamp}
        self._user_state_ttl_sec = getattr(config, "USER_STATE_TTL_SEC", 3600)
        self.hysteresis = HysteresisController()
        self.rule_engine = RuleEngine()

        self._load_model()

    def _safe_json_get(self, row: dict, key: str, default=None):
        value = row.get(key, default)
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
        return value

    def _is_manual_override_active(self, user_id: str, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        normalized_user = self._normalize_user_label(user_id)
        lock = self.manual_override_lockout.get(normalized_user)
        if not lock:
            return False
        expires_at = lock.get("expires_at")
        if expires_at and now >= expires_at:
            self.manual_override_lockout.pop(normalized_user, None)
            return False
        return True

    def _set_manual_override_lockout(self, user_id: str, reason: str = "manual_override") -> None:
        now = datetime.now()
        normalized_user = self._normalize_user_label(user_id)
        self.manual_override_lockout[normalized_user] = {
            "started_at": now,
            "expires_at": now + timedelta(seconds=int(self.manual_override_cooldown_sec)),
            "shadow_until": now + timedelta(seconds=int(self.manual_override_shadow_sec)),
            "reason": reason,
        }

    def _allow_ai_actuation_for_user(self, user_id: str, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        normalized_user = self._normalize_user_label(user_id)
        lock = self.manual_override_lockout.get(normalized_user)
        if not lock:
            return True
        if lock.get("expires_at") and now >= lock["expires_at"]:
            self.manual_override_lockout.pop(normalized_user, None)
            return True
        return False

    def _normalize_timestamp(self, value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return datetime.now()

    def _infer_scenario_type(self, row: dict) -> str:
        scenario = (row.get("scenario_type") or "").strip().lower()
        source = (row.get("source") or "").strip().lower()
        data_source = (row.get("data_source") or "").strip().lower()
        override_flag = int(row.get("override_flag", 0) or 0)
        presence = (row.get("presence_type") or "unknown").strip().lower()

        if scenario in {"sequence_behavior", "manual_override", "background_noise", "profile_personalization", "context_exception"}:
            return scenario

        if override_flag == 1 or "manual" in source:
            return "manual_override"

        if "profile" in source:
            return "profile_personalization"

        if presence in {"empty", "unknown", "animal"}:
            return "background_noise"

        if "context" in scenario:
            return "context_exception"

        if data_source == "simulated":
            return scenario or "sequence_behavior"

        return "sequence_behavior"

    def _infer_sequence_phase(self, row: dict) -> str:
        phase = (row.get("sequence_phase") or "").strip().lower()
        if phase in {"sequence_idle", "override_response", "background_noise", "profile_multi_user", "context_transition"}:
            return phase

        scenario = (row.get("scenario_type") or "").strip().lower()
        source = (row.get("source") or "").strip().lower()
        override_flag = int(row.get("override_flag", 0) or 0)

        if override_flag == 1 or "manual" in source:
            return "override_response"
        if scenario == "context_exception":
            return "context_transition"
        if scenario == "manual_override":
            return "override_response"
        if scenario == "profile_personalization":
            return "profile_multi_user"
        if scenario == "background_noise":
            return "background_noise"
        return "sequence_idle"

    def _normalize_user_label(self, value: str | None) -> str:
        text = (value or "unknown")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip().lower() or "unknown"
        if text == "guest":
            return "global_average"
        if text in {"", "none", "null"}:
            return "unknown"
        return text

    def _touch_user_state(self, user_id: str) -> None:
        with self._lock:
            normalized_user = self._normalize_user_label(user_id)
            self._user_state_last_seen[normalized_user] = time.monotonic()

    def _cleanup_stale_user_state(self, now_monotonic: float | None = None) -> None:
        now_monotonic = now_monotonic or time.monotonic()
        ttl = max(int(self._user_state_ttl_sec), 60)
        stale_users = []
        with self._lock:
            for user_id, last_seen in list(self._user_state_last_seen.items()):
                if now_monotonic - last_seen >= ttl:
                    stale_users.append(user_id)
            for user_id in stale_users:
                self._user_state_last_seen.pop(user_id, None)
                self.last_predict_time.pop(user_id, None)
                self.last_fan_state.pop(user_id, None)
                self.last_light_state.pop(user_id, None)
                self.user_buffers.pop(user_id, None)
        if stale_users:
            logger.info(f"[AI] Đã dọn sạch state của {len(stale_users)} user hết TTL.")

    def _row_to_training_item(self, row: dict) -> dict | None:
        try:
            dt = self._normalize_timestamp(row.get("timestamp"))
            user_id = self._normalize_user_label(row.get("user_id") or row.get("face_id") or "unknown")
            dominant_user_id = self._normalize_user_label(row.get("dominant_user_id") or user_id)
            label_conflict = int(user_id != dominant_user_id)
            temp = float(row.get("temperature", row.get("temp", 25.0)))
            hum = float(row.get("humidity", row.get("hum", 60.0)))
            light = float(row.get("light_level", row.get("light", 0.0)))
            motion = int(row.get("motion_detected", row.get("motion", 0)))
            motion_streak = int(row.get("motion_streak", 1 if motion else 0))
            time_bucket = int(row.get("time_bucket", dt.hour // 3))
            item = {
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "hour": int(row.get("hour", dt.hour)),
                "day_of_week": int(row.get("day_of_week", dt.weekday())),
                "temperature": temp,
                "humidity": hum,
                "light_level": light,
                "motion_detected": motion,
                "temp_delta": float(row.get("temp_delta", 0.0) or 0.0),
                "humidity_delta": float(row.get("humidity_delta", 0.0) or 0.0),
                "light_delta": float(row.get("light_delta", 0.0) or 0.0),
                "motion_streak": motion_streak,
                "time_bucket": time_bucket,
                "is_weekend": int(int(row.get("day_of_week", dt.weekday())) in (5, 6)),
                "user_id": user_id,
                "dominant_user_id": dominant_user_id,
                "light_state": int(row.get("light_state", 0)),
                "fan_state": int(row.get("fan_state", 0)),
                "scenario_type": self._infer_scenario_type(row),
                "sequence_phase": self._infer_sequence_phase(row),
                "override_flag": int(row.get("override_flag", 0)),
                "override_reason": (row.get("override_reason") or "").strip(),
                "data_source": (row.get("data_source") or "real").strip().lower(),
                "source": (row.get("source") or "auto").strip().lower(),
                "presence_type": (row.get("presence_type") or "unknown").strip().lower(),
                "face_count": int(row.get("face_count", 0)),
                "session_id": (row.get("session_id") or "").strip(),
                "confidence": float(row.get("confidence", 0.0)),
                "label_conflict": label_conflict,
            }
            return item
        except Exception as e:
            logger.warning(f"[AI] Bỏ qua 1 row không hợp lệ: {e}")
            return None

    # ─────────────────────────────────────────────────────
    def reset_context(self, user_id: str = None):
        """Xóa ngữ cảnh cũ để tránh user trước ảnh hưởng user sau."""
        with self._lock:
            if user_id in (None, "guest"):
                self.user_buffers.clear()
                self.last_predict_time.clear()
                self.last_fan_state.clear()
                self.last_light_state.clear()
                logger.info("[AI] Đã reset toàn bộ context người dùng.")
                return

            self.user_buffers.pop(user_id, None)
            self.last_predict_time.pop(user_id, None)
            self.last_fan_state.pop(user_id, None)
            self.last_light_state.pop(user_id, None)
            logger.info(f"[AI] Đã reset context cho user={user_id}.")

    # ─────────────────────────────────────────────────────
    def warmup_buffer(self, recent_logs: list):
        """Nạp dữ liệu lịch sử vào per-user buffer khi khởi động hoặc sau train."""
        with self._lock:
            if not self.is_trained or self.scaler is None or self.label_encoder is None:
                return
            current_scaler  = self.scaler
            current_encoder = self.label_encoder

        interval = getattr(config, "INFERENCE_INTERVAL_SEC", 120)
        max_gap  = getattr(config, "MAX_SEQUENCE_GAP_SEC",   300)

        user_features_temp = {}
        last_ts_by_user    = {}
        sorted_logs = sorted(recent_logs, key=lambda r: r.get('timestamp', ''), reverse=False)

        for r in sorted_logs:
            try:
                item = self._row_to_training_item(r)
                if item is None:
                    continue

                dt = self._normalize_timestamp(item["timestamp"])
                uid = item.get("user_id", "unknown")

                # [BẢN VÁ YOLO] Không nạp buffer cho phòng trống, trỏ guest sang global_average
                if uid == "unknown":
                    continue
                if uid == "guest":
                    uid = "global_average"
                    item["user_id"] = uid
                    item["dominant_user_id"] = uid

                last_ts = last_ts_by_user.get(uid)

                if last_ts is not None:
                    gap = abs((dt - last_ts).total_seconds())
                    if gap < (interval * 0.5):
                        continue
                    if gap > max_gap and uid in user_features_temp:
                        user_features_temp[uid].clear()

                last_ts_by_user[uid] = dt

                if uid not in user_features_temp:
                    user_features_temp[uid] = []

                user_features_temp[uid].append({
                    "hour": item["hour"],
                    "day_of_week": item["day_of_week"],
                    "temperature": float(item["temperature"]),
                    "humidity": float(item["humidity"]),
                    "light_level": float(item["light_level"]),
                    "motion_detected": int(item["motion_detected"]),
                    "temp_delta": float(item.get("temp_delta", 0.0)),
                    "humidity_delta": float(item.get("humidity_delta", 0.0)),
                    "light_delta": float(item.get("light_delta", 0.0)),
                    "motion_streak": int(item.get("motion_streak", 1 if item.get("motion_detected", 0) else 0)),
                    "time_bucket": int(item.get("time_bucket", item["hour"] // 3)),
                    "is_weekend": int(item.get("is_weekend", 0)),
                    "user_id": uid
                })
            except Exception as e:
                logger.warning(f"[AI] Bỏ qua 1 log bị lỗi khi warm-up: {e}")

        valid_features = []
        for feats in user_features_temp.values():
            valid_features.extend(feats)

        if not valid_features:
            return

        # Tái tạo feature giàu ngữ cảnh nhưng vẫn là tín hiệu chung, không bám cứng 4 scenario.
        enriched_features = []
        prev_by_user = {}
        motion_streak_by_user = {}
        last_seen_by_user = {}
        for feat in valid_features:
            uid = feat["user_id"]
            prev = prev_by_user.get(uid, feat)
            motion_streak_by_user[uid] = motion_streak_by_user.get(uid, 0) + (1 if feat.get("motion_detected", 0) else 0)
            enriched = dict(feat)
            enriched["temp_delta"] = abs(feat["temperature"] - prev.get("temperature", feat["temperature"]))
            enriched["humidity_delta"] = abs(feat["humidity"] - prev.get("humidity", feat["humidity"]))
            enriched["light_delta"] = abs(feat["light_level"] - prev.get("light_level", feat["light_level"]))
            enriched["face_count_delta"] = abs(float(feat.get("face_count", 0)) - float(prev.get("face_count", feat.get("face_count", 0))))
            enriched["motion_streak"] = motion_streak_by_user.get(uid, 0)
            enriched["time_bucket"] = feat["hour"] // 3
            enriched["is_weekend"] = int(feat["day_of_week"] in (5, 6))
            enriched_features.append(enriched)
            prev_by_user[uid] = feat
            last_seen_by_user[uid] = feat

        try:
            encoded = self._encode_features(enriched_features, custom_encoder=current_encoder)
            X_num = encoded[:, :-1]
            X_usr = encoded[:, -1:]
            
            X_num_scaled = current_scaler.transform(X_num)
            scaled = np.column_stack([X_num_scaled, X_usr])

            with self._lock:
                for i, f in enumerate(valid_features):
                    uid = f["user_id"]
                    if uid not in self.user_buffers:
                        self.user_buffers[uid] = []
                    self.user_buffers[uid].append(scaled[i])
                    if len(self.user_buffers[uid]) > self.TIME_STEPS:
                        self.user_buffers[uid].pop(0)

            logger.info(f"[AI] Nạp thành công {len(valid_features)} mẫu hợp lệ vào Per-user State Buffers.")
        except Exception as e:
            logger.error(f"[AI] Lỗi khi warm-up transform dữ liệu: {e}")

    # ─────────────────────────────────────────────────────
    def _encode_features(self, rows: list, custom_encoder=None) -> np.ndarray:
        encoder = custom_encoder if custom_encoder is not None else self.label_encoder
        users   = [self._normalize_user_label(r.get("dominant_user_id", r.get("user_id", "unknown"))) for r in rows]
        user_encoded = []

        if encoder is not None:
            classes = set(encoder.classes_)
            for u in users:
                if u in classes:
                    user_encoded.append(encoder.transform([u])[0])
                else:
                    if 'global_average' in classes:
                        fallback_idx = encoder.transform(['global_average'])[0]
                    elif 'unknown' in classes:
                        fallback_idx = encoder.transform(['unknown'])[0]
                    else:
                        fallback_idx = 0
                    user_encoded.append(fallback_idx)
        else:
            user_encoded = [0] * len(users)

        user_encoded = np.array(user_encoded)
        
        # 1. Cyclic Encoding cho Giờ (Chu kỳ 24h)
        hours        = np.array([r.get("hour", datetime.now().hour) for r in rows])
        hour_sin     = np.sin(2 * np.pi * hours / 24.0)
        hour_cos     = np.cos(2 * np.pi * hours / 24.0)

        # 2. Cyclic Encoding cho Ngày trong tuần (Chu kỳ 7 ngày)
        days         = np.array([r.get("day_of_week", 0) for r in rows])
        day_sin      = np.sin(2 * np.pi * days / 7.0)
        day_cos      = np.cos(2 * np.pi * days / 7.0)

        # 3. Ghép nối các đặc trưng
        presence_map = {
            "empty": 0,
            "single_unknown": 1,
            "single_known": 2,
            "multi_user": 3,
            "animal": 4,
            "unknown": 1,
        }
        scenario_map = {
            "background_noise": 0,
            "sequence_behavior": 1,
            "context_exception": 2,
            "manual_override": 3,
            "profile_personalization": 4,
        }
        phase_map = {
            "background_noise": 0,
            "sequence_idle": 1,
            "override_response": 2,
            "profile_multi_user": 3,
            "context_transition": 4,
        }
        # Chỉ giữ các nhãn ngắn, ổn định, dễ thu thập từ dữ liệu thật.
        presence_encoded = np.array([presence_map.get((r.get("presence_type") or "unknown").strip().lower(), 1) for r in rows])
        scenario_encoded = np.array([scenario_map.get((r.get("scenario_type") or "background_noise").strip().lower(), 0) for r in rows])
        sequence_phase_encoded = np.array([phase_map.get((r.get("sequence_phase") or "background_noise").strip().lower(), 0) for r in rows])
        face_count = np.array([
            float(r.get("face_count", 0))
            for r in rows
        ], dtype=float)
        face_count_delta = np.zeros_like(face_count, dtype=float)
        if len(face_count) > 1:
            face_count_delta[1:] = np.abs(np.diff(face_count))
        override_flag = np.array([r.get("override_flag", 0) for r in rows], dtype=float)
        temp_delta = np.array([r.get("temp_delta", 0.0) for r in rows], dtype=float)
        humidity_delta = np.array([r.get("humidity_delta", 0.0) for r in rows], dtype=float)
        light_delta = np.array([r.get("light_delta", 0.0) for r in rows], dtype=float)
        motion_streak = np.array([r.get("motion_streak", 0) for r in rows], dtype=float)
        time_bucket = np.array([r.get("time_bucket", int(r.get("hour", 0)) // 3) for r in rows], dtype=float)
        is_weekend = np.array([r.get("is_weekend", int(int(r.get("day_of_week", 0)) in (5, 6))) for r in rows], dtype=float)
        last_light = np.array([r.get("light_state", 0) for r in rows], dtype=float)
        last_fan = np.array([r.get("fan_state", 0) for r in rows], dtype=float)
        X = np.column_stack([
            hour_sin, hour_cos,
            day_sin, day_cos,
            [r["temperature"]            for r in rows],
            [r["humidity"]               for r in rows],
            [r["light_level"]            for r in rows],
            [r.get("motion_detected", 0) for r in rows],
            temp_delta,
            humidity_delta,
            light_delta,
            motion_streak,
            time_bucket,
            is_weekend,
            presence_encoded,
            scenario_encoded,
            sequence_phase_encoded,
            face_count,
            face_count_delta,
            override_flag,
            last_light,
            last_fan,
            user_encoded,
        ])
        return X.astype(float)
    
    def _create_sequences(self, X_scaled, rows, Y, time_steps):
        Xs, ys, meta = [], [], []
        max_gap = getattr(config, "MAX_SEQUENCE_GAP_SEC", 300)

        for i in range(len(X_scaled) - time_steps):
            window_rows = rows[i:i + time_steps + 1]

            users_in_window = set(r.get("dominant_user_id", r.get("user_id", "unknown")) for r in window_rows)
            if len(users_in_window) > 1:
                continue

            if any(r.get("timestamp") is None for r in window_rows):
                continue

            try:
                timestamps = []
                for r in window_rows:
                    if isinstance(r["timestamp"], str):
                        timestamps.append(datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S"))
                    else:
                        timestamps.append(r["timestamp"])

                gaps = [(timestamps[j + 1] - timestamps[j]).total_seconds() for j in range(len(timestamps) - 1)]
                if gaps and max(gaps) > max_gap:
                    continue
            except Exception:
                continue

            seq_row = window_rows[-1]
            session_id = (seq_row.get("session_id") or "").strip()
            seq_ts = timestamps[-1]
            group_key = session_id if session_id else f"{seq_ts.strftime('%Y-%m-%d')}::{seq_row.get('dominant_user_id', seq_row.get('user_id', 'unknown'))}"

            Xs.append(X_scaled[i:i + time_steps])
            ys.append(Y[i + time_steps])
            meta.append({
                "group_key": group_key,
                "timestamp": seq_ts,
                "day_of_week": int(seq_row.get("day_of_week", seq_ts.weekday())),
                "user_id": seq_row.get("dominant_user_id", seq_row.get("user_id", "unknown")),
                "session_id": session_id,
            })

        return np.array(Xs), np.array(ys), meta

    def _resample_event_rows(self, rows: list[dict], freq_seconds: int = 30) -> list[dict]:
        """Tái lấy mẫu chuỗi sự kiện để đầu vào LSTM đều bước hơn."""
        if not rows:
            return []
        try:
            import pandas as pd
        except Exception:
            return rows

        normalized = []
        for row in rows:
            item = self._row_to_training_item(row)
            if item is not None:
                normalized.append(item)
        if len(normalized) < 2:
            return normalized

        df = pd.DataFrame(normalized)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if df.empty:
            return normalized

        df = df.set_index("timestamp")
        numeric_cols = [
            "hour", "day_of_week", "temperature", "humidity", "light_level", "motion_detected",
            "temp_delta", "humidity_delta",
            "light_delta", "motion_streak", "time_bucket", "is_weekend", "face_count",
            "override_flag"
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0

        freq = max(int(freq_seconds), 1)
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f"{freq}s")
        resampled = df.reindex(full_index)
        numeric_fill_cols = [c for c in numeric_cols if c in resampled.columns]
        resampled[numeric_fill_cols] = resampled[numeric_fill_cols].ffill().bfill()
        if "user_id" in resampled.columns:
            resampled["user_id"] = resampled["user_id"].ffill().bfill().fillna("unknown")
        for col in ["dominant_user_id", "scenario_type", "source", "presence_type", "session_id", "override_reason"]:
            if col in resampled.columns:
                resampled[col] = resampled[col].ffill().bfill().fillna("")
        resampled = resampled.reset_index().rename(columns={"index": "timestamp"})
        resampled["timestamp"] = pd.to_datetime(resampled["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return resampled.to_dict("records")

    def build_sequence_dataset_from_events(self, rows: list[dict], time_steps: int = None):
        """Tạo dataset chuỗi trực tiếp từ canonical event rows."""
        time_steps = time_steps or self.TIME_STEPS
        normalized = self._resample_event_rows(rows, freq_seconds=self.sequence_resample_freq_sec)

        if not normalized:
            return np.array([]), np.array([]), []

        with self._lock:
            encoder = self.label_encoder
            scaler = self.scaler

        if encoder is None or scaler is None:
            return np.array([]), np.array([]), []

        X_raw = self._encode_features(normalized, custom_encoder=encoder)
        X_num = X_raw[:, :-1]
        X_usr = X_raw[:, -1:]
        X_scaled = np.column_stack([scaler.transform(X_num), X_usr])

        y = np.column_stack([
            [r["light_state"] for r in normalized],
            [r["fan_state"] for r in normalized],
        ])
        X_seq, y_seq = self._create_sequences(X_scaled, normalized, y, time_steps)
        return X_seq, y_seq, normalized

    def _build_lstm_model(self, n_features: int, n_users: int) -> Model:
        input_layer  = Input(shape=(self.TIME_STEPS, n_features), name="input_seq")

        num_features = tf.keras.layers.Lambda(lambda x: x[:, :, :-1],  name="num_features")(input_layer)
        user_feature = tf.keras.layers.Lambda(lambda x: tf.cast(x[:, :, -1], tf.int32), name="user_feature")(input_layer)

        user_embed   = Embedding(input_dim=n_users + 1, output_dim=4, name="user_emb")(user_feature)

        concat_features = Concatenate(axis=-1, name="concat_features")([num_features, user_embed])

        shared_lstm  = LSTM(128, return_sequences=True, dropout=0.1,
                            recurrent_dropout=0.1, name="shared_lstm")(concat_features)

        attn_out     = MultiHeadAttention(num_heads=4, key_dim=32, name="attention")(shared_lstm, shared_lstm)
        attention_out = LayerNormalization(name="layer_norm")(attn_out + shared_lstm)

        light_branch = GlobalAveragePooling1D(name="pool_light")(attention_out)
        light_branch = Dense(32, activation='relu', kernel_regularizer=l2(0.01), name="features_light")(light_branch)
        light_branch = Dropout(0.3, name="dropout_light")(light_branch)
        out_light    = Dense(1, activation='sigmoid', name="out_light")(light_branch)

        fan_branch   = GlobalAveragePooling1D(name="pool_fan")(attention_out)
        fan_branch   = Dense(32, activation='relu', kernel_regularizer=l2(0.01), name="features_fan")(fan_branch)
        fan_branch   = Dropout(0.3, name="dropout_fan")(fan_branch)
        out_fan      = Dense(1, activation='sigmoid', name="out_fan")(fan_branch)

        return Model(inputs=input_layer, outputs=[out_light, out_fan])

    def _build_global_average_profile(self, rows: list) -> list:
        user_data = {}
        for r in rows:
            uid = r.get("dominant_user_id", r.get("user_id", "unknown"))
            if uid not in ("unknown", "global_average", "guest"):
                if uid not in user_data:
                    user_data[uid] = []
                user_data[uid].append(r)

        if not user_data:
            return []

        min_samples = min(len(data) for data in user_data.values())
        target_samples = min(min_samples, 1000)
        if target_samples == 0:
            return []

        synthetic_rows = []
        for data in user_data.values():
            sampled_data = random.sample(data, target_samples)
            for r in sampled_data:
                synthetic = dict(r)
                synthetic["user_id"] = "global_average"
                synthetic["dominant_user_id"] = "global_average"
                synthetic_rows.append(synthetic)

        real_row_count = len([
            r for r in rows
            if r.get("dominant_user_id", r.get("user_id", "unknown")) not in ("global_average", "unknown", "guest")
        ])
        max_synthetic = int(real_row_count * 0.2)
        if len(synthetic_rows) > max_synthetic and max_synthetic > 0:
            synthetic_rows = random.sample(synthetic_rows, max_synthetic)
        elif max_synthetic == 0:
            synthetic_rows = []

        logger.info(
            f"[AI] Đã tạo profile 'global_average' từ {len(user_data)} users "
            f"({len(synthetic_rows)} mẫu)."
        )
        return synthetic_rows

    def record_override_feedback(self, *, user_id: str, predicted: dict, actual: dict, source: str = "manual_override", reason: str = "") -> bool:
        try:
            normalized_user = self._normalize_user_label(user_id)
            if source in ("manual_override", "manual"):
                self._set_manual_override_lockout(normalized_user, reason=reason or source)

            payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": normalized_user or "unknown",
                "dominant_user_id": normalized_user or "unknown",
                "light_state": int(actual.get("light", predicted.get("light", 0))),
                "fan_state": int(actual.get("fan", predicted.get("fan", 0))),
                "source": source,
                "scenario_type": "manual_override",
                "sequence_phase": "override_response",
                "override_flag": 1,
                "override_reason": reason or source,
                "session_id": "",
                "presence_type": actual.get("presence_type", "human"),
                "face_count": int(actual.get("face_count", 1)),
                "decision_mode": actual.get("decision_mode", "manual"),
                "data_source": "real",
                "temperature": float(actual.get("temperature", 0.0)),
                "humidity": float(actual.get("humidity", 0.0)),
                "light_level": float(actual.get("light_level", 0.0)),
                "motion_detected": int(actual.get("motion_detected", 0)),
                "hour": int(actual.get("hour", datetime.now().hour)),
                "minute": int(actual.get("minute", datetime.now().minute)),
                "day_of_week": int(actual.get("day_of_week", datetime.now().weekday())),
                "confidence": float(predicted.get("confidence", 0.0)),
                
            }
            self.db.insert_event_log(payload)
            return True
        except Exception as e:
            logger.warning(f"[AI] Không ghi được override feedback: {e}")
            return False

    # ─────────────────────────────────────────────────────
    def _prepare_training_rows(self, data_source: str) -> tuple[list, str]:
        prepared_csv = os.path.join(getattr(config, "DATA_DIR", os.path.dirname(self.db.db_path)), "training_data.csv")
        prepared_json = os.path.join(getattr(config, "DATA_DIR", os.path.dirname(self.db.db_path)), "training_data_prepared.json")
        source_map = {
            "simulated": ("simulated",),
            "real": ("real", "manual_override", "manual"),
            "hybrid": ("real", "simulated", "manual_override", "manual"),
        }

        rows = []
        source_name = "db"
        if data_source in source_map:
            rows = self.db.get_training_data_by_source(source_map[data_source])
            source_name = f"db_{data_source}"

        rows = [self._row_to_training_item(r) for r in rows]
        rows = [r for r in rows if r is not None]
        rows = self.db._enrich_training_rows(rows)
        for r in rows:
            if r.get("dominant_user_id") is None and r.get("user_id") is not None:
                r["dominant_user_id"] = r["user_id"]

        if not rows and os.path.exists(prepared_json):
            try:
                with open(prepared_json, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_rows = cached.get("feature_rows", []) or cached.get("feature_bundle", {}).get("rows", [])
                rows = [self._row_to_training_item(r) for r in cached_rows]
                rows = [r for r in rows if r is not None]
                if rows:
                    source_name = "prepared_csv_json"
                    logger.info(f"[AI] Đã nạp {len(rows)} mẫu từ file chuẩn hóa {prepared_json}.")
            except Exception as e:
                logger.warning(f"[AI] Không đọc được file chuẩn hóa: {e}")

        if not rows and os.path.exists(prepared_csv):
            try:
                from train_from_csv import _prepare_rows, _parse_rows
                raw_rows = _parse_rows(Path(prepared_csv))
                parsed_rows, summary = _prepare_rows(raw_rows)
                rows = [self._row_to_training_item(r) for r in parsed_rows]
                rows = [r for r in rows if r is not None]
                rows = self.db._enrich_training_rows(rows)
                source_name = "prepared_csv"
                logger.info(f"[AI] Đã nạp {len(rows)} mẫu từ CSV {prepared_csv}.")
                logger.info(f"[AI] Override={summary.override_count}, sequence={summary.sequence_count}, context={summary.context_exception_count}")
            except Exception as e:
                logger.warning(f"[AI] Không đọc được training_data.csv bằng script mới: {e}")

        if not rows:
            try:
                from data_extractor import SmartHomeDataExtractor
                extractor = SmartHomeDataExtractor(db_path=self.db.db_path)
                df = extractor.extract_and_process(lookahead_minutes=5, correction_weight=3.0)
                if df is not None and not df.empty:
                    df['timestamp'] = df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
                    rows = [self._row_to_training_item(r) for r in df.to_dict('records')]
                    rows = [r for r in rows if r is not None]
                    rows = self.db._enrich_training_rows(rows)
                    source_name = "golden_extractor"
                    logger.info(f"[AI] Đã tải {len(rows)} bản ghi Golden Data từ Extractor.")
            except Exception as e:
                logger.warning(f"[AI] Không tải được extractor data: {e}")

        rows = self.db._enrich_training_rows(rows)
        return rows, source_name

    def _apply_training_filters(self, rows: list) -> list:
        original_len = len(rows)
        rows = [r for r in rows if r.get("user_id") not in ("animal", "unknown") and r.get("presence_type") != "animal"]
        for r in rows:
            if r.get("dominant_user_id") in (None, ""):
                r["dominant_user_id"] = r.get("user_id", "unknown")
            r["label_conflict"] = int((r.get("user_id") or "unknown") != (r.get("dominant_user_id") or "unknown"))
            scenario = (r.get("scenario_type") or "").strip().lower()
            base_weight = float(r.get("sample_weight", 1.0))
            if int(r.get("override_flag", 0)) == 1:
                base_weight *= 1.6
            if scenario == "manual_override":
                base_weight *= 1.4
            elif scenario == "sequence_behavior":
                base_weight *= 1.1
            elif scenario == "context_exception":
                base_weight *= 1.05
            elif scenario == "profile_personalization":
                base_weight *= 1.15
            elif scenario == "background_noise":
                base_weight *= 0.9
            r["sample_weight"] = float(np.clip(base_weight, 0.5, 2.0))
        if len(rows) < original_len:
            logger.info(f"[AI] Đã lọc bỏ {original_len - len(rows)} mẫu nhiễu (phòng trống/vật nuôi) khỏi tập huấn luyện.")
        conflict_count = sum(int(r.get("label_conflict", 0)) for r in rows)
        logger.info(
            f"[AI] Label disagreement raw/stable: {conflict_count}/{len(rows)} "
            f"({(conflict_count / max(len(rows), 1)):.1%})"
        )
        return rows

    def _build_training_dataset(self, rows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        label_encoder_temp = LabelEncoder()
        fit_classes = sorted({
            r.get("dominant_user_id", r.get("user_id", "unknown"))
            for r in rows
            if r.get("dominant_user_id", r.get("user_id", "unknown")) not in (None, "", "unknown", "guest")
        })
        all_classes = sorted({
            r.get("dominant_user_id", r.get("user_id", "unknown"))
            for r in rows
            if r.get("dominant_user_id", r.get("user_id", "unknown")) not in (None, "", "guest")
        })
        if any((r.get("scenario_type") or "").strip().lower() == "profile_personalization" for r in rows):
            # profile_personalization là nhãn ngữ cảnh, không phải user. Giữ user_id bình thường nhưng tăng độ phủ của label set qua global_average nếu có.
            pass
        if any(r.get("user_id") == "global_average" for r in rows) and "global_average" not in all_classes:
            all_classes.append("global_average")
        if "unknown" not in all_classes:
            all_classes.append("unknown")
        fit_classes = sorted(set(fit_classes) | set(all_classes)) or ["unknown"]
        label_encoder_temp.fit(fit_classes)

        X_raw_unscaled = self._encode_features(rows, custom_encoder=label_encoder_temp)
        X_numeric_unscaled = X_raw_unscaled[:, :-1]
        X_user_unscaled = X_raw_unscaled[:, -1:]

        y_light = np.array([r["light_state"] for r in rows])
        y_fan = np.array([r["fan_state"] for r in rows])
        y_scenario = np.array([
            4 if (r.get("scenario_type") or "").strip().lower() == "profile_personalization" else
            3 if (r.get("scenario_type") or "").strip().lower() == "manual_override" else
            2 if (r.get("scenario_type") or "").strip().lower() == "context_exception" else
            1 if (r.get("scenario_type") or "").strip().lower() == "sequence_behavior" else 0
            for r in rows
        ], dtype=int)
        # Sequence phase được rút gọn để ưu tiên nhãn dễ thu thập từ dữ liệu thật.
        y_sequence_phase = np.array([
            4 if (r.get("sequence_phase") or "").strip().lower() == "context_transition" else
            3 if (r.get("sequence_phase") or "").strip().lower() == "profile_multi_user" else
            2 if (r.get("sequence_phase") or "").strip().lower() == "override_response" else
            1 if (r.get("sequence_phase") or "").strip().lower() == "sequence_idle" else
            0
            for r in rows
        ], dtype=int)
        y_override = np.array([int(r.get("override_flag", 0)) for r in rows], dtype=int)
        weights = np.array([
            1.0 if r.get("is_synthetic") else float(r.get("sample_weight", 1.0))
            for r in rows
        ], dtype=float)
        return X_numeric_unscaled, X_user_unscaled, y_light, y_fan, np.column_stack([y_scenario, y_sequence_phase, y_override, weights])

    def _run_quality_audit(self, *, rows: list, label_encoder: LabelEncoder, scaler: StandardScaler,
                           feature_matrix: np.ndarray, y_light: np.ndarray, y_fan: np.ndarray,
                           y_scenario: np.ndarray, y_sequence_phase: np.ndarray, y_override: np.ndarray) -> dict:
        total_rows = len(rows)
        unique_users = len({r.get("dominant_user_id", r.get("user_id", "unknown")) for r in rows})
        missing_values = int(np.isnan(feature_matrix).sum()) if isinstance(feature_matrix, np.ndarray) else 0
        dup_pairs = len(rows) - len({(r.get("timestamp"), r.get("dominant_user_id"), r.get("light_state"), r.get("fan_state")) for r in rows})
        light_ratio = float(np.mean(y_light)) if len(y_light) else 0.0
        fan_ratio = float(np.mean(y_fan)) if len(y_fan) else 0.0
        override_ratio = float(np.mean(y_override)) if len(y_override) else 0.0
        scenario_counts = {
            "background_noise": int(np.sum(y_scenario == 0)),
            "sequence_behavior": int(np.sum(y_scenario == 1)),
            "context_exception": int(np.sum(y_scenario == 2)),
            "manual_override": int(np.sum(y_scenario == 3)),
            "profile_personalization": int(np.sum(y_scenario == 4)),
        }
        phase_counts = {
            "background_noise": int(np.sum(y_sequence_phase == 0)),
            "sequence_idle": int(np.sum(y_sequence_phase == 1)),
            "override_response": int(np.sum(y_sequence_phase == 2)),
            "profile_multi_user": int(np.sum(y_sequence_phase == 3)),
            "context_transition": int(np.sum(y_sequence_phase == 4)),
        }
        dominant_scenario = max(scenario_counts, key=scenario_counts.get) if scenario_counts else "unknown"
        scenario_total = max(sum(scenario_counts.values()), 1)
        dominant_phase = max(phase_counts, key=phase_counts.get) if phase_counts else "unknown"
        phase_total = max(sum(phase_counts.values()), 1)

        thresholds = {
            "min_rows": int(getattr(config, "QUALITY_MIN_ROWS", max(self.TIME_STEPS * 20, 100))),
            "min_unique_users": int(getattr(config, "QUALITY_MIN_UNIQUE_USERS", 2)),
            "max_missing_values": int(getattr(config, "QUALITY_MAX_MISSING_VALUES", 0)),
            "max_duplicate_ratio": float(getattr(config, "QUALITY_MAX_DUPLICATE_RATIO", 0.1)),
            "min_class_balance_ratio": float(getattr(config, "QUALITY_MIN_CLASS_BALANCE_RATIO", 0.1)),
            "min_override_ratio": float(getattr(config, "QUALITY_MIN_OVERRIDE_RATIO", 0.02)),
            "max_dominant_scenario_ratio": float(getattr(config, "QUALITY_MAX_DOMINANT_SCENARIO_RATIO", 0.65)),
            "max_dominant_phase_ratio": float(getattr(config, "QUALITY_MAX_DOMINANT_PHASE_RATIO", 0.70)),
            "fail_max_warnings": int(getattr(config, "QUALITY_FAIL_MAX_WARNINGS", 2)),
            "warn_min_score": float(getattr(config, "QUALITY_WARN_MIN_SCORE", 75.0)),
            "pass_min_score": float(getattr(config, "QUALITY_PASS_MIN_SCORE", 100.0)),
        }

        checks = {
            "row_count": {
                "value": total_rows,
                "threshold": thresholds["min_rows"],
                "passed": total_rows >= thresholds["min_rows"],
                "severity": "critical",
                "message": f"Số mẫu {total_rows} / tối thiểu {thresholds['min_rows']}",
            },
            "user_diversity": {
                "value": unique_users,
                "threshold": thresholds["min_unique_users"],
                "passed": unique_users >= thresholds["min_unique_users"],
                "severity": "major",
                "message": f"Số user khác nhau {unique_users} / tối thiểu {thresholds['min_unique_users']}",
            },
            "missing_values": {
                "value": missing_values,
                "threshold": thresholds["max_missing_values"],
                "passed": missing_values <= thresholds["max_missing_values"],
                "severity": "critical",
                "message": f"Giá trị thiếu {missing_values} / tối đa {thresholds['max_missing_values']}",
            },
            "duplicates": {
                "value": dup_pairs,
                "threshold": int(total_rows * thresholds["max_duplicate_ratio"]),
                "passed": dup_pairs <= int(total_rows * thresholds["max_duplicate_ratio"]),
                "severity": "major",
                "message": f"Mẫu trùng {dup_pairs} / tối đa {int(total_rows * thresholds['max_duplicate_ratio'])}",
            },
            "light_balance": {
                "value": light_ratio,
                "threshold": thresholds["min_class_balance_ratio"],
                "passed": min(light_ratio, 1 - light_ratio) >= thresholds["min_class_balance_ratio"],
                "severity": "major",
                "message": f"Tỷ lệ đèn bật {light_ratio:.2%}",
            },
            "fan_balance": {
                "value": fan_ratio,
                "threshold": thresholds["min_class_balance_ratio"],
                "passed": min(fan_ratio, 1 - fan_ratio) >= thresholds["min_class_balance_ratio"],
                "severity": "major",
                "message": f"Tỷ lệ quạt bật {fan_ratio:.2%}",
            },
            "override_coverage": {
                "value": override_ratio,
                "threshold": thresholds["min_override_ratio"],
                "passed": override_ratio >= thresholds["min_override_ratio"],
                "severity": "minor",
                "message": f"Tỷ lệ override {override_ratio:.2%}",
            },
            "scenario_balance": {
                "value": scenario_counts.get(dominant_scenario, 0) / scenario_total,
                "threshold": thresholds["max_dominant_scenario_ratio"],
                "passed": (scenario_counts.get(dominant_scenario, 0) / scenario_total) <= thresholds["max_dominant_scenario_ratio"],
                "severity": "major",
                "message": f"Kịch bản trội nhất '{dominant_scenario}' chiếm {(scenario_counts.get(dominant_scenario, 0) / scenario_total):.2%}",
            },
            "phase_balance": {
                "value": phase_counts.get(dominant_phase, 0) / phase_total,
                "threshold": thresholds["max_dominant_phase_ratio"],
                "passed": (phase_counts.get(dominant_phase, 0) / phase_total) <= thresholds["max_dominant_phase_ratio"],
                "severity": "major",
                "message": f"Sequence phase trội nhất '{dominant_phase}' chiếm {(phase_counts.get(dominant_phase, 0) / phase_total):.2%}",
            },
        }

        warnings = [key for key, check in checks.items() if not check["passed"]]
        critical_fail = any((not check["passed"]) and check["severity"] == "critical" for check in checks.values())
        major_fail_count = sum((not check["passed"]) and check["severity"] in ("critical", "major") for check in checks.values())
        score = round((sum(1 for check in checks.values() if check["passed"]) / max(len(checks), 1)) * 100.0, 1)
        status = "pass" if not warnings else (
            "fail" if critical_fail or major_fail_count >= thresholds["fail_max_warnings"] or score < thresholds["warn_min_score"] else "warn"
        )

        report = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "status": status,
            "thresholds": thresholds,
            "summary": {
                "rows": total_rows,
                "unique_users": unique_users,
                "missing_values": missing_values,
                "duplicate_signature_count": dup_pairs,
                "light_positive_ratio": round(light_ratio, 4),
                "fan_positive_ratio": round(fan_ratio, 4),
                "override_ratio": round(override_ratio, 4),
                "dominant_scenario": dominant_scenario,
                "dominant_phase": dominant_phase,
            },
            "checks": checks,
            "warnings": warnings,
            "score": score,
        }
        try:
            report_dir = Path(getattr(config, "DATA_DIR", os.path.dirname(self.db.db_path)))
            report_dir.mkdir(parents=True, exist_ok=True)
            audit_path = report_dir / "training_quality_audit.json"
            with audit_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"[AI] Đã lưu quality audit: {audit_path}")
            report["report_path"] = str(audit_path)
        except Exception as e:
            logger.warning(f"[AI] Không lưu được quality audit: {e}")
        return report

    def train(self, incremental: bool = False, data_source: str = "hybrid", eval_source: str = "real") -> dict:
        logger.info("[AI] Khởi chạy luồng huấn luyện LSTM-XGBoost...")
        data_source = (data_source or "hybrid").strip().lower()
        eval_source = (eval_source or "real").strip().lower()
        
        rows = []
        eval_rows_raw = []
        source_name = "db"

        source_map = {
            "simulated": ("simulated",),
            "real": ("real", "manual_override", "manual"),
            "hybrid": ("real", "simulated", "manual_override", "manual"),
        }
        eval_map = {
            "simulated": ("simulated",),
            "real": ("real", "manual_override", "manual"),
            "hybrid": ("real",),
        }

        if data_source in source_map:
            eval_rows_raw = self.db.get_training_data_by_source(eval_map.get(eval_source, eval_map["real"]))

        rows, source_name = self._prepare_training_rows(data_source)
        rows = self._apply_training_filters(rows)
        rows = self.db._enrich_training_rows(rows)

        synthetic_rows = self._build_global_average_profile(rows)
        for sr in synthetic_rows:
            sr['sample_weight'] = 1.0
            sr['is_synthetic'] = True
        rows.extend(synthetic_rows)

        total = len(rows)
        if total < config.MIN_TRAINING_ROWS + self.TIME_STEPS:
            return {"success": False, "message": "Chưa đủ dữ liệu chuỗi.", "total_rows": total}

        train_size_estimate = int(total * 0.85)

        X_numeric_unscaled, X_user_unscaled, y_light, y_fan, y_meta = self._build_training_dataset(rows)
        train_scenario_codes = y_meta[:, 0].astype(int)
        train_sequence_phase_codes = y_meta[:, 1].astype(int)
        train_override_flags = y_meta[:, 2].astype(int)
        weights = np.clip(y_meta[:, 3].astype(float), 0.1, 3.0)
        feedback_scores = np.array([
            0.0 if str(r.get("feedback_source", r.get("source", ""))).strip().lower() in ("manual_override", "active_learning_feedback") else 1.0
            for r in rows
        ], dtype=float)
        Y = np.column_stack([y_light, y_fan, train_scenario_codes, train_sequence_phase_codes, train_override_flags, weights, feedback_scores])

        label_encoder_temp = LabelEncoder()
        fit_classes = sorted({
            r.get("dominant_user_id", r.get("user_id", "unknown"))
            for r in rows
            if r.get("dominant_user_id", r.get("user_id", "unknown")) not in (None, "", "unknown", "guest")
        }) or ["unknown"]
        if any((r.get("dominant_user_id", r.get("user_id")) == "global_average") for r in rows) and "global_average" not in fit_classes:
            fit_classes.append("global_average")
        if "unknown" not in fit_classes:
            fit_classes.append("unknown")
        label_encoder_temp.fit(sorted(set(fit_classes)))
        n_users = len(label_encoder_temp.classes_)

        train_cut_raw = int(len(X_numeric_unscaled) * 0.85)
        X_train_numeric_raw = X_numeric_unscaled[:train_cut_raw]

        scaler_temp = StandardScaler()
        scaler_temp.fit(X_train_numeric_raw)

        X_numeric_scaled = scaler_temp.transform(X_numeric_unscaled)
        X_raw_scaled = np.column_stack([X_numeric_scaled, X_user_unscaled])
        
        X_seq_scaled, Y_seq, seq_meta = self._create_sequences(X_raw_scaled, rows, Y, self.TIME_STEPS)
        raw_override_counts = dict(zip(*np.unique(train_override_flags, return_counts=True))) if len(train_override_flags) > 0 else {}
        sequence_override_counts = dict(zip(*np.unique(Y_seq[:, 4].astype(int), return_counts=True))) if len(Y_seq) > 0 and Y_seq.shape[1] > 4 else {}
        logger.info(
            "[AI Override] raw=%s | after_sequence=%s | total_seq=%d",
            raw_override_counts,
            sequence_override_counts,
            len(Y_seq),
        )

        if len(X_seq_scaled) == 0:
            return {
                "success": False,
                "message": "Không tạo được chuỗi hợp lệ sau khi lọc Gap/User.",
                "total_rows": total
            }

        seq_groups = np.array([m.get("group_key", "") for m in seq_meta], dtype=object)
        seq_timestamps = np.array([m.get("timestamp") for m in seq_meta], dtype=object)

        combined_labels_all = (Y_seq[:, 0].astype(int) * 2) + Y_seq[:, 1].astype(int)
        unique_combined_all, combined_counts_all = np.unique(combined_labels_all, return_counts=True)
        unique_light_all, light_counts_all = np.unique(Y_seq[:, 0].astype(int), return_counts=True)

        can_stratify_combined_all = len(unique_combined_all) >= 2 and np.min(combined_counts_all) >= 2
        can_stratify_light_all = len(unique_light_all) >= 2 and np.min(light_counts_all) >= 2
        eval_split_mode = "time_tail"
        if can_stratify_combined_all:
            eval_split_mode = "stratified_combined"
        elif can_stratify_light_all:
            eval_split_mode = "stratified_light"

        all_indices = np.arange(len(X_seq_scaled))
        train_idx = all_indices
        test_idx = np.array([], dtype=int)

        # Ưu tiên giữ tối thiểu một tỷ lệ dương tính ở cả train và test.
        light_pos_idx = np.where(Y_seq[:, 0].astype(int) == 1)[0]
        fan_pos_idx   = np.where(Y_seq[:, 1].astype(int) == 1)[0]

        # Ưu tiên split theo group/session/day nếu có, để tránh leakage chuỗi.
        group_keys_all = np.array([m.get("group_key", "") for m in seq_meta], dtype=object)
        unique_groups = []
        group_first_idx = {}
        for idx, key in enumerate(group_keys_all):
            if key not in group_first_idx:
                group_first_idx[key] = idx
                unique_groups.append(key)

        use_group_split = len(unique_groups) >= 5 and len(set(unique_groups)) > 1
        if use_group_split:
            group_labels = []
            for key in unique_groups:
                idxs = np.where(group_keys_all == key)[0]
                if len(idxs) == 0:
                    group_labels.append(0)
                    continue
                group_y = Y_seq[idxs]
                group_combined = (group_y[:, 0].astype(int) * 2) + group_y[:, 1].astype(int)
                vals, counts = np.unique(group_combined, return_counts=True)
                group_labels.append(int(vals[np.argmax(counts)]))
            group_labels = np.array(group_labels, dtype=int)
            can_stratify_groups = len(np.unique(group_labels)) >= 2 and np.min(np.unique(group_labels, return_counts=True)[1]) >= 1
            try:
                group_train_pos, group_test_pos = train_test_split(
                    np.arange(len(unique_groups)),
                    test_size=min(0.2, max(1 / max(len(unique_groups), 1), 0.1)),
                    random_state=42,
                    shuffle=True,
                    stratify=group_labels if can_stratify_groups else None,
                )
                test_group_set = {unique_groups[i] for i in np.array(group_test_pos, dtype=int)}
                train_mask = np.array([g not in test_group_set for g in group_keys_all], dtype=bool)
                test_mask = ~train_mask
                train_idx = all_indices[train_mask]
                test_idx = all_indices[test_mask]
            except ValueError:
                use_group_split = False

        if not use_group_split:
            test_size = min(0.2, max(0.1, 20 / len(X_seq_scaled))) if len(X_seq_scaled) > 0 else 0.2
            stratify_labels_all = combined_labels_all if can_stratify_combined_all else (Y_seq[:, 0].astype(int) if can_stratify_light_all else None)
            try:
                train_idx, test_idx = train_test_split(
                    all_indices,
                    test_size=test_size,
                    random_state=42,
                    shuffle=True,
                    stratify=stratify_labels_all,
                )
                train_idx = np.array(sorted(train_idx.tolist()), dtype=int)
                test_idx = np.array(sorted(test_idx.tolist()), dtype=int)
            except ValueError:
                train_idx = all_indices
                test_idx = np.array([], dtype=int)

        def _ensure_min_positive_split(train_idx_arr, test_idx_arr, label_array, min_test_pos=2, min_train_pos=2):
            train_set = list(train_idx_arr.tolist())
            test_set = list(test_idx_arr.tolist())
            if len(test_set) == 0 or len(np.unique(label_array)) < 2:
                return train_idx_arr, test_idx_arr

            def _count_pos(idxs):
                return int(np.sum(label_array[idxs] == 1))

            train_pos = _count_pos(train_set)
            test_pos = _count_pos(test_set)
            if test_pos >= min_test_pos and train_pos >= min_train_pos:
                return train_idx_arr, test_idx_arr

            pos_idx = np.where(label_array == 1)[0].tolist()
            neg_idx = np.where(label_array == 0)[0].tolist()
            if len(pos_idx) < 2 or len(neg_idx) < 2:
                return train_idx_arr, test_idx_arr

            desired_test_pos = min(max(min_test_pos, 1), len(pos_idx) - 1)
            desired_test_neg = max(1, int(round(len(test_set) - test_pos + (min_test_pos - test_pos))))
            rng = random.Random(42)
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            new_test = pos_idx[:desired_test_pos] + neg_idx[:max(1, min(desired_test_neg, len(neg_idx) - 1))]
            new_test = sorted(set(new_test))
            new_train = [i for i in all_indices if i not in set(new_test)]
            return np.array(sorted(new_train), dtype=int), np.array(sorted(new_test), dtype=int)

        train_idx, test_idx = _ensure_min_positive_split(train_idx, test_idx, Y_seq[:, 0].astype(int), min_test_pos=2, min_train_pos=2)

        X_train_full = X_seq_scaled[train_idx]
        y_train_full = Y_seq[train_idx]
        meta_train_full = [seq_meta[i] for i in train_idx]
        X_test = X_seq_scaled[test_idx] if len(test_idx) > 0 else np.array([])
        y_test = Y_seq[test_idx] if len(test_idx) > 0 else np.array([])
        logger.info(
            "[AI Split] override(full train/test)=%s/%s | train_idx=%d | test_idx=%d",
            dict(zip(*np.unique(Y_seq[train_idx, 4].astype(int), return_counts=True))) if len(train_idx) > 0 else {},
            dict(zip(*np.unique(Y_seq[test_idx, 4].astype(int), return_counts=True))) if len(test_idx) > 0 else {},
            len(train_idx),
            len(test_idx),
        )
        if len(X_test) > 0 and len(np.unique(y_test[:, 0].astype(int))) < 2 and can_stratify_light_all and not use_group_split:
            try:
                _, test_idx = train_test_split(
                    all_indices[train_idx],
                    test_size=min(0.2, max(0.1, 20 / len(X_seq_scaled))),
                    random_state=99,
                    shuffle=True,
                    stratify=Y_seq[train_idx, 0].astype(int),
                )
                test_idx = np.array(sorted(test_idx.tolist()), dtype=int)
                train_idx = np.array([i for i in all_indices if i not in set(test_idx.tolist())], dtype=int)
                train_idx, test_idx = _ensure_min_positive_split(train_idx, test_idx, Y_seq[:, 0].astype(int), min_test_pos=2, min_train_pos=2)
                X_train_full = X_seq_scaled[train_idx]
                y_train_full = Y_seq[train_idx]
                meta_train_full = [seq_meta[i] for i in train_idx]
                X_test = X_seq_scaled[test_idx]
                y_test = Y_seq[test_idx]
            except ValueError:
                pass

        train_light_counts = dict(zip(*np.unique(y_train_full[:, 0].astype(int), return_counts=True))) if len(y_train_full) else {}
        train_fan_counts = dict(zip(*np.unique(y_train_full[:, 1].astype(int), return_counts=True))) if len(y_train_full) else {}
        train_override_counts = dict(zip(*np.unique(y_train_full[:, 4].astype(int), return_counts=True))) if len(y_train_full) else {}
        test_light_counts = dict(zip(*np.unique(y_test[:, 0].astype(int), return_counts=True))) if len(y_test) else {}
        test_fan_counts = dict(zip(*np.unique(y_test[:, 1].astype(int), return_counts=True))) if len(y_test) else {}
        test_override_counts = dict(zip(*np.unique(y_test[:, 4].astype(int), return_counts=True))) if len(y_test) else {}
        logger.info(
            "[AI Split] mode=%s | train=%d | test=%d | groups(train/test)=%d/%d | light(train/test)=%s/%s | fan(train/test)=%s/%s | override(train/test)=%s/%s",
            eval_split_mode,
            len(X_train_full),
            len(X_test),
            len(set([m.get('group_key', '') for m in meta_train_full])) if meta_train_full else 0,
            len(set([seq_meta[i].get('group_key', '') for i in test_idx])) if len(test_idx) > 0 else 0,
            train_light_counts,
            test_light_counts,
            train_fan_counts,
            test_fan_counts,
            train_override_counts,
            test_override_counts,
        )

        n_features = X_train_full.shape[-1]

        X_temp = X_train_full
        y_temp = y_train_full
        y_temp_light = y_temp[:, 0]
        y_temp_fan = y_temp[:, 1]
        y_temp_scenario = y_temp[:, 2].astype(int)
        y_temp_phase = y_temp[:, 3].astype(int)
        y_temp_override = y_temp[:, 4].astype(int)
        y_temp_weight = y_temp[:, 5].astype(float)

        val_size = int(len(X_temp) * 0.1)
        xgb_size = max(1, int(len(X_temp) * 0.2))
        lstm_size = max(1, len(X_temp) - val_size - xgb_size)
        if lstm_size <= 0:
            lstm_size = len(X_temp)

        # Tách riêng nhánh override để bảo toàn đủ 2 lớp trong train override.
        def _split_binary_override_branch(X_data, y_data, target_idx=4, min_train_pos=2, min_train_neg=2):
            labels = y_data[:, target_idx].astype(int)
            classes = np.unique(labels)
            total_counts = dict(zip(*np.unique(labels, return_counts=True))) if len(labels) > 0 else {}
            if len(classes) < 2:
                return X_data, y_data, total_counts, total_counts, False

            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            if len(pos_idx) < 2 or len(neg_idx) < 2:
                return X_data, y_data, total_counts, total_counts, False

            desired_train_size = min(max(4, int(round(len(X_data) * 0.8))), len(X_data) - 1)
            stratify_labels = labels if min(len(pos_idx), len(neg_idx)) >= 2 else None
            try:
                train_idx, _ = train_test_split(
                    np.arange(len(X_data)),
                    train_size=desired_train_size,
                    random_state=24,
                    shuffle=True,
                    stratify=stratify_labels,
                )
                train_idx = np.array(sorted(train_idx.tolist()), dtype=int)
            except ValueError:
                rng = random.Random(24)
                pos_list = pos_idx.tolist()
                neg_list = neg_idx.tolist()
                rng.shuffle(pos_list)
                rng.shuffle(neg_list)
                train_pos_n = max(min_train_pos, int(round(desired_train_size * (len(pos_idx) / len(labels)))))
                train_neg_n = max(min_train_neg, desired_train_size - train_pos_n)
                train_pos_n = min(train_pos_n, len(pos_list) - 1)
                train_neg_n = min(train_neg_n, len(neg_list) - 1)
                train_idx = np.array(sorted((pos_list[:train_pos_n] + neg_list[:train_neg_n])), dtype=int)

            train_labels = labels[train_idx]
            train_counts = dict(zip(*np.unique(train_labels, return_counts=True))) if len(train_labels) > 0 else {}
            if len(np.unique(train_labels)) < 2:
                return X_data, y_data, total_counts, train_counts, False
            return X_data[train_idx], y_data[train_idx], total_counts, train_counts, True

        X_train_xgb = X_temp
        y_train_xgb = y_temp
        X_train_override = X_temp
        y_train_override = y_temp
        if len(X_temp) > 1:
            xgb_stratify_labels = (y_temp[:, 0].astype(int) * 2) + y_temp[:, 1].astype(int)
            xgb_can_stratify = len(np.unique(xgb_stratify_labels)) >= 2 and np.min(np.unique(xgb_stratify_labels, return_counts=True)[1]) >= 2
            xgb_train_size = min(xgb_size, len(X_temp) - 1)
            try:
                X_train_xgb, _, y_train_xgb, _ = train_test_split(
                    X_temp,
                    y_temp,
                    train_size=xgb_train_size,
                    random_state=42,
                    shuffle=True,
                    stratify=xgb_stratify_labels if xgb_can_stratify and xgb_train_size < len(X_temp) else None,
                )
            except ValueError:
                X_train_xgb = X_temp[:xgb_train_size]
                y_train_xgb = y_temp[:xgb_train_size]

            logger.info(
                "[AI Split] xgb(light/fan) train=%s | override=%s",
                dict(zip(*np.unique(y_train_xgb[:, 4].astype(int), return_counts=True))) if len(y_train_xgb) > 0 and y_train_xgb.shape[1] > 4 else {},
                dict(zip(*np.unique(y_train_xgb[:, 4].astype(int), return_counts=True))) if len(y_train_xgb) > 0 and y_train_xgb.shape[1] > 4 else {},
            )

            override_labels_all = y_temp[:, 3].astype(int)
            override_classes_all = np.unique(override_labels_all)
            if len(override_classes_all) >= 2:
                X_train_override, y_train_override, override_total_counts, override_train_counts, override_split_ok = _split_binary_override_branch(
                    X_temp, y_temp, target_idx=3, min_train_pos=2, min_train_neg=2
                )
                if not override_split_ok:
                    X_train_override = X_temp
                    y_train_override = y_temp
            logger.info(
                "[AI Split] override branch train=%s | total=%s | split_ok=%s",
                dict(zip(*np.unique(y_train_override[:, 3].astype(int), return_counts=True))) if len(y_train_override) > 0 else {},
                dict(zip(*np.unique(override_labels_all, return_counts=True))) if len(override_labels_all) > 0 else {},
                bool(len(np.unique(y_train_override[:, 3].astype(int))) >= 2),
            )
        else:
            X_train_xgb = X_temp
            y_train_xgb = y_temp
            X_train_override = X_temp
            y_train_override = y_temp
            logger.info("[AI Split] xgb/override fallback dùng toàn bộ tập vì X_temp <= 1")

        if len(X_temp) > 20:
            # Giữ split theo thứ tự thời gian để tránh leakage chuỗi vào validation.
            X_tr_lstm = X_temp[:lstm_size]
            y_tr_lstm = y_temp[:lstm_size]
            X_val_lstm = X_temp[lstm_size:lstm_size + val_size] if val_size > 0 else np.array([])
            y_val_lstm = y_temp[lstm_size:lstm_size + val_size] if val_size > 0 else np.array([])
            val_data = (X_val_lstm, {'out_light': y_val_lstm[:, 0], 'out_fan': y_val_lstm[:, 1]}) if len(X_val_lstm) > 0 else None
        else:
            X_tr_lstm = X_temp
            y_tr_lstm = y_temp
            val_data = None

        if len(X_tr_lstm) == 0:
            X_tr_lstm = X_temp
            y_tr_lstm = y_temp
            val_data = None
        if len(X_tr_lstm) > 0 and len(y_tr_lstm) > 0 and len(X_val_lstm) > 0:
            pass

        if len(y_train_xgb) > 0 and y_train_xgb.shape[1] > 4 and len(np.unique(y_train_xgb[:, 4].astype(int))) < 2:
            logger.warning("[AI] Override chỉ có 1 lớp sau split XGB, sẽ bỏ qua train override.")

        if len(X_test) > 0:
            X_test_lstm = X_test
            y_test_eval = y_test
        else:
            X_test_lstm = np.array([])
            y_test_eval = np.array([])

        # Real-only evaluation split: tính sau khi đã train xong LSTM/XGBoost.
        real_eval_rows = [self._row_to_training_item(r) for r in eval_rows_raw]
        real_eval_rows = [r for r in real_eval_rows if r is not None]
        real_eval_rows = self.db._enrich_training_rows(real_eval_rows)
        real_eval_metrics = {}
        real_eval_cm = {"light": [], "fan": []}

        train_sample_weights = y_train_xgb[:, 4].astype(float) if y_train_xgb.shape[1] > 4 else np.ones(len(y_train_xgb), dtype=float)
        feedback_quality     = y_train_xgb[:, 5].astype(float) if y_train_xgb.shape[1] > 5 else np.ones(len(y_train_xgb), dtype=float)
        sample_weights_xgb   = np.clip(train_sample_weights * np.clip(feedback_quality, 0.5, 1.0), 0.1, 3.0)
        lstm_sample_weights   = np.ones(len(X_tr_lstm), dtype=float)

        # ── LSTM ──────────────────────────────────────────
        lstm_model_temp = self._build_lstm_model(n_features, n_users)
        lstm_model_temp.compile(
            optimizer    = 'adam',
            loss         = {'out_light': 'binary_crossentropy', 'out_fan': 'binary_crossentropy'},
            loss_weights = {'out_light': 0.5, 'out_fan': 0.5},
            metrics      = {'out_light': 'accuracy', 'out_fan': 'accuracy'}
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        logger.info("   -> Kích hoạt mạng Multi-task LSTM (Feature Learning)...")
        lstm_model_temp.fit(
            X_tr_lstm,
            {'out_light': y_tr_lstm[:, 0], 'out_fan': y_tr_lstm[:, 1]},
            epochs          = self.LSTM_EPOCHS,
            batch_size      = self.LSTM_BATCH_SIZE,
            shuffle         = False,
            validation_data = val_data,
            sample_weight   = lstm_sample_weights,
            callbacks       = callbacks,
            verbose         = 0
        )

        fe_light = Model(inputs=lstm_model_temp.input,
                         outputs=lstm_model_temp.get_layer("features_light").output)
        fe_fan   = Model(inputs=lstm_model_temp.input,
                         outputs=lstm_model_temp.get_layer("features_fan").output)

        # ── XGBoost ───────────────────────────────────────
        lstm_feat_xgb_light = fe_light(X_train_xgb, training=False).numpy()
        lstm_feat_xgb_fan   = fe_fan(X_train_xgb,   training=False).numpy()

        pos_l         = max(sum(y_train_xgb[:, 0]), 1)
        neg_l         = max(len(y_train_xgb[:, 0]) - pos_l, 1)
        pos_weight_l  = float(neg_l / pos_l)

        pos_f         = max(sum(y_train_xgb[:, 1]), 1)
        neg_f         = max(len(y_train_xgb[:, 1]) - pos_f, 1)
        pos_weight_f  = float(neg_f / pos_f)

        xgb_light_temp = xgb.XGBClassifier(
            n_estimators     = self.XGB_N_ESTIMATORS,
            learning_rate    = self.XGB_LEARNING_RATE,
            max_depth        = self.XGB_MAX_DEPTH,
            scale_pos_weight = pos_weight_l,
            eval_metric      = 'logloss'
        )
        xgb_fan_temp = xgb.XGBClassifier(
            n_estimators     = self.XGB_N_ESTIMATORS,
            learning_rate    = self.XGB_LEARNING_RATE,
            max_depth        = self.XGB_MAX_DEPTH,
            scale_pos_weight = pos_weight_f,
            eval_metric      = 'logloss'
        )

        # Chỉ dùng trọng số mẫu nếu thực sự có ý nghĩa, tránh làm lệch việc học đèn/quạt.
        try:
            xgb_light_temp.fit(lstm_feat_xgb_light, y_train_xgb[:, 0], sample_weight=sample_weights_xgb)
        except ValueError as e:
            logger.warning(f"[AI] Không train được XGBoost đèn (thiếu lớp hoặc dữ liệu lỗi): {e}")
            xgb_light_temp = None

        try:
            xgb_fan_temp.fit(lstm_feat_xgb_fan, y_train_xgb[:, 1], sample_weight=sample_weights_xgb)
        except ValueError as e:
            logger.warning(f"[AI] Không train được XGBoost quạt (thiếu lớp hoặc dữ liệu lỗi): {e}")
            xgb_fan_temp = None

        xgb_scenario_temp = None
        xgb_override_temp = None
        xgb_scenario_y_raw = y_train_xgb[:, 2].astype(int) if y_train_xgb.shape[1] > 2 else np.zeros(len(y_train_xgb), dtype=int)
        xgb_override_y_all = y_train_override[:, 4].astype(int) if y_train_override.shape[1] > 4 else np.zeros(len(y_train_override), dtype=int)

        if len(xgb_scenario_y_raw) > 0 and len(xgb_scenario_y_raw) == len(lstm_feat_xgb_light):
            try:
                scenario_classes = np.unique(xgb_scenario_y_raw)
                scenario_class_map = {cls: idx for idx, cls in enumerate(scenario_classes)}
                xgb_scenario_y = np.array([scenario_class_map[v] for v in xgb_scenario_y_raw], dtype=int)
                xgb_scenario_temp = xgb.XGBClassifier(
                    n_estimators=self.XGB_N_ESTIMATORS,
                    learning_rate=self.XGB_LEARNING_RATE,
                    max_depth=self.XGB_MAX_DEPTH,
                    objective='multi:softprob',
                    num_class=max(len(scenario_classes), 2),
                    eval_metric='mlogloss'
                )
                if len(np.unique(xgb_scenario_y)) >= 2:
                    xgb_scenario_temp.fit(lstm_feat_xgb_light, xgb_scenario_y, sample_weight=sample_weights_xgb)
                else:
                    xgb_scenario_temp = None
            except ValueError as e:
                logger.warning(f"[AI] Không train được XGBoost scenario: {e}")
                xgb_scenario_temp = None
        else:
            logger.warning(
                f"[AI] Bỏ qua train XGBoost scenario do label không hợp lệ: "
                f"X={len(lstm_feat_xgb_light)}, y={len(xgb_scenario_y_raw)}"
            )

        if len(xgb_override_y_all) > 0:
            try:
                override_pos_idx = np.where(xgb_override_y_all == 1)[0]
                override_neg_idx = np.where(xgb_override_y_all == 0)[0]
                logger.info(
                    f"[AI] Override distribution before split: 0={len(override_neg_idx)} | 1={len(override_pos_idx)}"
                )

                if len(override_pos_idx) == 0 or len(override_neg_idx) == 0:
                    logger.warning(
                        f"[AI] Bỏ qua train XGBoost override vì chỉ có 1 lớp: {np.unique(xgb_override_y_all).tolist()}"
                    )
                    xgb_override_temp = None
                else:
                    # Split riêng cho override, stratify theo override_flag để không rơi về 1 lớp nếu đủ dữ liệu.
                    override_min_count = min(len(override_pos_idx), len(override_neg_idx))
                    override_can_stratify = override_min_count >= 2
                    override_train_size = min(max(4, int(round(len(xgb_override_y_all) * 0.8))), len(xgb_override_y_all) - 1)
                    override_stratify = xgb_override_y_all if override_can_stratify else None
                    try:
                        override_train_idx, _ = train_test_split(
                            np.arange(len(xgb_override_y_all)),
                            train_size=override_train_size,
                            random_state=42,
                            shuffle=True,
                            stratify=override_stratify,
                        )
                        override_train_idx = np.array(sorted(override_train_idx.tolist()), dtype=int)
                    except ValueError:
                        rng = random.Random(42)
                        pos_idx = override_pos_idx.tolist()
                        neg_idx = override_neg_idx.tolist()
                        rng.shuffle(pos_idx)
                        rng.shuffle(neg_idx)
                        target_pos = max(2, int(round(override_train_size * (len(pos_idx) / len(xgb_override_y_all)))))
                        target_neg = max(2, override_train_size - target_pos)
                        target_pos = min(target_pos, len(pos_idx) - 1)
                        target_neg = min(target_neg, len(neg_idx) - 1)
                        override_train_idx = np.array(sorted(pos_idx[:target_pos] + neg_idx[:target_neg]), dtype=int)

                    override_y_train = xgb_override_y_all[override_train_idx]
                    override_feat_train = fe_light(X_temp[override_train_idx], training=False).numpy()
                    override_weights_train = sample_weights_xgb[override_train_idx] if len(sample_weights_xgb) == len(xgb_override_y_all) else np.ones(len(override_train_idx), dtype=float)

                    logger.info(
                        f"[AI] Override after own split: train={dict(zip(*np.unique(override_y_train, return_counts=True)))} | total={dict(zip(*np.unique(xgb_override_y_all, return_counts=True)))}"
                    )

                    if len(np.unique(override_y_train)) < 2:
                        logger.warning(
                            f"[AI] Bỏ qua train XGBoost override vì split train vẫn còn 1 lớp: {np.unique(override_y_train).tolist()}"
                        )
                        xgb_override_temp = None
                    else:
                        xgb_override_temp = xgb.XGBClassifier(
                            n_estimators=self.XGB_N_ESTIMATORS,
                            learning_rate=self.XGB_LEARNING_RATE,
                            max_depth=self.XGB_MAX_DEPTH,
                            eval_metric='logloss'
                        )
                        if len(np.unique(override_y_train)) >= 2:
                            xgb_override_temp.fit(override_feat_train, override_y_train, sample_weight=override_weights_train)
                        else:
                            xgb_override_temp = None
            except ValueError as e:
                logger.warning(f"[AI] Không train được XGBoost override: {e}")
                xgb_override_temp = None
        else:
            logger.warning(
                f"[AI] Bỏ qua train XGBoost override do label không hợp lệ: "
                f"X={len(lstm_feat_xgb_light)}, y={len(xgb_override_y_all)}"
            )

        # ── Evaluate ──────────────────────────────────────
        light_class_metrics = {}
        fan_class_metrics = {}
        if len(X_test_lstm) > 0 and len(y_test_eval) > 0:
            lstm_feat_test_light = fe_light(X_test_lstm, training=False).numpy()
            lstm_feat_test_fan   = fe_fan(X_test_lstm,   training=False).numpy()
            
            light_proba = xgb_light_temp.predict_proba(lstm_feat_test_light)[:, 1] if xgb_light_temp is not None else np.array([])
            pred_l = (light_proba >= getattr(self, "LIGHT_POS_THRESH", 0.30)).astype(int) if len(light_proba) > 0 else np.array([])
            pred_f = xgb_fan_temp.predict(lstm_feat_test_fan) if xgb_fan_temp is not None else np.array([])
            
            if len(pred_l) > 0 and len(pred_f) > 0:
                proba_l = light_proba if xgb_light_temp is not None else np.array([])
                acc_light = accuracy_score(y_test_eval[:, 0], pred_l)
                acc_fan   = accuracy_score(y_test_eval[:, 1], pred_f)
                
                f1_light  = f1_score(y_test_eval[:, 0], pred_l, zero_division=0)
                f1_fan    = f1_score(y_test_eval[:, 1], pred_f, zero_division=0)
                f1_light_macro = f1_score(y_test_eval[:, 0], pred_l, average="macro", zero_division=0)
                f1_fan_macro    = f1_score(y_test_eval[:, 1], pred_f, average="macro", zero_division=0)

                test_light_counts = {int(k): int(v) for k, v in zip(*np.unique(y_test_eval[:, 0], return_counts=True))}
                test_fan_counts = {int(k): int(v) for k, v in zip(*np.unique(y_test_eval[:, 1], return_counts=True))}
                logger.info(f"[AI Eval] Phân bố test đèn: {test_light_counts} | quạt: {test_fan_counts} | split={eval_split_mode}")
                logger.info(
                    f"[AI Eval] Train XGB đèn: {dict(zip(*np.unique(y_train_xgb[:, 0], return_counts=True)))} | "
                    f"Train XGB quạt: {dict(zip(*np.unique(y_train_xgb[:, 1], return_counts=True)))}"
                )
                logger.info(f"[AI Eval] Macro F1 Đèn: {f1_light_macro:.2f}, Quạt: {f1_fan_macro:.2f}")
                
                light_report = classification_report(y_test_eval[:, 0], pred_l, labels=[0, 1], output_dict=True, zero_division=0)
                fan_report = classification_report(y_test_eval[:, 1], pred_f, labels=[0, 1], output_dict=True, zero_division=0)
                light_class_metrics = {
                    cls: {
                        "precision": round(float(vals.get("precision", 0.0)), 4),
                        "recall": round(float(vals.get("recall", 0.0)), 4),
                        "f1": round(float(vals.get("f1-score", 0.0)), 4),
                        "support": int(vals.get("support", 0)),
                    }
                    for cls, vals in light_report.items()
                    if cls not in ("accuracy", "macro avg", "weighted avg")
                }
                fan_class_metrics = {
                    cls: {
                        "precision": round(float(vals.get("precision", 0.0)), 4),
                        "recall": round(float(vals.get("recall", 0.0)), 4),
                        "f1": round(float(vals.get("f1-score", 0.0)), 4),
                        "support": int(vals.get("support", 0)),
                    }
                    for cls, vals in fan_report.items()
                    if cls not in ("accuracy", "macro avg", "weighted avg")
                }

                # Chọn threshold theo validation để ưu tiên recall cho light.
                best_thresh = getattr(self, "LIGHT_POS_THRESH", 0.30)
                best_score = -1.0
                if len(proba_l) > 0:
                    thresholds = np.round(np.arange(0.15, 0.61, 0.03), 2)
                    for thresh in thresholds:
                        adj_pred = (proba_l >= thresh).astype(int)
                        adj_recall = recall_score(y_test_eval[:, 0], adj_pred, zero_division=0)
                        adj_prec = precision_score(y_test_eval[:, 0], adj_pred, zero_division=0)
                        balanced = (2.0 * adj_recall + adj_prec) / 3.0
                        if balanced > best_score:
                            best_score = balanced
                            best_thresh = float(thresh)
                    self.LIGHT_POS_THRESH = best_thresh

                logger.info(f"[AI Eval] Đèn  - Accuracy: {acc_light:.2%}, F1-Score: {f1_light:.2f}, thresh={best_thresh:.2f}")
                logger.info(f"[AI Eval] Quạt - Accuracy: {acc_fan:.2%}, F1-Score: {f1_fan:.2f}")
                
                cm_light = confusion_matrix(y_test_eval[:, 0], pred_l, labels=[0, 1]).tolist()
                logger.info(f"[AI Eval] Confusion Matrix Đèn:\n{cm_light}")
            else:
                acc_light = acc_fan = 0.0
                f1_light = f1_fan = 0.0
                cm_light = []
        else:
            acc_light = acc_fan = 0.0
            f1_light = f1_fan = 0.0
            cm_light = []

        if real_eval_metrics:
            logger.info(
                f"[AI Eval] Real-only split - light={real_eval_metrics.get('light_accuracy', 0):.2%}, "
                f"fan={real_eval_metrics.get('fan_accuracy', 0):.2%}"
            )

        # ── Atomic update ─────────────────────────────────
        with self._lock:
            self.lstm_model       = lstm_model_temp
            self.fe_light         = fe_light
            self.fe_fan           = fe_fan
            self.xgb_light        = xgb_light_temp
            self.xgb_fan          = xgb_fan_temp
            self.xgb_scenario     = xgb_scenario_temp
            self.xgb_override     = xgb_override_temp
            self.scaler           = scaler_temp
            self.label_encoder    = label_encoder_temp
            self.is_trained       = True

        self._save_model()

        if len(real_eval_rows) >= self.TIME_STEPS + 2:
            try:
                real_X_raw = self._encode_features(real_eval_rows, custom_encoder=label_encoder_temp)
                real_X_num = real_X_raw[:, :-1]
                real_X_usr = real_X_raw[:, -1:]
                real_X_scaled = np.column_stack([scaler_temp.transform(real_X_num), real_X_usr])
                real_Y = np.column_stack([
                    [r["light_state"] for r in real_eval_rows],
                    [r["fan_state"] for r in real_eval_rows],
                ])
                real_seq_X, real_seq_Y, real_seq_meta = self._create_sequences(real_X_scaled, real_eval_rows, real_Y, self.TIME_STEPS)
                if len(real_seq_X) > 0 and xgb_light_temp is not None and xgb_fan_temp is not None:
                    real_ext_l = fe_light(real_seq_X, training=False).numpy()
                    real_ext_f = fe_fan(real_seq_X, training=False).numpy()
                    real_pred_l = xgb_light_temp.predict(real_ext_l)
                    real_pred_f = xgb_fan_temp.predict(real_ext_f)
                    real_eval_metrics = {
                        "light_accuracy": float(accuracy_score(real_seq_Y[:, 0], real_pred_l)),
                        "fan_accuracy": float(accuracy_score(real_seq_Y[:, 1], real_pred_f)),
                        "light_f1": float(f1_score(real_seq_Y[:, 0], real_pred_l, zero_division=0)),
                        "fan_f1": float(f1_score(real_seq_Y[:, 1], real_pred_f, zero_division=0)),
                        "light_macro_f1": float(f1_score(real_seq_Y[:, 0], real_pred_l, average="macro", zero_division=0)),
                        "fan_macro_f1": float(f1_score(real_seq_Y[:, 1], real_pred_f, average="macro", zero_division=0)),
                    }
                    real_eval_cm["light"] = confusion_matrix(real_seq_Y[:, 0], real_pred_l, labels=[0, 1]).tolist()
                    real_eval_cm["fan"] = confusion_matrix(real_seq_Y[:, 1], real_pred_f, labels=[0, 1]).tolist()
            except Exception as e:
                logger.warning(f"[AI Eval] Không đánh giá được real-only split: {e}")

        try:
            # Không warm-up lại từ dữ liệu ngay sau huấn luyện để tránh lẫn
            # ngữ cảnh gần thời điểm test vào buffer vận hành.
            logger.info("[AI] Bỏ qua warm-up sau train để giảm nguy cơ leakage ngữ cảnh.")
        except Exception as e:
            logger.error(f"[AI] Lỗi khi xử lý hậu train: {e}")

        data_source_counts = {
            "simulated": int(sum(1 for r in rows if (r.get("data_source") or "") == "simulated")),
            "real": int(sum(1 for r in rows if (r.get("data_source") or "") == "real")),
            "hybrid": int(sum(1 for r in rows if (r.get("data_source") or "") == "hybrid")),
        }

        override_rows = int(sum(1 for r in rows if int(r.get("override_flag", 0)) == 1))
        scenario_counts = {
            "sequence_behavior": int(sum(1 for r in rows if (r.get("scenario_type") or "") == "sequence_behavior")),
            "context_exception": int(sum(1 for r in rows if (r.get("scenario_type") or "") == "context_exception")),
            "manual_override": int(sum(1 for r in rows if (r.get("scenario_type") or "") == "manual_override")),
            "profile_personalization": int(sum(1 for r in rows if (r.get("scenario_type") or "") == "profile_personalization")),
            "background_noise": int(sum(1 for r in rows if (r.get("scenario_type") or "") == "background_noise")),
        }
        total_scenarios = max(sum(scenario_counts.values()), 1)
        dominant_scenario = max(scenario_counts, key=scenario_counts.get) if scenario_counts else "unknown"
        real_rows = int(data_source_counts.get("real", 0))
        real_only_available = bool(real_eval_metrics)
        quality_flags = {
            "has_real_data": real_rows > 0,
            "real_only_eval_available": real_only_available,
            "override_ratio": round(override_rows / max(len(rows), 1), 4),
            "dominant_scenario": dominant_scenario,
        }
        recommendations = []
        quality_audit = self._run_quality_audit(
            rows=rows,
            label_encoder=label_encoder_temp,
            scaler=scaler_temp,
            feature_matrix=X_numeric_unscaled,
            y_light=y_light,
            y_fan=y_fan,
            y_scenario=train_scenario_codes,
            y_sequence_phase=train_sequence_phase_codes,
            y_override=train_override_flags,
        )
        if quality_audit.get("status") != "pass":
            recommendations.append(f"Dữ liệu train cần xem lại: {', '.join(quality_audit.get('warnings', []))}")
        if not real_only_available:
            recommendations.append("Bổ sung dữ liệu thực để đo khả năng tổng quát hóa ngoài mô phỏng.")
        if override_rows / max(len(rows), 1) < 0.05:
            recommendations.append("Tăng số mẫu override để mô hình học tốt hơn các tình huống manual.")
        if max(scenario_counts.values()) / total_scenarios > 0.5:
            recommendations.append("Giảm lệch kịch bản bằng cách cân bằng thêm các scenario ít xuất hiện.")
        if not recommendations:
            recommendations.append("Tiếp tục theo dõi real-only eval và bổ sung dữ liệu thật khi có thể.")
        if eval_source == "real" and data_source == "real":
            recommendations.append("Bạn đang train và eval cùng nguồn real; để đo generalization, nên train simulated/hybrid và eval real-only.")
        if total_scenarios > 0 and max(scenario_counts.values()) / total_scenarios > 0.35:
            recommendations.append("Nên giảm độ lệch kịch bản để mô hình không học quá cứng vào 4 sự kiện mà vẫn giữ tổng quát hóa.")

        report = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_name,
            "data_source": data_source,
            "eval_source": eval_source,
            "eval_split_mode": eval_split_mode,
            "total_rows": int(total),
            "rows_after_filter": int(len(rows)),
            "eval_rows": int(len(real_eval_rows)),
            "sequence_steps": int(self.TIME_STEPS),
            "train_split": 0.85,
            "lstm_epochs": int(self.LSTM_EPOCHS),
            "lstm_batch_size": int(self.LSTM_BATCH_SIZE),
            "xgb_estimators": int(self.XGB_N_ESTIMATORS),
            "xgb_max_depth": int(self.XGB_MAX_DEPTH),
            "xgb_learning_rate": float(self.XGB_LEARNING_RATE),
            "accuracy_light": round(acc_light * 100, 2),
            "accuracy_fan": round(acc_fan * 100, 2),
            "f1_light": round(f1_light, 4),
            "f1_fan": round(f1_fan, 4),
            "light_class_metrics": light_class_metrics,
            "fan_class_metrics": fan_class_metrics,
            "override_rows": override_rows,
            "data_source_counts": data_source_counts,
            "scenario_counts": scenario_counts,
            "quality_flags": quality_flags,
            "quality_audit": quality_audit,
            "recommendations": recommendations,
            "real_only_eval": {
                "source": eval_source,
                "metrics": real_eval_metrics,
                "confusion_matrices": real_eval_cm,
            },
            "confusion_matrices": {
                "light": cm_light,
            },
        }

        try:
            report_dir = Path(getattr(config, "DATA_DIR", os.path.dirname(self.db.db_path)))
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "training_report.json"
            audit_path = report_dir / "training_quality_audit.json"
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            with audit_path.open("w", encoding="utf-8") as f:
                json.dump(quality_audit, f, ensure_ascii=False, indent=2)
            logger.info(f"[AI] Đã lưu báo cáo huấn luyện: {report_path}")
            logger.info(f"[AI] Đã lưu báo cáo chất lượng: {audit_path}")
        except Exception as e:
            logger.warning(f"[AI] Không lưu được báo cáo huấn luyện: {e}")

        return {
            "success": True,
            "data_source": data_source,
            "eval_source": eval_source,
            "accuracy_light": round(acc_light * 100, 2),
            "accuracy_fan": round(acc_fan * 100, 2),
            "f1_light": round(f1_light, 4),
            "f1_fan": round(f1_fan, 4),
            "light_class_metrics": light_class_metrics,
            "fan_class_metrics": fan_class_metrics,
            "quality_flags": quality_flags,
            "recommendations": recommendations,
            "confusion_matrices": {
                "light": cm_light,
            },
            "real_only_eval": {
                "source": eval_source,
                "metrics": real_eval_metrics,
                "confusion_matrices": real_eval_cm,
            },
            "report_path": str((Path(getattr(config, "DATA_DIR", os.path.dirname(self.db.db_path))) / "training_report.json")),
            "audit_path": str((Path(getattr(config, "DATA_DIR", os.path.dirname(self.db.db_path))) / "training_quality_audit.json")),
        }

    # ─────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────
    def _finalize_prediction(self, *, hour: int, day_of_week: int, temp: float, hum: float,
                             light: float, motion: bool, user_id: str, pir_handled: bool,
                             candidate_light: int, candidate_fan: int, confidence: float,
                             source_parts: list[str], hysteresis_meta: dict) -> dict:
        confidence = float(np.clip(confidence, 0.0, 1.0))
        source_final = "+".join(source_parts)
        decision = DeviceDecision(
            light=int(candidate_light),
            fan=int(candidate_fan),
            confidence=round(confidence, 2),
            source=source_final,
            user_id=user_id,
            shadow=False,
            pir_handled=pir_handled,
        )
        result = decision.__dict__
        result["hysteresis_meta"] = hysteresis_meta
        result["needs_review"] = bool(confidence < min(self.LIGHT_CONF_THRESH, self.FAN_CONF_THRESH))
        result["review_reason"] = "low_confidence" if result["needs_review"] else ""
        result["feature_importance"] = self.get_feature_importance(
            hour=hour, day_of_week=day_of_week, temp=temp, hum=hum, light=light, motion=motion, user_id=user_id
        )
        result["hysteresis"] = hysteresis_meta
        return result

    def predict(self, hour: int, day_of_week: int, temp: float,
                hum: float, light: float, motion: bool = False,
                user_id: str = "unknown", pir_handled: bool = False) -> dict:
        self._cleanup_stale_user_state()

        normalized_user_id = self._normalize_user_label(user_id)

        if normalized_user_id == "guest":
            user_id = "global_average"
        elif normalized_user_id == "unknown":
            user_id = "unknown"
        else:
            user_id = normalized_user_id

        self._touch_user_state(user_id)

        if user_id == "unknown":
            fallback = self._rule_based_fallback_v2(temp, light, motion, user_id="global_average")
            if light < config.ENV_THRESHOLDS.get("light_dark", 100) or motion:
                fallback["light"] = 1
            fallback["fan"] = 1 if temp > config.ENV_THRESHOLDS.get("temp_high", 29.0) and motion else 0
            fallback["confidence"] = 0.25
            fallback["source"] = "yolo_unknown_safe_fallback"
            fallback["user_id"] = "global_average"
            return fallback

        if self._is_manual_override_active(user_id):
            with self._lock:
                last_light = self.last_light_state.get(user_id)
                last_fan = self.last_fan_state.get(user_id)
            shadow = self._rule_based_fallback_v2(temp, light, motion, user_id)
            shadow["source"] = "manual_override_cooldown"
            shadow["user_id"] = user_id
            shadow["shadow_run"] = True
            shadow["manual_override_active"] = True
            shadow["light"] = int(last_light) if last_light is not None else int(shadow.get("light", 0))
            shadow["fan"] = int(last_fan) if last_fan is not None else int(shadow.get("fan", 0))
            shadow["confidence"] = min(float(shadow.get("confidence", 0.25)), 0.35)
            self._touch_user_state(user_id)
            return shadow

        VALID_TEMP = (-10.0, 60.0)
        VALID_HUM  = (5.0,  100.0)
            
        if not (VALID_TEMP[0] < temp < VALID_TEMP[1] and VALID_HUM[0] < hum < VALID_HUM[1]):
            logger.warning("[AI] Dữ liệu nhiễu lọt vào Pipeline! Ép chuyển sang Rule-based Tối giản.")
            return self._rule_based_fallback_v2(temp, light, motion, user_id)

        with self._lock:
            is_trained      = self.is_trained
            fe_light        = self.fe_light
            fe_fan          = self.fe_fan
            xgb_light       = self.xgb_light
            xgb_fan         = self.xgb_fan
            current_scaler  = self.scaler
            current_encoder = self.label_encoder

        if not is_trained or fe_light is None or fe_fan is None or xgb_light is None or xgb_fan is None:
            fallback_res = self._rule_based_fallback_v2(temp, light, motion, user_id)
            fallback_res["source"] = "model_unavailable"
            fallback_res["user_id"] = user_id
            return fallback_res

        if current_scaler is None or current_encoder is None:
            fallback_res = self._rule_based_fallback_v2(temp, light, motion, user_id)
            fallback_res["source"] = "model_missing_preprocessor"
            fallback_res["user_id"] = user_id
            return fallback_res

        current_scaled = None
        if is_trained and current_scaler is not None and current_encoder is not None:
            current_feature = self._encode_features([{
                "hour": hour, "day_of_week": day_of_week,
                "temperature": temp, "humidity": hum,
                "light_level": light,
                "motion_detected": 1 if motion else 0,
                "user_id": user_id
            }], custom_encoder=current_encoder)
            
            X_num = current_feature[:, :-1]
            X_usr = current_feature[:, -1:]
            X_num_scaled = current_scaler.transform(X_num)
            current_scaled = np.column_stack([X_num_scaled, X_usr])[0]

        seq_input = None

        with self._lock:
    
            now       = datetime.now()
            last_time = self.last_predict_time.get(user_id)
            if last_time and (now - last_time).total_seconds() > 1800:
                logger.info(f"[AI] User {user_id} vắng mặt > 30 phút. Làm sạch Memory.")
                if user_id in self.user_buffers:
                    self.user_buffers[user_id].clear()
                self.last_fan_state.pop(user_id, None)
                self.last_light_state.pop(user_id, None)

            self.last_predict_time[user_id] = now

            if current_scaled is not None:
                if user_id not in self.user_buffers or len(self.user_buffers[user_id]) == 0:
                    self.user_buffers[user_id] = [current_scaled] * self.TIME_STEPS
                else:
                    self.user_buffers[user_id].append(current_scaled)
                    if len(self.user_buffers[user_id]) > self.TIME_STEPS:
                        self.user_buffers[user_id].pop(0)
                
                if len(self.user_buffers[user_id]) == self.TIME_STEPS:
                    seq_input = np.array([self.user_buffers[user_id]])

        # ── Tầng 2/3: Fallback khi buffer chưa đủ hoặc chưa train ────────────
        if seq_input is None or not is_trained:
            fallback_res = self._rule_based_fallback_v2(temp, light, motion, user_id)
            fallback_res["source"] = "rule_minimal_buffering" if is_trained else "not_trained"
            fallback_res["user_id"] = user_id

            with self._lock:
                self.last_light_state[user_id] = fallback_res["light"]
                self.last_fan_state[user_id]   = fallback_res["fan"]
            return fallback_res

        # ── Tầng 1: AI Inference + XGBoost, sau đó đi qua bộ hysteresis độc lập ──────────
        ext_l = fe_light(seq_input, training=False).numpy()
        ext_f = fe_fan(seq_input,   training=False).numpy()

        proba_l    = xgb_light.predict_proba(ext_l)[0]
        proba_f    = xgb_fan.predict_proba(ext_f)[0]

        pred_l_idx = int(np.argmax(proba_l))
        pred_f_idx = int(np.argmax(proba_f))

        pred_l = int(xgb_light.classes_[pred_l_idx]) if hasattr(xgb_light, 'classes_') else pred_l_idx
        pred_f = int(xgb_fan.classes_[pred_f_idx])   if hasattr(xgb_fan,   'classes_') else pred_f_idx

        conf_l = float(proba_l[pred_l_idx]) if len(proba_l) else 0.0
        conf_f = float(proba_f[pred_f_idx]) if len(proba_f) else 0.0

        dynamic_thresh_l = self.LIGHT_CONF_THRESH
        dynamic_thresh_f = self.FAN_CONF_THRESH

        source_parts = []
        with self._lock:
            if conf_l >= dynamic_thresh_l:
                candidate_light = pred_l
                source_parts.append("ai_light")
            else:
                candidate_light, src_l = self._hysteresis_or_rule_light(user_id, light, motion)
                source_parts.append(src_l)

            if conf_f >= dynamic_thresh_f:
                candidate_fan = pred_f
                source_parts.append("ai_fan_guest_average" if user_id == "global_average" else "ai_fan")
            else:
                last_fan = self.last_fan_state.get(user_id)
                if last_fan is not None:
                    candidate_fan = last_fan
                    source_parts.append("hysteresis_fan_hold")
                else:
                    candidate_fan = 1 if temp > config.ENV_THRESHOLDS.get("temp_high", 29.0) else 0
                    source_parts.append("rule_fan_minimal")

        final_light, final_fan, hysteresis_meta = self.hysteresis.apply(
            user_id,
            temp=temp,
            light=light,
            motion=motion,
            desired_light=candidate_light,
            desired_fan=candidate_fan,
        )
        source_parts.append("hysteresis_gate")

        with self._lock:
            self.last_light_state[user_id] = final_light
            self.last_fan_state[user_id]   = final_fan

        confidence   = (conf_l * 0.55) + (conf_f * 0.45)
        confidence   = float(np.clip(confidence, 0.0, 1.0))
        source_final = "+".join(source_parts)

        decision = DeviceDecision(
            light=final_light,
            fan=final_fan,
            confidence=round(confidence, 2),
            source=source_final,
            user_id=user_id,
            shadow=False,
            pir_handled=pir_handled,
        )
        result = decision.__dict__
        result["hysteresis_meta"] = hysteresis_meta
        result["needs_review"] = bool(confidence < min(self.LIGHT_CONF_THRESH, self.FAN_CONF_THRESH))
        result["review_reason"] = "low_confidence" if result["needs_review"] else ""
        result["feature_importance"] = self.get_feature_importance(
            hour=hour,
            day_of_week=day_of_week,
            temp=temp,
            hum=hum,
            light=light,
            motion=motion,
            user_id=user_id,
        )
        result["hysteresis"] = hysteresis_meta
        return result

    def get_feature_importance(self, hour: int, day_of_week: int, temp: float,
                               hum: float, light: float, motion: bool = False,
                               user_id: str = "unknown") -> dict:
        with self._lock:
            if not self.is_trained or self.fe_light is None or self.fe_fan is None:
                return {"features": [], "light": [], "fan": []}
            xgb_light = self.xgb_light
            xgb_fan = self.xgb_fan

        def _extract(model):
            if model is None or not hasattr(model, "feature_importances_"):
                return np.zeros(len(AI_FEATURE_NAMES), dtype=float)
            values = np.asarray(model.feature_importances_, dtype=float)
            if values.size < len(AI_FEATURE_NAMES):
                values = np.pad(values, (0, len(AI_FEATURE_NAMES) - values.size))
            elif values.size > len(AI_FEATURE_NAMES):
                values = values[:len(AI_FEATURE_NAMES)]
            total = float(values.sum())
            if total > 0:
                values = values / total
            return values

        light_imp = _extract(xgb_light)
        fan_imp = _extract(xgb_fan)
        return {
            "features": AI_FEATURE_NAMES,
            "light": [round(float(v), 6) for v in light_imp.tolist()],
            "fan": [round(float(v), 6) for v in fan_imp.tolist()],
        }

    def _hysteresis_or_rule_light(self, user_id: str, light: float, motion: bool) -> tuple:
        last_light = self.last_light_state.get(user_id)
        dark_thresh = config.ENV_THRESHOLDS.get("light_dark", 100)
        bright_thresh = config.ENV_THRESHOLDS.get("light_bright", dark_thresh + 120)
        pos_thresh = getattr(self, "LIGHT_POS_THRESH", 0.58)

        if last_light == 1:
            if motion or light < bright_thresh:
                return 1, "hysteresis_light"
            return 0, "hysteresis_light"

        if motion and light < dark_thresh * 0.92:
            return 1, "rule_light_minimal"
        if light <= dark_thresh * pos_thresh:
            return 1, "rule_light_dim"
        return 0, "rule_light_minimal"

    def _rule_based_fallback_v2(self, temp: float, light: float,
                                motion: bool = False, user_id: str = "unknown") -> dict:
        with self._lock:
            last_light = self.last_light_state.get(user_id)
            last_fan = self.last_fan_state.get(user_id)

        decision = self.rule_engine.decide(
            temp=temp,
            light=light,
            motion=motion,
            user_id=user_id,
            allow_fan_when_hot=True,
            empty_room_safe=True,
        )
        if last_light is not None:
            decision["light"] = int(last_light) if motion else decision["light"]
        if last_fan is not None and temp <= config.ENV_THRESHOLDS.get("temp_high", 29.0):
            decision["fan"] = int(last_fan)
        decision["confidence"] = float(decision.get("confidence", 0.25))
        decision.setdefault("source", "rule_based_fallback")
        decision.setdefault("user_id", user_id)
        return decision

    # ─────────────────────────────────────────────────────
    def clear_user_memory(self, user_id: str):
        if user_id == "unknown":
            return
        if user_id == "guest":
            user_id = "global_average"

        with self._lock:
            self.last_fan_state.pop(user_id, None)
            self.last_light_state.pop(user_id, None)
            self.user_buffers.pop(user_id, None)
            self.last_predict_time.pop(user_id, None)
            self._user_state_last_seen.pop(user_id, None)

    # ─────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────
    def _save_model(self):
        with self._lock:
            if self.lstm_model is None or self.xgb_light is None or self.xgb_fan is None:
                return
            lstm_model    = self.lstm_model
            xgb_light     = self.xgb_light
            xgb_fan       = self.xgb_fan
            scaler        = self.scaler
            label_encoder = self.label_encoder
            time_steps    = self.TIME_STEPS
            n_features    = self.scaler.n_features_in_ + 1 if self.scaler else 9

        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        base_path = config.MODEL_PATH.replace('.pkl', '')

        try:
            lstm_model.save(f"{base_path}_tmp_lstm_full.keras")
            os.replace(f"{base_path}_tmp_lstm_full.keras", f"{base_path}_lstm_full.keras")

            lstm_model.save_weights(f"{base_path}_tmp_lstm.weights.h5")
            os.replace(f"{base_path}_tmp_lstm.weights.h5", f"{base_path}_lstm.weights.h5")

            xgb_light.save_model(f"{base_path}_xgb_light.tmp.json")
            os.replace(f"{base_path}_xgb_light.tmp.json", f"{base_path}_xgb_light.json")

            xgb_fan.save_model(f"{base_path}_xgb_fan.tmp.json")
            os.replace(f"{base_path}_xgb_fan.tmp.json", f"{base_path}_xgb_fan.json")

            meta_path     = f"{base_path}_meta.joblib"
            existing_data = joblib.load(meta_path) if os.path.exists(meta_path) else {}
            existing_data.update({
                "scaler":        scaler,
                "label_encoder": label_encoder,
                "time_steps":    time_steps,
                "n_features":    n_features
            })

            joblib.dump(existing_data, f"{base_path}_meta.tmp.joblib")
            os.replace(f"{base_path}_meta.tmp.joblib", f"{base_path}_meta.joblib")

        except Exception as e:
            logger.error(f"[AI] Lỗi trong quá trình Atomic Save Model: {e}")

    def _load_model(self):
        base_path = config.MODEL_PATH.replace('.pkl', '')
        meta_path = f"{base_path}_meta.joblib"

        if not os.path.exists(meta_path):
            return

        try:
            data = joblib.load(meta_path)

            saved_time_steps = data.get("time_steps", self.TIME_STEPS)
            if saved_time_steps != self.TIME_STEPS:
                logger.warning(
                    f"[AI] TIME_STEPS thay đổi ({saved_time_steps} → {self.TIME_STEPS}). "
                    f"Bỏ qua load model, yêu cầu Train lại."
                )
                return

            self.scaler        = data.get("scaler")
            self.label_encoder = data.get("label_encoder")

            if (os.path.exists(f"{base_path}_xgb_light.json") and
                    os.path.exists(f"{base_path}_xgb_fan.json")):
                self.xgb_light = xgb.XGBClassifier()
                self.xgb_light.load_model(f"{base_path}_xgb_light.json")
                self.xgb_fan   = xgb.XGBClassifier()
                self.xgb_fan.load_model(f"{base_path}_xgb_fan.json")


            full_model_path = f"{base_path}_lstm_full.keras"
            if os.path.exists(full_model_path):
                try:
                    self.lstm_model = tf.keras.models.load_model(full_model_path, safe_mode=False)
                    self.fe_light   = Model(inputs=self.lstm_model.input,
                                            outputs=self.lstm_model.get_layer("features_light").output)
                    self.fe_fan     = Model(inputs=self.lstm_model.input,
                                            outputs=self.lstm_model.get_layer("features_fan").output)

                    self.is_trained = all([
                        self.lstm_model, self.fe_light, self.fe_fan,
                        self.xgb_light,  self.xgb_fan,  self.scaler
                    ])
                except Exception as ve:
                    logger.warning(f"[AI] Lỗi khi load full model: {ve}. Thử build lại từ weights.")
                    self.is_trained = False

            if not self.is_trained and os.path.exists(f"{base_path}_lstm.weights.h5"):
                n_features = data.get("n_features", self.scaler.n_features_in_ + 1 if self.scaler else 9)
                n_users    = len(self.label_encoder.classes_) if self.label_encoder else 1

                try:
                    self.lstm_model = self._build_lstm_model(n_features, n_users)
                    self.lstm_model.load_weights(f"{base_path}_lstm.weights.h5")
                    self.fe_light   = Model(inputs=self.lstm_model.input,
                                            outputs=self.lstm_model.get_layer("features_light").output)
                    self.fe_fan     = Model(inputs=self.lstm_model.input,
                                            outputs=self.lstm_model.get_layer("features_fan").output)

                    self.is_trained = all([
                        self.lstm_model, self.fe_light, self.fe_fan,
                        self.xgb_light,  self.xgb_fan,  self.scaler
                    ])
                except ValueError as ve:
                    logger.warning(
                        f"[AI] Kiến trúc mạng Neural đã thay đổi. "
                        f"Cần Train lại để tạo file Weights mới: {ve}"
                    )
                    self.is_trained = False
                    return

        except Exception as e:
            logger.error(f"[AI] Lỗi phân tích khôi phục Model: {e}")