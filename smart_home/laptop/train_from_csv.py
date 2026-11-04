
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from bisect import bisect_left
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import config

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH =  Path("custom_scenario.csv")
OUT_PATH = Path(getattr(config, "DATA_DIR", BASE_DIR / "data")) / "custom_scenario_prepared.json"

REQUIRED_COLUMNS = {
    "timestamp",
    "hour",
    "minute",
    "day_of_week",
    "temperature",
    "humidity",
    "light_level",
    "motion_detected",
    "user_id",
    "dominant_user_id",
    "presence_type",
    "face_count",
    "light_state",
    "fan_state",
    "confidence",
    "source",
    "scenario_type",
    "sequence_phase",
    "override_flag",
    "override_reason",
    "session_id",
    "decision_mode",
    "data_source",
    
}

SESSION_COLUMNS = {
    "session_id",
    "user_id",
    "dominant_user_id",
    "active_users",
    "presence_type",
    "decision_mode",
    "started_at",
    "ended_at",
    "end_reason",
    "event_count",
    "last_event_at",
    "is_active",
    "metadata_json",
}

OPTIONAL_COLUMNS = {
    "decision_mode",
}


@dataclass
class PreparedSummary:
    total_rows: int
    valid_rows: int
    invalid_rows: int
    scenario_counts: dict[str, int]
    user_counts: dict[str, int]
    override_count: int
    sequence_count: int
    context_exception_count: int
    manual_override_count: int
    background_noise_count: int
    feature_names: list[str]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(raw)
    return rows


def _parse_session_rows(session_path: Path) -> list[dict[str, Any]]:
    if not session_path.exists():
        return []
    with session_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        if not SESSION_COLUMNS.intersection(set(reader.fieldnames)):
            return []
        return list(reader)


def _validate_columns(fieldnames: list[str]) -> None:
    missing = REQUIRED_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {sorted(missing)}")


def _prepare_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], PreparedSummary]:
    prepared: list[dict[str, Any]] = []
    invalid_rows = 0

    def _infer_scenario(row: dict[str, Any], override_flag: int, override_reason: str) -> str:
        scenario = (row.get("scenario_type") or "").strip().lower()
        if scenario:
            return scenario
        source = (row.get("source") or "").strip().lower()
        if override_flag == 1 or "manual" in source or override_reason:
            return "manual_override"
        if source.startswith("ai"):
            return "sequence_behavior"
        return "background_noise"

    scenario_counts = Counter()
    user_counts = Counter()
    override_count = 0
    seq_count = 0
    ctx_count = 0
    manual_count = 0
    bg_count = 0

    for row in rows:
        try:
            ts = row.get("timestamp", "")
            if isinstance(ts, str):
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            else:
                dt = datetime.fromtimestamp(float(ts))

            override_flag = _to_int(row.get("override_flag", 0), 0)
            override_reason = (row.get("override_reason") or "").strip()
            scenario = _infer_scenario(row, override_flag, override_reason)
            user_id = (row.get("user_id") or "unknown").strip()
            sequence_phase = (row.get("sequence_phase") or "").strip().lower()
            dominant_user_id = (row.get("dominant_user_id") or user_id).strip()
            confidence = _to_float(row.get("confidence", 0.0), 0.0)
            decision_mode = (row.get("decision_mode") or "single_user").strip().lower()

            item = {
                "timestamp": dt.isoformat(sep=" "),
                "hour": _to_int(row.get("hour", dt.hour), dt.hour),
                "minute": _to_int(row.get("minute", dt.minute), dt.minute),
                "day_of_week": _to_int(row.get("day_of_week", dt.weekday()), dt.weekday()),
                "temperature": _to_float(row.get("temperature", 0.0)),
                "humidity": _to_float(row.get("humidity", 0.0)),
                "light_level": _to_float(row.get("light_level", 0.0)),
                "motion_detected": _to_int(row.get("motion_detected", 0), 0),
                "user_id": user_id,
                "dominant_user_id": dominant_user_id,
                "light_state": _to_int(row.get("light_state", 0), 0),
                "fan_state": _to_int(row.get("fan_state", 0), 0),
                "presence_type": (row.get("presence_type") or "unknown").strip().lower(),
                "face_count": _to_int(row.get("face_count", 0), 0),
                "temp_delta": abs(_to_float(row.get("temperature", 0.0)) - _to_float(row.get("temperature", 0.0))),
                "humidity_delta": abs(_to_float(row.get("humidity", 0.0)) - _to_float(row.get("humidity", 0.0))),
                "light_delta": abs(_to_float(row.get("light_level", 0.0)) - _to_float(row.get("light_level", 0.0))),
                "confidence": confidence,
                "scenario_type": scenario,
                "sequence_phase": sequence_phase,
                "override_flag": override_flag,
                "override_reason": override_reason,
                "source": (row.get("source") or "real").strip().lower(),
                "data_source": (row.get("data_source") or "real").strip().lower(),
                "decision_mode": decision_mode,
            }
            prepared.append(item)

            scenario_counts[scenario] += 1
            user_counts[user_id] += 1
            override_count += 1 if override_flag else 0
            if scenario == "sequence_behavior":
                seq_count += 1
            elif scenario == "context_exception":
                ctx_count += 1
            elif scenario == "manual_override":
                manual_count += 1
            else:
                bg_count += 1
        except Exception:
            invalid_rows += 1

    summary = PreparedSummary(
        total_rows=len(rows),
        valid_rows=len(prepared),
        invalid_rows=invalid_rows,
        scenario_counts=dict(scenario_counts),
        user_counts=dict(user_counts),
        override_count=override_count,
        sequence_count=seq_count,
        context_exception_count=ctx_count,
        manual_override_count=manual_count,
        background_noise_count=bg_count,
        feature_names=[
            "hour",
            "minute",
            "day_of_week",
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
            "face_count",
            "scenario_type",
            "sequence_phase",
            "override_flag",
            "confidence",
            "decision_mode",
        ],
    )
    return prepared, summary


def _load_session_augmented_rows(rows: list[dict[str, Any]], session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not session_rows:
        return rows

    session_index: dict[str, dict[str, Any]] = {}
    for s in session_rows:
        sid = (s.get("session_id") or "").strip()
        if sid:
            session_index[sid] = s

    augmented = []
    for row in rows:
        sid = (row.get("session_id") or "").strip()
        sess = session_index.get(sid)
        row = dict(row)
        if sess:
            row["session_event_count"] = _to_int(sess.get("event_count", 0), 0)
            row["session_duration_sec"] = _session_duration_seconds(sess)
            row["session_is_active"] = _to_int(sess.get("is_active", 0), 0)
            row["session_start_hour"] = _safe_dt(sess.get("started_at")).hour if _safe_dt(sess.get("started_at")) else 0
            row["session_end_hour"] = _safe_dt(sess.get("ended_at")).hour if _safe_dt(sess.get("ended_at")) else 0
            row["session_end_reason"] = (sess.get("end_reason") or "").strip().lower()
        else:
            row["session_event_count"] = 0
            row["session_duration_sec"] = 0
            row["session_is_active"] = 0
            row["session_start_hour"] = 0
            row["session_end_hour"] = 0
            row["session_end_reason"] = ""
        augmented.append(row)
    return augmented


def _safe_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None


def _session_duration_seconds(sess: dict[str, Any]) -> int:
    started = _safe_dt(sess.get("started_at"))
    ended = _safe_dt(sess.get("ended_at"))
    if not started:
        return 0
    end = ended or started
    return max(0, int((end - started).total_seconds()))


def _build_feature_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_map = {
        "background_noise": 0,
        "sequence_behavior": 1,
        "context_exception": 2,
        "manual_override": 3,
    }
    phase_map = {
        "background_noise": 0,
        "sequence_idle": 1,
        "override_response": 2,
        "profile_multi_user": 3,
        "context_transition": 4,
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("user_id", "unknown")].append(row)

    X = []
    y_light = []
    y_fan = []
    y_override = []
    feature_rows = []
    sequences = []
    sequence_labels = []
    sequence_users = []

    time_steps = int(getattr(config, "LSTM_TIME_STEPS", 6))
    max_gap_sec = int(getattr(config, "MAX_SEQUENCE_GAP_SEC", 180))

    for user_id, user_rows in grouped.items():
        user_rows = sorted(user_rows, key=lambda r: r["timestamp"])
        encoded_rows = []
        parsed_ts = [datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S") for r in user_rows]

        prev_temp = None
        prev_hum = None
        prev_light = None
        prev_motion = 0

        for idx, row in enumerate(user_rows):
            temp = row["temperature"]
            hum = row["humidity"]
            light = row["light_level"]
            motion = row["motion_detected"]
            temp_delta = row.get("temp_delta", abs(temp - prev_temp) if prev_temp is not None else 0.0)
            hum_delta = row.get("humidity_delta", abs(hum - prev_hum) if prev_hum is not None else 0.0)
            light_delta = row.get("light_delta", abs(light - prev_light) if prev_light is not None else 0.0)
            motion_streak = prev_motion + 1 if motion else 0
            time_bucket = row["hour"] // 3
            is_weekend = 1 if row["day_of_week"] in (5, 6) else 0
            sequence_phase = row.get("sequence_phase", "")
            phase_encoded = phase_map.get(sequence_phase, 0)

            feature_row = {
                "session_event_count": row.get("session_event_count", 0),
                "session_duration_sec": row.get("session_duration_sec", 0),
                "session_is_active": row.get("session_is_active", 0),
                "session_start_hour": row.get("session_start_hour", 0),
                "session_end_hour": row.get("session_end_hour", 0),
                "session_end_reason": row.get("session_end_reason", ""),
                "hour": row["hour"],
                "minute": row["minute"],
                "day_of_week": row["day_of_week"],
                "temperature": temp,
                "humidity": hum,
                "light_level": light,
                "motion_detected": motion,
                "temp_delta": temp_delta,
                "humidity_delta": hum_delta,
                "light_delta": light_delta,
                "motion_streak": motion_streak,
                "time_bucket": time_bucket,
                "is_weekend": is_weekend,
                "face_count": row["face_count"],
                "scenario_type": row["scenario_type"],
                "scenario_code": scenario_map.get(row["scenario_type"], 0),
                "sequence_phase": sequence_phase,
                "sequence_phase_encoded": phase_encoded,
                "override_flag": row["override_flag"],
                "user_id": row["user_id"],
                "dominant_user_id": row.get("dominant_user_id", row["user_id"]),
                "source": row.get("source", "real"),
                "data_source": row.get("data_source", "real"),
                "decision_mode": row.get("decision_mode", "single_user"),
                "light_state": row["light_state"],
                "fan_state": row["fan_state"],
                "presence_type": row["presence_type"],
                "override_reason": row["override_reason"],
                "confidence": row.get("confidence", 0.0),
            }
            feature_rows.append(feature_row)
            X.append([
                feature_row["hour"],
                feature_row["minute"],
                feature_row["day_of_week"],
                feature_row["temperature"],
                feature_row["humidity"],
                feature_row["light_level"],
                feature_row["motion_detected"],
                feature_row["temp_delta"],
                feature_row["humidity_delta"],
                feature_row["light_delta"],
                feature_row["motion_streak"],
                feature_row["time_bucket"],
                feature_row["is_weekend"],
                feature_row["face_count"],
                feature_row["scenario_code"],
                feature_row["sequence_phase_encoded"],
                feature_row["override_flag"],
                feature_row["session_event_count"],
                feature_row["session_duration_sec"],
                feature_row["session_is_active"],
                feature_row["confidence"],
                0,
            ])
            y_light.append(row["light_state"])
            y_fan.append(row["fan_state"])
            y_override.append(row["override_flag"])
            encoded_rows.append([
                feature_row["hour"], feature_row["minute"], feature_row["day_of_week"],
                feature_row["temperature"], feature_row["humidity"], feature_row["light_level"],
                feature_row["motion_detected"], feature_row["temp_delta"], feature_row["humidity_delta"],
                feature_row["light_delta"], feature_row["motion_streak"], feature_row["time_bucket"],
                feature_row["is_weekend"], feature_row["face_count"],
                feature_row["scenario_code"], feature_row["sequence_phase_encoded"],
                feature_row["override_flag"], feature_row["session_event_count"], feature_row["session_duration_sec"],
                feature_row["session_is_active"],
                feature_row["confidence"],
            ])

            prev_temp = temp
            prev_hum = hum
            prev_light = light
            prev_motion = motion_streak

        for i in range(len(encoded_rows) - time_steps):
            window_ts = parsed_ts[i:i + time_steps + 1]
            if any((window_ts[j + 1] - window_ts[j]).total_seconds() > max_gap_sec for j in range(len(window_ts) - 1)):
                continue
            sequences.append(encoded_rows[i:i + time_steps])
            sequence_labels.append({
                "user_id": user_id,
                "next_light": user_rows[i + time_steps]["light_state"],
                "next_fan": user_rows[i + time_steps]["fan_state"],
                "scenario_type": user_rows[i + time_steps]["scenario_type"],
                "override_flag": user_rows[i + time_steps]["override_flag"],
            })
            sequence_users.append(user_id)

    return {
        "rows": feature_rows,
        "X": X,
        "y_light": y_light,
        "y_fan": y_fan,
        "y_override": y_override,
        "sequences": sequences,
        "sequence_labels": sequence_labels,
        "sequence_users": sequence_users,
        "sequence_time_steps": time_steps,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Smart Home training data from CSV")
    parser.add_argument(
        "--data-source",
        choices=["simulated", "real", "hybrid"],
        default=getattr(config, "TRAIN_DATA_SOURCE", "hybrid"),
        help="Nguồn dữ liệu train sẽ được ghi vào file chuẩn hóa",
    )
    parser.add_argument(
        "--eval-source",
        choices=["simulated", "real", "hybrid"],
        default=getattr(config, "TRAIN_EVAL_SOURCE", "real"),
        help="Nguồn dữ liệu eval ưu tiên cho downstream train",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Chỉ chuẩn hóa dữ liệu, không thay đổi metadata train/eval mặc định",
    )
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"Không tìm thấy file CSV: {CSV_PATH}")
        return 1

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("CSV rỗng hoặc không có header.")
            return 1
        _validate_columns(reader.fieldnames)
        all_rows = list(reader)

    session_csv = Path(getattr(config, "DATA_DIR", BASE_DIR / "data")) / "face_sessions.csv"
    session_rows = _parse_session_rows(session_csv)

    prepared, summary = _prepare_rows(all_rows)
    prepared = _load_session_augmented_rows(prepared, session_rows)
    payload = _build_feature_matrix(prepared)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "summary": asdict(summary),
            "prepared_count": len(prepared),
            "data_source": args.data_source,
            "eval_source": args.eval_source,
            "feature_rows": payload["rows"],
            "session_rows_loaded": len(session_rows),
            "feature_bundle": {
                "X": payload["X"],
                "y_light": payload["y_light"],
                "y_fan": payload["y_fan"],
                "y_override": payload["y_override"],
                "sequences": payload["sequences"],
                "sequence_labels": payload["sequence_labels"],
                "sequence_users": payload["sequence_users"],
                "sequence_time_steps": payload["sequence_time_steps"],
            },
        }, f, ensure_ascii=False, indent=2)

    print("=== Training Data Preparation Summary ===")
    print(f"CSV path          : {CSV_PATH}")
    print(f"Total rows        : {summary.total_rows}")
    print(f"Valid rows        : {summary.valid_rows}")
    print(f"Invalid rows      : {summary.invalid_rows}")
    print(f"Override rows     : {summary.override_count}")
    print(f"Sequence rows     : {summary.sequence_count}")
    print(f"Context rows      : {summary.context_exception_count}")
    print(f"Manual rows       : {summary.manual_override_count}")
    print(f"Background rows   : {summary.background_noise_count}")
    print(f"Saved summary     : {OUT_PATH}")
    print(f"Sequence windows  : {len(payload['sequences'])}")
    print("Feature columns   :", ", ".join(summary.feature_names))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
