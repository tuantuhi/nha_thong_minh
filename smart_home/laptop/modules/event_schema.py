from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any


SCHEMA_VERSION = "v2"


@dataclass(slots=True)
class CanonicalEvent:
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
    source: str
    scenario_type: str
    sequence_phase: str
    override_flag: int
    override_reason: str
    session_id: str
    decision_mode: str
    data_source: str
    


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _lower(value: Any, default: str = "") -> str:
    return _text(value, default).lower()


def canonicalize_event(payload: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now()
    timestamp = payload.get("timestamp") or now.strftime("%Y-%m-%d %H:%M:%S")
    user_id = _lower(payload.get("user_id") or payload.get("face_id") or "unknown", "unknown")
    source = _lower(payload.get("source") or "auto", "auto")
    scenario_type = _lower(payload.get("scenario_type") or "background_noise", "background_noise")
    sequence_phase = _lower(payload.get("sequence_phase") or "background_noise", "background_noise")
    presence_type = _lower(payload.get("presence_type") or "unknown", "unknown")
    dominant_user_id = _lower(payload.get("dominant_user_id") or user_id, user_id)
    confidence = float(payload.get("confidence", 0.0) or 0.0)

    event = CanonicalEvent(
        timestamp=timestamp,
        hour=int(payload.get("hour", now.hour) or now.hour),
        minute=int(payload.get("minute", now.minute) or now.minute),
        day_of_week=int(payload.get("day_of_week", now.weekday()) or now.weekday()),
        temperature=float(payload.get("temp", payload.get("temperature", 0.0)) or 0.0),
        humidity=float(payload.get("hum", payload.get("humidity", 0.0)) or 0.0),
        light_level=float(payload.get("light", payload.get("light_level", 0.0)) or 0.0),
        motion_detected=int(bool(payload.get("motion", payload.get("motion_detected", False)))),
        user_id=user_id,
        dominant_user_id=dominant_user_id,
        presence_type=presence_type,
        face_count=int(payload.get("face_count", 0) or 0),
        light_state=int(payload.get("light_state", payload.get("pred_light", 0)) or 0),
        fan_state=int(payload.get("fan_state", payload.get("pred_fan", 0)) or 0),
        confidence=confidence,
        source=source,
        scenario_type=scenario_type,
        sequence_phase=sequence_phase,
        override_flag=int(payload.get("override_flag", 0) or 0),
        override_reason=_text(payload.get("override_reason") or "", ""),
        session_id=_text(payload.get("session_id"), ""),
        decision_mode=_lower(payload.get("decision_mode") or "single_user", "single_user"),
        data_source=_lower(payload.get("data_source") or "real", "real"),
        
    )
    return asdict(event)
