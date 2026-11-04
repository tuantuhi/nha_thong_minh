from __future__ import annotations

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class RuleDecision:
    light: int
    fan: int
    confidence: float = 0.25
    source: str = "rule_based"
    user_id: str = "unknown"
    reason: str = ""


class RuleEngine:
    """Rule-based engine tách riêng khỏi ML pipeline.

    Mục tiêu:
    - Xử lý các tình huống an toàn / fallback khi model chưa sẵn sàng
    - Tập trung toàn bộ ngưỡng rule ở một nơi
    - Cho phép AI model chỉ lo inference, không ôm logic điều khiển cứng
    """

    def __init__(self):
        self.dark_threshold = getattr(config, "ENV_THRESHOLDS", {}).get("light_dark", 100)
        self.bright_threshold = getattr(config, "ENV_THRESHOLDS", {}).get("light_bright", 300)
        self.temp_high = getattr(config, "ENV_THRESHOLDS", {}).get("temp_high", 29.0)

    def decide(self, *, temp: float, light: float, motion: bool, user_id: str = "unknown",
               allow_fan_when_hot: bool = True, empty_room_safe: bool = True) -> dict:
        user_id = (user_id or "unknown").strip().lower() or "unknown"

        if user_id == "unknown" and empty_room_safe:
            return self.empty_room_decide(temp=temp, light=light, motion=motion, user_id="global_average", allow_fan_when_hot=allow_fan_when_hot)

        if light < self.dark_threshold and motion:
            return RuleDecision(
                light=1,
                fan=1 if allow_fan_when_hot and temp > self.temp_high else 0,
                confidence=0.35,
                source="rule_dark_motion",
                user_id=user_id,
                reason="dark_motion",
            ).__dict__

        if temp > self.temp_high and motion:
            return RuleDecision(
                light=1 if light < self.bright_threshold else 0,
                fan=1,
                confidence=0.3,
                source="rule_hot_motion",
                user_id=user_id,
                reason="hot_motion",
            ).__dict__

        return RuleDecision(
            light=0 if light > self.bright_threshold else 1,
            fan=0,
            confidence=0.2,
            source="rule_minimal",
            user_id=user_id,
            reason="minimal",
        ).__dict__

    def pir_motion_decide(self, *, temp: float, light: float, user_id: str = "unknown", fan_state: int = 0) -> dict:
        user_id = (user_id or "unknown").strip().lower() or "unknown"
        return RuleDecision(
            light=1,
            fan=int(fan_state),
            confidence=1.0,
            source="pir_motion_rule",
            user_id=user_id,
            reason="pir_motion",
        ).__dict__

    def pir_timeout_decide(self, *, temp: float, light: float, user_id: str = "unknown") -> dict:
        user_id = (user_id or "unknown").strip().lower() or "unknown"
        base = self.empty_room_decide(temp=temp, light=light, motion=False, user_id=user_id, allow_fan_when_hot=False)
        base["light"] = 0
        base["fan"] = 0
        base["confidence"] = 1.0
        base["source"] = "pir_timeout_total_off_rule"
        base["reason"] = "pir_timeout"
        return base

    def empty_room_decide(self, *, temp: float, light: float, motion: bool, user_id: str = "global_average", allow_fan_when_hot: bool = True) -> dict:
        user_id = (user_id or "global_average").strip().lower() or "global_average"
        light_on = 1 if (light < self.dark_threshold or motion) else 0
        fan_on = 1 if allow_fan_when_hot and temp > self.temp_high and motion else 0
        return RuleDecision(
            light=light_on,
            fan=fan_on,
            confidence=0.25,
            source="rule_empty_room_safe",
            user_id=user_id,
            reason="empty_room",
        ).__dict__
