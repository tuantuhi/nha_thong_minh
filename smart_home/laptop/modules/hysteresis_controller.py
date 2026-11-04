import json
import logging
import threading
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class HysteresisState:
    light: int = 0
    fan: int = 0


class HysteresisController:
    """Bộ điều khiển hysteresis độc lập để chống nhấp nháy thiết bị."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[str, HysteresisState] = {}

    def _get_state(self, user_id: str) -> HysteresisState:
        if user_id not in self._state:
            self._state[user_id] = HysteresisState()
        return self._state[user_id]

    def _profile(self, user_id: str) -> dict:
        profiles = getattr(config, "USER_HYSTERESIS_PROFILES", {}) or {}
        if user_id in profiles:
            return profiles[user_id]
        if user_id in ("global_average", "guest", "unknown"):
            return profiles.get("guest", {}) or {}
        return profiles.get("owner", {}) or {}

    def apply(self, user_id: str, *, temp: float, light: float, motion: bool, desired_light: int, desired_fan: int) -> tuple[int, int, dict]:
        t = config.ENV_THRESHOLDS
        profile = self._profile(user_id)
        fan_on_at = t.get("temp_high", 29.0) + float(profile.get("fan_on_delta", getattr(config, "FAN_HYSTERESIS_ON_OFFSET", 0.8)))
        fan_off_at = t.get("temp_high", 29.0) - float(profile.get("fan_off_delta", getattr(config, "FAN_HYSTERESIS_OFF_OFFSET", 0.6)))
        light_on_at = t.get("light_dark", 100) - float(profile.get("light_on_delta", getattr(config, "LIGHT_HYSTERESIS_ON_OFFSET", 0.0)))
        light_off_at = t.get("light_bright", light_on_at + 20) - float(profile.get("light_off_delta", getattr(config, "LIGHT_HYSTERESIS_OFF_OFFSET", 20.0)))

        with self._lock:
            state = self._get_state(user_id)

            if state.light == 1:
                if motion or light < light_off_at:
                    state.light = 1
                else:
                    state.light = 0
            else:
                if motion and light <= light_on_at:
                    state.light = 1
                elif light <= light_on_at * 0.6:
                    state.light = 1

            if state.fan == 1:
                if temp <= fan_off_at:
                    state.fan = 0
            else:
                if temp >= fan_on_at:
                    state.fan = 1

            return state.light, state.fan, {
                "fan_on_at": round(float(fan_on_at), 2),
                "fan_off_at": round(float(fan_off_at), 2),
                "light_on_at": round(float(light_on_at), 2),
                "light_off_at": round(float(light_off_at), 2),
                "profile": profile,
            }
