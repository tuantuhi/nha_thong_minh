import logging
import time
from dataclasses import asdict

from modules.decision_coordinator import DeviceDecision
from modules.event_schema import canonicalize_event

logger = logging.getLogger(__name__)


class DeviceService:
    """Điều khiển thiết bị dùng chung cho CLI và Web."""

    def __init__(self, mqtt_handler=None, state_lock=None, state_ref=None, event_broadcaster=None):
        self.mqtt_handler = mqtt_handler
        self.state_lock = state_lock
        self.state_ref = state_ref
        self.event_broadcaster = event_broadcaster

    def _mqtt_ok(self, ok: bool) -> bool:
        return bool(ok)

    def _publish(self, kind: str, value: bool) -> bool:
        if kind == "light":
            return self._mqtt_ok(self.mqtt_handler.control_light(bool(value)))
        if kind == "fan":
            return self._mqtt_ok(self.mqtt_handler.control_fan(bool(value)))
        return False

    def _current_state(self):
        with self.state_lock:
            return self.state_ref["devices"].get("light", 0), self.state_ref["devices"].get("fan", 0)

    def _set_state(self, light: int | None = None, fan: int | None = None):
        with self.state_lock:
            if light is not None:
                self.state_ref["devices"]["light"] = int(bool(light))
            if fan is not None:
                self.state_ref["devices"]["fan"] = int(bool(fan))

    def _broadcast(self):
        if self.event_broadcaster and self.state_lock and self.state_ref:
            with self.state_lock:
                devices_copy = self.state_ref["devices"].copy()
            self.event_broadcaster("devices", devices_copy)

    def apply_decision(self, decision: DeviceDecision, broadcast_state: bool = True):
        if decision.shadow:
            return {"ok": True, "shadow": True, "decision": asdict(decision)}

        result = {
            "ok": True,
            "decision": asdict(decision),
            "light": {"attempted": False, "ok": True, "error": None},
            "fan": {"attempted": False, "ok": True, "error": None},
        }
        if self.mqtt_handler is None or self.state_ref is None or self.state_lock is None:
            result["ok"] = False
            result["error"] = "DeviceService chưa được khởi tạo đầy đủ"
            return result

        desired_light = int(bool(decision.light))
        desired_fan = int(bool(decision.fan))
        curr_light, curr_fan = self._current_state()

        # Cố gắng gửi riêng cho từng thiết bị để một lỗi không chặn thiết bị còn lại.
        if desired_light != curr_light:
            result["light"]["attempted"] = True
            if self._publish("light", desired_light):
                self._set_state(light=desired_light)
                time.sleep(0.12)
            else:
                result["light"]["ok"] = False
                result["light"]["error"] = "Không gửi được lệnh đèn"
                result["ok"] = False

        if desired_fan != curr_fan:
            result["fan"]["attempted"] = True
            if self._publish("fan", desired_fan):
                self._set_state(fan=desired_fan)
            else:
                result["fan"]["ok"] = False
                result["fan"]["error"] = "Không gửi được lệnh quạt"
                result["ok"] = False

        if broadcast_state:
            self._broadcast()

        if not result["ok"]:
            errors = []
            if not result["light"]["ok"]:
                errors.append(result["light"]["error"])
            if not result["fan"]["ok"]:
                errors.append(result["fan"]["error"])
            result["error"] = "; ".join(errors)

        return result
