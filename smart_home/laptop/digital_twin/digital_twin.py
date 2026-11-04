

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from random import Random
from typing import Any


class SimulationMode(str, Enum):
    OFF = "off"
    DEMO = "demo"
    MANUAL = "manual"


@dataclass
class ShowcasePreset:
    name: str
    temperature: float
    humidity: float
    light: float
    occupancy: int
    note: str = ""


@dataclass
class TwinSnapshot:
    timestamp: str
    mode: str
    scenario: str
    environment: dict[str, Any]
    occupancy: dict[str, Any]
    control: dict[str, Any]
    devices: dict[str, Any]
    stream: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DigitalTwinEngine:
    PRESETS: dict[str, ShowcasePreset] = {
        "comfortable": ShowcasePreset("comfortable", 26.0, 58.0, 420.0, 0, "Phòng ổn định, thiết bị gần như tắt."),
        "hot": ShowcasePreset("hot", 35.0, 68.0, 780.0, 1, "Kéo nhiệt độ lên cao để thấy quạt/AC bật ngay."),
        "dark": ShowcasePreset("dark", 27.0, 60.0, 0.0, 1, "Hạ ánh sáng xuống 0 lux để đèn phản ứng tức thì."),
        "visitor": ShowcasePreset("visitor", 29.0, 63.0, 180.0, 1, "Mô phỏng có người bước vào phòng."),
    }

    DEFAULT_USERS = ["unknown", "guest", "user_a", "user_b"]

    def __init__(self, mode: SimulationMode = SimulationMode.DEMO, seed: int | None = None, start_time: datetime | None = None) -> None:
        self.mode = mode
        self._rng = Random(seed)
        self._now = start_time or datetime.now()
        self._events: list[dict[str, Any]] = []
        self._scenario = "comfortable"
        self._manual_until: datetime | None = None
        self._manual_target: dict[str, Any] = {}
        self._occupancy = {"present": False, "count": 0, "source": "idle", "users": []}
        self._environment = {"temperature": 26.0, "humidity": 58.0, "light": 420.0, "motion": 0}
        self._devices = {"light": False, "fan": False, "ac": False, "door": False}
        self._showcase_clock = self._now.replace(hour=9, minute=0, second=0, microsecond=0)

    @property
    def presets(self) -> dict[str, dict[str, Any]]:
        return {key: asdict(value) for key, value in self.PRESETS.items()}

    def snapshot(self) -> TwinSnapshot:
        return TwinSnapshot(
            timestamp=self._now.strftime("%Y-%m-%d %H:%M:%S"),
            mode=self.mode.value,
            scenario=self._scenario,
            environment=dict(self._environment),
            occupancy=dict(self._occupancy),
            control={
                "manual_active": self._manual_is_active(),
                "manual_until": self._manual_until.strftime("%Y-%m-%d %H:%M:%S") if self._manual_until else "",
                "manual_target": dict(self._manual_target),
                "showcase_clock": self._showcase_clock.strftime("%Y-%m-%d %H:%M:%S"),
                "available_users": list(self.DEFAULT_USERS),
            },
            devices=dict(self._devices),
            stream=list(self._events),
        )

    def step(self, seconds: int = 1) -> TwinSnapshot:
        if self.mode == SimulationMode.OFF:
            return self.snapshot()
        seconds = max(1, int(seconds))
        self._now += timedelta(seconds=seconds)
        self._decay_manual_override()
        self._apply_demo_dynamics(seconds)
        self._apply_device_logic()
        self._emit("step", {"seconds": seconds, "scenario": self._scenario})
        return self.snapshot()

    def load_preset(self, preset_name: str) -> TwinSnapshot:
        preset = self.PRESETS[preset_name]
        self._scenario = preset.name
        self._environment.update({"temperature": preset.temperature, "humidity": preset.humidity, "light": preset.light})
        self._occupancy.update({"present": bool(preset.occupancy), "count": int(preset.occupancy), "source": f"preset:{preset.name}"})
        self._environment["motion"] = 1 if preset.occupancy else 0
        self._emit("preset_loaded", {"preset": preset.name, "note": preset.note})
        self._apply_device_logic()
        return self.snapshot()

    def set_sliders(self, *, temperature: float | None = None, humidity: float | None = None, light: float | None = None) -> TwinSnapshot:
        if temperature is not None:
            self._environment["temperature"] = float(temperature)
        if humidity is not None:
            self._environment["humidity"] = float(humidity)
        if light is not None:
            self._environment["light"] = float(light)
        self._scenario = "custom"
        self._emit("slider_update", {"temperature": temperature, "humidity": humidity, "light": light})
        self._apply_device_logic()
        return self.snapshot()

    def simulate_person_enter(self, count: int = 1, label: str = "visitor", users: list[str] | None = None) -> TwinSnapshot:
        count = max(1, int(count))
        selected_users = self._normalize_users(users)
        if not selected_users:
            selected_users = [label]
        self._occupancy.update({"present": True, "count": count, "source": label, "users": selected_users})
        self._environment["motion"] = 1
        self._scenario = "occupancy_change"
        self._emit("person_enter", {"count": count, "label": label, "users": selected_users})
        self._apply_device_logic()
        return self.snapshot()

    def set_showcase_datetime(self, when: datetime | str | None = None) -> TwinSnapshot:
        dt = self._parse_datetime(when) if when is not None else self._now
        self._showcase_clock = dt
        self._now = dt
        self._scenario = "datetime_override"
        self._emit("datetime_update", {"timestamp": dt.strftime("%Y-%m-%d %H:%M:%S")})
        self._apply_device_logic()
        return self.snapshot()

    def set_showcase_users(self, users: list[str] | None = None) -> TwinSnapshot:
        selected = self._normalize_users(users)
        self._occupancy["users"] = selected
        self._occupancy["count"] = len(selected)
        self._occupancy["present"] = bool(selected)
        self._occupancy["source"] = "manual_users"
        self._environment["motion"] = 1 if selected else 0
        self._scenario = "user_override"
        self._emit("users_update", {"users": selected})
        self._apply_device_logic()
        return self.snapshot()

    def simulate_person_leave(self) -> TwinSnapshot:
        self._occupancy.update({"present": False, "count": 0, "source": "vacant"})
        self._environment["motion"] = 0
        self._scenario = "vacant"
        self._emit("person_leave", {})
        self._apply_device_logic()
        return self.snapshot()

    def set_manual_override(self, device: str, state: Any, duration_sec: int = 15, reason: str = "demo_override") -> TwinSnapshot:
        self._manual_target = {device: state}
        self._manual_until = self._now + timedelta(seconds=max(1, int(duration_sec)))
        self._devices[device] = state
        self._scenario = "manual_override"
        self._emit("manual_override", {"device": device, "state": state, "duration_sec": duration_sec, "reason": reason})
        return self.snapshot()

    def clear_manual_override(self) -> TwinSnapshot:
        self._manual_until = None
        self._manual_target = {}
        self._emit("manual_override_clear", {})
        self._apply_device_logic()
        return self.snapshot()

    def trigger_showcase(self, preset_name: str) -> TwinSnapshot:
        return self.load_preset(preset_name)

    def _normalize_users(self, users: list[str] | None) -> list[str]:
        if not users:
            return []
        cleaned: list[str] = []
        for user in users:
            value = (user or "").strip()
            if value and value not in cleaned:
                cleaned.append(value)
        return cleaned

    def _parse_datetime(self, when: datetime | str | None) -> datetime:
        if isinstance(when, datetime):
            return when
        if isinstance(when, str):
            normalized = when.replace("T", " ").strip()
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(normalized, fmt)
                except ValueError:
                    continue
        return self._now

    def _manual_is_active(self) -> bool:
        return self._manual_until is not None and self._now <= self._manual_until

    def _decay_manual_override(self) -> None:
        if self._manual_until is not None and self._now > self._manual_until:
            self._manual_until = None
            self._manual_target = {}
            self._emit("manual_override_expired", {})

    def _apply_demo_dynamics(self, seconds: int) -> None:
        drift = min(1.0, seconds / 30.0)
        if not self._occupancy["present"]:
            self._environment["motion"] = 1 if self._rng.random() > 0.96 else 0
        hour = self._showcase_clock.hour + self._showcase_clock.minute / 60.0
        daylight = max(0.0, 1.0 - abs((hour - 12.0) / 12.0))
        self._environment["light"] += ((80.0 + 840.0 * daylight) - self._environment["light"]) * 0.03 * drift
        self._environment["temperature"] += (26.0 - self._environment["temperature"]) * 0.02 * drift
        self._environment["humidity"] += (58.0 - self._environment["humidity"]) * 0.02 * drift
        self._environment["light"] += (420.0 - self._environment["light"]) * 0.02 * drift
        self._environment["temperature"] = round(max(0.0, self._environment["temperature"] + self._rng.uniform(-0.05, 0.05)), 2)
        self._environment["humidity"] = round(max(0.0, min(100.0, self._environment["humidity"] + self._rng.uniform(-0.1, 0.1))), 2)
        self._environment["light"] = round(max(0.0, self._environment["light"] + self._rng.uniform(-1.5, 1.5)), 2)
        self._showcase_clock += timedelta(seconds=seconds)

    def _apply_device_logic(self) -> None:
        if self._manual_is_active():
            for device, state in self._manual_target.items():
                self._devices[device] = state
            return
        if self.mode == SimulationMode.OFF:
            return
        occupied = bool(self._occupancy["present"])
        self._devices["light"] = occupied and self._environment["light"] < 250.0
        self._devices["fan"] = occupied and self._environment["temperature"] >= 28.0
        self._devices["ac"] = occupied and self._environment["temperature"] >= 30.0
        self._devices["door"] = occupied

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self._events.append({"timestamp": self._now.strftime("%Y-%m-%d %H:%M:%S"), "type": event_type, "payload": payload})
        if len(self._events) > 300:
            self._events = self._events[-300:]
