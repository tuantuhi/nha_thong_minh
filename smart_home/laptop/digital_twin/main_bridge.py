"""Opt-in bridge for wiring the showcase engine into the app."""

from __future__ import annotations

from typing import Any, Optional

import config

from .digital_twin import DigitalTwinEngine, SimulationMode
from .digital_twin_adapter import DigitalTwinAdapter, build_twin_adapter, normalize_readout


class MainDigitalTwinBridge:
    def __init__(self, engine: Optional[DigitalTwinEngine] = None) -> None:
        self._engine = engine if self._is_enabled() else None
        self._adapter: DigitalTwinAdapter = build_twin_adapter(self._engine)

    def _is_enabled(self) -> bool:
        return bool(getattr(config, "ENABLE_DIGITAL_TWIN", False))

    @property
    def enabled(self) -> bool:
        return self._adapter.enabled

    def read(self) -> dict[str, Any]:
        return normalize_readout(self._adapter.read())

    def step(self, seconds: int = 1) -> dict[str, Any]:
        return normalize_readout(self._adapter.step_and_read(seconds=seconds))

    def apply_preset(self, preset_name: str) -> dict[str, Any]:
        return normalize_readout(self._adapter.apply_preset(preset_name))

    def update_sliders(self, *, temperature: float | None = None, humidity: float | None = None, light: float | None = None) -> dict[str, Any]:
        return normalize_readout(self._adapter.update_sliders(temperature=temperature, humidity=humidity, light=light))

    def person_enter(self, count: int = 1, label: str = "visitor") -> dict[str, Any]:
        return normalize_readout(self._adapter.person_enter(count=count, label=label))

    def person_leave(self) -> dict[str, Any]:
        return normalize_readout(self._adapter.person_leave())

    def attach_engine(self, engine: DigitalTwinEngine) -> None:
        if self._is_enabled():
            self._engine = engine
            self._adapter.attach_engine(engine)

    def detach_engine(self) -> None:
        self._engine = None
        self._adapter.detach_engine()

    def snapshot(self) -> dict[str, Any]:
        return self.read()

    def as_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "adapter": normalize_readout(self._adapter.read())}


def build_main_digital_twin_bridge(engine: Optional[DigitalTwinEngine] = None) -> MainDigitalTwinBridge:
    return MainDigitalTwinBridge(engine=engine)


def create_default_twin_engine() -> DigitalTwinEngine:
    return DigitalTwinEngine(mode=SimulationMode.DEMO)


def main() -> None:
    bridge = build_main_digital_twin_bridge(engine=create_default_twin_engine() if bool(getattr(config, "ENABLE_DIGITAL_TWIN", False)) else None)
    print(bridge.as_dict())


if __name__ == "__main__":
    main()