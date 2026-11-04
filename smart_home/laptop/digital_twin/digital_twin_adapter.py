

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from .digital_twin import DigitalTwinEngine, SimulationMode, TwinSnapshot


@dataclass
class DigitalTwinReadout:
    enabled: bool
    source: str
    timestamp: str
    scenario: str
    environment: dict[str, Any]
    occupancy: dict[str, Any]
    control: dict[str, Any]
    devices: dict[str, Any]
    stream: list[dict[str, Any]]
    presets: dict[str, dict[str, Any]]


class DigitalTwinAdapter:
    def __init__(self, engine: Optional[DigitalTwinEngine] = None) -> None:
        self._engine = engine

    @property
    def enabled(self) -> bool:
        return self._engine is not None and self._engine.mode != SimulationMode.OFF

    def has_engine(self) -> bool:
        return self._engine is not None

    def read(self) -> DigitalTwinReadout:
        if self._engine is None:
            return self._disabled_readout()
        return self._from_snapshot(self._engine.snapshot())

    def step_and_read(self, seconds: int = 1) -> DigitalTwinReadout:
        if self._engine is None:
            return self._disabled_readout()
        return self._from_snapshot(self._engine.step(seconds=seconds))

    def apply_preset(self, preset_name: str) -> DigitalTwinReadout:
        if self._engine is None:
            return self._disabled_readout()
        return self._from_snapshot(self._engine.load_preset(preset_name))

    def update_sliders(self, *, temperature: float | None = None, humidity: float | None = None, light: float | None = None) -> DigitalTwinReadout:
        if self._engine is None:
            return self._disabled_readout()
        return self._from_snapshot(self._engine.set_sliders(temperature=temperature, humidity=humidity, light=light))

    def person_enter(self, count: int = 1, label: str = "visitor", users: list[str] | None = None) -> DigitalTwinReadout:
        if self._engine is None:
            return self._disabled_readout()
        return self._from_snapshot(self._engine.simulate_person_enter(count=count, label=label, users=users))

    def set_showcase_datetime(self, when: Any = None) -> DigitalTwinReadout:
        if self._engine is None:
            return self._disabled_readout()
        return self._from_snapshot(self._engine.set_showcase_datetime(when))

    def set_showcase_users(self, users: list[str] | None = None) -> DigitalTwinReadout:
        if self._engine is None:
            return self._disabled_readout()
        return self._from_snapshot(self._engine.set_showcase_users(users))

    def person_leave(self) -> DigitalTwinReadout:
        if self._engine is None:
            return self._disabled_readout()
        return self._from_snapshot(self._engine.simulate_person_leave())

    def attach_engine(self, engine: DigitalTwinEngine) -> None:
        self._engine = engine

    def detach_engine(self) -> None:
        self._engine = None

    def _from_snapshot(self, snapshot: TwinSnapshot) -> DigitalTwinReadout:
        return DigitalTwinReadout(
            enabled=True,
            source="digital_twin",
            timestamp=snapshot.timestamp,
            scenario=snapshot.scenario,
            environment=dict(snapshot.environment),
            occupancy=dict(snapshot.occupancy),
            control=dict(snapshot.control),
            devices=dict(snapshot.devices),
            stream=list(snapshot.stream),
            presets=self._engine.presets if self._engine is not None else {},
        )

    def _disabled_readout(self) -> DigitalTwinReadout:
        return DigitalTwinReadout(
            enabled=False,
            source="disabled",
            timestamp="",
            scenario="off",
            environment={},
            occupancy={},
            control={},
            devices={},
            stream=[],
            presets={},
        )


def build_twin_adapter(engine: Optional[DigitalTwinEngine] = None) -> DigitalTwinAdapter:
    return DigitalTwinAdapter(engine=engine)


def normalize_readout(readout: DigitalTwinReadout) -> dict[str, Any]:
    return asdict(readout)
