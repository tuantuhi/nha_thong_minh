

from __future__ import annotations

import json
import time
from datetime import datetime

try:
    from .digital_twin import DigitalTwinEngine, SimulationMode
    from .digital_twin_adapter import build_twin_adapter, normalize_readout
except ImportError:
    from digital_twin import DigitalTwinEngine, SimulationMode
    from digital_twin_adapter import build_twin_adapter, normalize_readout


def _pretty_snapshot(snapshot: dict) -> str:
    return json.dumps(
        {
            "timestamp": snapshot.get("timestamp"),
            "enabled": snapshot.get("enabled"),
            "source": snapshot.get("source"),
            "environment": snapshot.get("environment"),
            "user": snapshot.get("user"),
            "override": snapshot.get("override"),
            "devices": snapshot.get("devices"),
            "events": snapshot.get("events", [])[-3:],
        },
        ensure_ascii=False,
        indent=2,
    )


def main() -> None:
    engine = DigitalTwinEngine(mode=SimulationMode.AUTOPILOT, seed=7, start_time=datetime.now())
    adapter = build_twin_adapter(engine)

    print("Digital Twin demo started. Press Ctrl+C to stop.\n")
    engine.simulate_user_enter("owner", "Owner")

    for tick in range(20):
        if tick == 6:
            engine.set_manual_override("light", True, duration_sec=8, reason="demo_manual_light_on")
        if tick == 12:
            engine.simulate_user_leave()

        readout = adapter.step_and_read(seconds=5)
        print(_pretty_snapshot(normalize_readout(readout)))
        print("-" * 80)
        time.sleep(0.2)


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(
            "Run this demo from the package root, for example:\n"
            "  python -m digital_twin.digital_twin_demo\n"
            "or from smart_home/laptop with:\n"
            "  python -m smart_home.laptop.digital_twin.digital_twin_demo\n"
            f"\n\nImport error: {exc}"
        ) from exc
