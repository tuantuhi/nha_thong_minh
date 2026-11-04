

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

from .digital_twin import DigitalTwinEngine, SimulationMode
from .digital_twin_adapter import DigitalTwinAdapter, build_twin_adapter, normalize_readout

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    mqtt = None
    MQTT_AVAILABLE = False


DEFAULT_BROKER_HOST = os.getenv("DT_MQTT_HOST", "localhost")
DEFAULT_BROKER_PORT = int(os.getenv("DT_MQTT_PORT", "1883"))
DEFAULT_STATE_TOPIC = os.getenv("DT_MQTT_STATE_TOPIC", "smart_home/twin/state")
DEFAULT_EVENT_TOPIC = os.getenv("DT_MQTT_EVENT_TOPIC", "smart_home/twin/event")
DEFAULT_CLIENT_ID = os.getenv("DT_MQTT_CLIENT_ID", "smart_home_digital_twin")
DEFAULT_PUBLISH_INTERVAL_SEC = float(os.getenv("DT_MQTT_INTERVAL_SEC", "1.0"))


class DigitalTwinMQTTBridge:
    """Read states from the adapter and publish them to MQTT."""

    def __init__(
        self,
        adapter: Optional[DigitalTwinAdapter] = None,
        *,
        broker_host: str = DEFAULT_BROKER_HOST,
        broker_port: int = DEFAULT_BROKER_PORT,
        state_topic: str = DEFAULT_STATE_TOPIC,
        event_topic: str = DEFAULT_EVENT_TOPIC,
        client_id: str = DEFAULT_CLIENT_ID,
        publish_interval_sec: float = DEFAULT_PUBLISH_INTERVAL_SEC,
    ) -> None:
        self.adapter = adapter or build_twin_adapter(
            DigitalTwinEngine(mode=SimulationMode.AUTOPILOT, seed=7, start_time=datetime.now())
        )
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.state_topic = state_topic
        self.event_topic = event_topic
        self.client_id = client_id
        self.publish_interval_sec = max(0.1, float(publish_interval_sec))
        self._client = None
        self._running = False

    @property
    def enabled(self) -> bool:
        return self.adapter.enabled

    def connect(self) -> bool:
        if not MQTT_AVAILABLE:
            logger.error("[TwinMQTT] paho-mqtt is not installed.")
            return False

        try:
            self._client = mqtt.Client(client_id=self.client_id, clean_session=True)
            self._client.connect(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()
            logger.info("[TwinMQTT] Connected to MQTT broker %s:%s", self.broker_host, self.broker_port)
            return True
        except Exception as exc:
            logger.error("[TwinMQTT] Failed to connect MQTT broker: %s", exc)
            self._client = None
            return False

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass
        finally:
            self._client = None

    def publish_once(self, seconds: int = 1) -> bool:
        if self._client is None and not self.connect():
            return False

        readout = self.adapter.step_and_read(seconds=seconds)
        payload = normalize_readout(readout)
        topic = self.event_topic if payload.get("events") else self.state_topic
        return self._publish(topic, payload)

    def run_forever(self) -> None:
        self._running = True
        try:
            while self._running:
                self.publish_once(seconds=int(max(1, round(self.publish_interval_sec))))
                time.sleep(self.publish_interval_sec)
        finally:
            self.close()

    def stop(self) -> None:
        self._running = False

    def _publish(self, topic: str, payload: dict) -> bool:
        if self._client is None:
            return False
        try:
            message = json.dumps(payload, ensure_ascii=False)
            result = self._client.publish(topic, message, qos=0, retain=False)
            if hasattr(result, "rc") and result.rc != 0:
                logger.warning("[TwinMQTT] Publish returned rc=%s", result.rc)
            return True
        except Exception as exc:
            logger.error("[TwinMQTT] Publish failed: %s", exc)
            return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    bridge = DigitalTwinMQTTBridge()
    if not bridge.enabled:
        logger.info("[TwinMQTT] Digital twin is disabled in adapter, but bridge can still publish if attached.")

    if not bridge.connect():
        return

    logger.info("[TwinMQTT] Bridge started. Press Ctrl+C to stop.")
    try:
        bridge.run_forever()
    except KeyboardInterrupt:
        logger.info("[TwinMQTT] Stopping bridge...")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
