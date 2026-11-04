import logging
from datetime import datetime

from modules.decision_coordinator import DecisionCoordinator, DeviceDecision
from modules.event_schema import canonicalize_event
from modules.priority_policy import map_scenario_type

logger = logging.getLogger(__name__)


class ControlPipeline:
    """Điểm hợp nhất cho apply decision + sync state + log."""

    def __init__(self, db, device_service, state_lock, shared_state, broadcaster=None):
        self.db = db
        self.device_service = device_service
        self.state_lock = state_lock
        self.shared_state = shared_state
        self.broadcaster = broadcaster
        self.coordinator = DecisionCoordinator()

    def sync_from_decision(self, decision: DeviceDecision, user_id: str | None = None, source: str | None = None):
        with self.state_lock:
            self.shared_state["devices"]["light"] = int(bool(decision.light))
            self.shared_state["devices"]["fan"] = int(bool(decision.fan))
            self.shared_state.setdefault("system", {})["last_prediction"] = {
                "light": int(bool(decision.light)),
                "fan": int(bool(decision.fan)),
                "confidence": float(getattr(decision, "confidence", 0.0)),
                "source": source or getattr(decision, "source", "unknown"),
                "user_id": user_id or getattr(decision, "user_id", "unknown"),
                "ts": datetime.now().isoformat(),
            }
            devices_copy = self.shared_state["devices"].copy()

        if self.broadcaster:
            try:
                self.broadcaster("devices", devices_copy)
            except Exception:
                logger.debug("[Pipeline] Broadcaster lỗi khi sync devices", exc_info=True)

    def infer_scenario_type(self, source: str = "", override_flag: int = 0, review_reason: str = "",
                            event_type: str = "", presence_type: str = "",
                            decision_mode: str = "", target_user: str = "") -> str:
        return map_scenario_type(
            decision_mode=decision_mode,
            source=source,
            target_user=target_user,
            override_flag=override_flag,
            review_reason=review_reason,
            event_type=event_type,
            presence_type=presence_type,
        )

    def apply_and_log(self, decision: DeviceDecision, env_log_id: int | None = None, user_id: str | None = None,
                      env_source: str | None = None, data_source: str = "real", scenario_type: str = "background_noise",
                      override_flag: int = 0, override_reason: str = "", broadcast: bool = True,
                      sensor_snapshot: dict | None = None, evidence: dict | None = None):
        try:
            result = self.device_service.apply_decision(decision, broadcast_state=False)
        except Exception as e:
            logger.error(f"[Pipeline] apply_decision lỗi: {e}", exc_info=True)
            return {"ok": False, "error": str(e), "decision": getattr(decision, "__dict__", {})}
        if not result.get("ok"):
            return result

        self.sync_from_decision(decision, user_id=user_id, source=env_source)

        if env_log_id is not None:
            resolved_user = user_id or decision.user_id
            self.db.log_event_bundle(
                env_log_id=env_log_id,
                user_id=resolved_user,
                light=decision.light,
                fan=decision.fan,
                source=env_source or decision.source or "auto",
                data_source=data_source,
                scenario_type=scenario_type,
                override_flag=override_flag,
                override_reason=override_reason,
                confidence=float(getattr(decision, "confidence", 0.0)),
                event_type="device",
                actor_type="system",
                trigger_source=env_source or decision.source or "ai",
                session_id=getattr(decision, "session_id", None),
                presence_type=getattr(decision, "presence_type", "unknown"),
                face_count=int(getattr(decision, "face_count", 0) or 0),
                hysteresis_meta=str(getattr(decision, "hysteresis_meta", "")),
                feedback_source=env_source or decision.source or "ai",
                decision_evidence={
                    "sensor_json": sensor_snapshot or {},
                    "evidence_json": evidence or {},
                    "frame_url": (evidence or {}).get("frame_url", ""),
                    "frame_path": (evidence or {}).get("frame_path", ""),
                    "reason": getattr(decision, "priority_reason", "") or env_source or decision.source or "ai",
                    "active_users": list(getattr(decision, "active_users", ()) or []),
                },
            )

        if broadcast and self.broadcaster:
            with self.state_lock:
                self.broadcaster("devices", self.shared_state["devices"].copy())
        return result
