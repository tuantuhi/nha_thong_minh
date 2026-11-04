
import os
import sys
import time
import socket
import logging
import threading
import queue
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from database import Database
from ai_model import SmartHomeAI

from modules.face_recognition_module import FaceRecognitionSystem
from modules.mqtt_handler import MQTTHandler
from modules.decision_coordinator import DecisionCoordinator
from modules.device_service import DeviceService
from modules.control_pipeline import ControlPipeline
from modules.rule_engine import RuleEngine

logger_prefix = "MQTT (Mosquitto)"

# ─── Cấu hình Logging ────────────────────────────────────
os.makedirs("data", exist_ok=True)

console_handler = logging.StreamHandler(sys.stdout)
file_handler    = logging.FileHandler(config.LOG_FILE, encoding="utf-8")

log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
console_handler.setLevel(log_level)
file_handler.setLevel(log_level)

logging.basicConfig(
    level    = log_level,
    format   = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [console_handler, file_handler]
)
logger = logging.getLogger("MainApp")


# ═════════════════════════════════════════════════════════
def is_web_server_running(port=5000):
    """Kiểm tra xem web_server.py có đang chiếm dụng hệ thống không."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


class SmartHomeSystem:
    """Hệ thống nhà thông minh – điều phối toàn bộ (Standalone Mode)."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("  Thiết kế hệ thống nhà thông minh dựa trên thói quen người dùng ")
        logger.info("  [CHẾ ĐỘ CLI / DEBUG ĐỘC LẬP - Phiên bản v4.2 YOLO]")
        logger.info("=" * 60)

        self.db   = Database(config.DATABASE_PATH)
        self.ai   = SmartHomeAI(self.db)
        self.mqtt = MQTTHandler(db=self.db, on_sensor_data=self._on_sensor_data)
        self.face = FaceRecognitionSystem(
            self.db,
            on_user_detected = self._on_user_detected,
            on_user_left     = self._on_user_left,
            on_face_event    = self._on_face_event
        )

        self._current_user          = "unknown"
        self._ai_enabled            = True
        self._manual_mode           = False
        self._last_env_id           = None
        self._lock                  = threading.Lock()
        self._train_data_source     = getattr(config, "TRAIN_DATA_SOURCE", "hybrid")
        self._train_eval_source     = getattr(config, "TRAIN_EVAL_SOURCE", "real")

        self.shared_state = {
            "devices": {
                "light": self.mqtt.get_light_state(),
                "fan": self.mqtt.get_fan_state(),
            }
        }

        self._cmd_queue             = queue.Queue()
        self._sensor_queue          = queue.Queue(maxsize=100)
        self._running               = True

        self._last_ai_run           = 0.0
        self._last_user_change_time = 0.0
        self._pir_override_until    = 0.0
        self._pir_light_timer       = None
        self._retrain_timer         = None
        self._active_learning_timer  = None
        self._cleanup_timer          = None
        self._last_hysteresis_meta  = {
            "candidate_user": None,
            "candidate_count": 0,
            "last_seen": 0.0,
            "stable_user": "unknown",
        }
        self._active_face_sessions  = {}
        self._manual_override_state  = {}

        self._is_training = threading.Event()
        self.coordinator = DecisionCoordinator()
        self.device_service = DeviceService(
            mqtt_handler=self.mqtt,
            state_lock=self._lock,
            state_ref=self.shared_state,
            event_broadcaster=None,
        )
        self.pipeline = ControlPipeline(
            db=self.db,
            device_service=self.device_service,
            state_lock=self._lock,
            shared_state=self.shared_state,
            broadcaster=None,
        )
        self.rule_engine = RuleEngine()

        try:
            shadow_ts = self.db.get_meta('shadow_start_time')
            if shadow_ts:
                self._shadow_start_time = datetime.fromisoformat(shadow_ts)
            else:
                self._shadow_start_time = datetime.now()
                self.db.set_meta('shadow_start_time', self._shadow_start_time.isoformat())

            shadow_mode_val = self.db.get_meta('shadow_mode')
            if shadow_mode_val:
                self._shadow_mode = (shadow_mode_val.lower() == 'true')
            else:
                self._shadow_mode = True
                self.db.set_meta('shadow_mode', 'true')
        except Exception:
            self._shadow_start_time = datetime.now()
            self._shadow_mode       = True

    # ─────────────────────────────────────────────────────
    # Workers (Chống Race Condition & Blocking)
    # ─────────────────────────────────────────────────────
    def _enqueue_command(self, cmd_type, args):
        self._cmd_queue.put((cmd_type, args))

    def _get_stable_detected_user(self, raw_user: str | None) -> str:
        """Debounce/hysteresis cho user từ camera để tránh nhảy trạng thái quá nhanh."""
        candidate = (raw_user or "unknown").strip() or "unknown"
        now = time.time()

        debounce_frames = max(2, int(getattr(config, "FACE_USER_DEBOUNCE_FRAMES", 3)))
        debounce_sec = max(0.5, float(getattr(config, "FACE_USER_DEBOUNCE_SECONDS", 1.5)))
        unknown_grace_sec = max(0.5, float(getattr(config, "FACE_UNKNOWN_GRACE_SECONDS", 2.0)))

        meta = self._last_hysteresis_meta
        stable_user = meta.get("stable_user", "unknown")
        candidate_first_seen = meta.get("candidate_first_seen")

        if candidate in (None, "", "unknown"):
            # Chỉ chuyển về unknown nếu trạng thái xấu kéo dài đủ lâu.
            if stable_user not in (None, "unknown") and (now - meta.get("last_seen", 0.0) < unknown_grace_sec):
                return stable_user
            meta["candidate_user"] = "unknown"
            meta["candidate_count"] = 0
            meta["candidate_first_seen"] = None
            meta["last_seen"] = now
            meta["stable_user"] = "unknown"
            return "unknown"

        if candidate == stable_user:
            meta["candidate_user"] = candidate
            meta["candidate_count"] = debounce_frames
            meta["candidate_first_seen"] = now
            meta["last_seen"] = now
            return candidate

        if meta.get("candidate_user") != candidate:
            meta["candidate_user"] = candidate
            meta["candidate_count"] = 1
            meta["candidate_first_seen"] = now
            meta["last_seen"] = now
            return stable_user

        meta["candidate_count"] = int(meta.get("candidate_count", 0)) + 1
        if candidate_first_seen is None:
            meta["candidate_first_seen"] = now
            candidate_first_seen = now
        meta["last_seen"] = now

        if meta["candidate_count"] >= debounce_frames or (now - candidate_first_seen) >= debounce_sec:
            meta["stable_user"] = candidate
            return candidate

        return stable_user

    def _set_stable_user(self, user_id: str):
        user_id = (user_id or "unknown").strip() or "unknown"
        self._last_hysteresis_meta["stable_user"] = user_id
        self._last_hysteresis_meta["candidate_user"] = user_id
        self._last_hysteresis_meta["candidate_count"] = max(
            int(self._last_hysteresis_meta.get("candidate_count", 0)),
            max(2, int(getattr(config, "FACE_USER_DEBOUNCE_FRAMES", 3)))
        )
        self._last_hysteresis_meta["last_seen"] = time.time()

    def _maybe_update_current_user_from_camera(self) -> str:
        detected_user = self.face.get_current_user()
        stable_user = self._get_stable_detected_user(detected_user)
        with self._lock:
            self._current_user = stable_user
            self._camera_person_present = stable_user not in (None, "unknown")
        return stable_user

    def _apply_and_log_decision(self, decision, env_id=None, source=None):
        """Thực thi quyết định AI bằng cùng một luồng chuẩn."""
        scenario_type = "manual_override" if int(getattr(decision, "override_flag", 0)) else "background_noise"
        override_flag = int(getattr(decision, "override_flag", 0))
        override_reason = getattr(decision, "override_reason", "") or ""
        return self.pipeline.apply_and_log(
            decision,
            env_log_id=env_id,
            user_id=decision.user_id,
            env_source=source or decision.source,
            data_source="real",
            scenario_type=scenario_type,
            override_flag=override_flag,
            override_reason=override_reason,
            broadcast=False,
        )

    def _set_manual_override_lockout(self, user_id: str, reason: str = "manual_override"):
        now = datetime.now()
        cooldown_sec = int(getattr(config, "MANUAL_OVERRIDE_COOLDOWN_SEC", 1800))
        self.ai.manual_override_lockout[self.ai._normalize_user_label(user_id)] = {
            "started_at": now,
            "expires_at": now + timedelta(seconds=cooldown_sec),
            "shadow_until": now + timedelta(seconds=int(getattr(config, "MANUAL_OVERRIDE_SHADOW_SEC", 600))),
            "reason": reason,
        }

    def _command_worker(self):
        while self._running:
            try:
                cmd, args = self._cmd_queue.get(timeout=1)
                if cmd == 'light':
                    self.mqtt.control_light(args[0])
                elif cmd == 'fan':
                    self.mqtt.control_fan(args[0])
                self._cmd_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[CommandWorker] Bỏ qua lệnh lỗi: {e}")

    def _sensor_worker(self):
        while self._running:
            try:
                temp, hum, light, motion = self._sensor_queue.get(timeout=1)
                self._process_sensor_data(temp, hum, light, motion)
                self._sensor_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[SensorWorker] Lỗi xử lý sensor: {e}", exc_info=True)

    # ─────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────
    def _on_sensor_data(self, temp: float, hum: float, light: float, motion: bool = False):
        try:
            self._sensor_queue.put_nowait((temp, hum, light, motion))
        except queue.Full:
            logger.warning("[Main] Sensor queue đầy, bỏ qua 1 frame dữ liệu.")

    def _on_face_event(self, payload: dict):
        try:
            if not payload:
                return
            enriched = dict(payload)
            with self._lock:
                enriched.setdefault("user_id", self._current_user)
                enriched.setdefault("dominant_user_id", self._current_user)
                enriched.setdefault("decision_mode", "single_user")
                enriched.setdefault("presence_type", "human" if self._current_user not in ("unknown", "guest") else "unknown")
                enriched.setdefault("source", "face_recognition")
                enriched.setdefault("data_source", "real")
                enriched.setdefault("scenario_type", enriched.get("scenario_type", self.pipeline.infer_scenario_type(
                    source=enriched.get("source", "face_recognition"),
                    override_flag=int(enriched.get("override_flag", 0) or 0),
                    review_reason=enriched.get("override_reason", enriched.get("decision_reason", "")),
                    event_type=enriched.get("event_type", ""),
                    presence_type=enriched.get("presence_type", "unknown"),
                    decision_mode=enriched.get("decision_mode", "single_user"),
                    target_user=enriched.get("dominant_user_id", enriched.get("user_id", self._current_user)),
                )))
            active_users = enriched.get("active_users", [enriched.get("dominant_user_id", enriched.get("user_id", self._current_user))])
            if isinstance(active_users, str):
                active_users = [u.strip() for u in active_users.split(",") if u.strip()]
            active_users = active_users if isinstance(active_users, list) else [active_users]
            enriched["active_users"] = active_users
            enriched["dominant_user_id"] = enriched.get("dominant_user_id", enriched.get("user_id", self._current_user))
            enriched["priority_reason"] = enriched.get("priority_reason", "face_event")

            session_user = enriched.get("user_id", self._current_user)
            session_id = enriched.get("session_id") or f"sess_{session_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_state = self._active_face_sessions.get(session_user)
            if session_state is None:
                self.db.create_face_session(
                    session_id=session_id,
                    user_id=session_user,
                    dominant_user_id=enriched.get("dominant_user_id", session_user),
                    active_users=active_users,
                    presence_type=enriched.get("presence_type", "unknown"),
                    decision_mode=enriched.get("decision_mode", "single_user"),
                    metadata={"source": enriched.get("source", "face_recognition")},
                )
                self._active_face_sessions[session_user] = session_id
                logger.info(f"[Main] Open face session: {session_id} for {session_user}")
            else:
                session_id = session_state

            enriched["session_id"] = session_id
            self.db.touch_face_session(
                session_id,
                user_id=session_user,
                dominant_user_id=enriched.get("dominant_user_id", session_user),
                active_users=active_users,
                presence_type=enriched.get("presence_type", "unknown"),
                decision_mode=enriched.get("decision_mode", "single_user"),
            )

            if hasattr(self.mqtt, "publish_event_log"):
                self.mqtt.publish_event_log(enriched)
            else:
                self.db.log_event_bundle(
                    env_log_id=None,
                    user_id=enriched.get("user_id", "unknown"),
                    light=int(enriched.get("light_state", 0) or 0),
                    fan=int(enriched.get("fan_state", 0) or 0),
                    source=enriched.get("source", "face_recognition"),
                    data_source=enriched.get("data_source", "real"),
                    scenario_type=enriched.get("scenario_type", "sequence_behavior"),
                    override_flag=int(enriched.get("override_flag", 0) or 0),
                    override_reason=enriched.get("override_reason", ""),
                    confidence=float(enriched.get("confidence", 0.0) or 0.0),
                    session_id=enriched.get("session_id", ""),
                    presence_type=enriched.get("presence_type", "unknown"),
                    face_count=int(enriched.get("face_count", 0) or 0),
                    feedback_source=enriched.get("source", "face_recognition"),
                )
        except Exception as e:
            logger.debug(f"[Main] Bỏ qua face_event vì lỗi: {e}")

    def _process_sensor_data(self, temp: float, hum: float, light: float, motion: bool = False):
        pir_str = "CÓ" if motion else "KHÔNG"
        logger.info(f"[Sensor] 🌡 Nhiệt: {temp}°C | 💧 Ẩm: {hum}% | ☀️ Sáng: {light} lux | 🚶‍♂️ PIR: {pir_str}")

        env_id = self.db.insert_sensor_data(temp, hum, light, motion)

        with self._lock:
            self._last_env_id = env_id
            is_manual = self._manual_mode
            is_ai_enabled = self._ai_enabled
            user_id = self._current_user

        curr_light = self.mqtt.get_light_state()
        curr_fan = self.mqtt.get_fan_state()
        heartbeat_decision = self.coordinator.normalize({"light": curr_light, "fan": curr_fan, "confidence": 1.0, "source": "periodic_heartbeat"}, user_id=user_id)
        self.pipeline.sync_from_decision(
            heartbeat_decision,
            user_id=user_id,
            source="periodic_heartbeat",
        )

        if is_manual or self._shadow_mode:
            return

        # ── [TIN TƯỞNG HOÀN TOÀN VÀO CAMERA YOLO] ────────────────────────────
        user_id = self._maybe_update_current_user_from_camera()

        pir_handled = False
        is_pir_auto = getattr(config, 'PIR_AUTO_CONTROL_LIGHT', True)
        if is_pir_auto and motion and light < config.ENV_THRESHOLDS.get("light_dark", 100):
            self._on_motion_detected(env_id)
            pir_handled = True

        if is_ai_enabled:
            with self._lock:
                now        = time.time()
                should_run = (now - self._last_ai_run >= getattr(config, "INFERENCE_INTERVAL_SEC", 120))
                if should_run:
                    self._last_ai_run = now

            if should_run:
                threading.Thread(
                    target  = self._run_ai_prediction,
                    args    = (env_id, temp, hum, light, motion, user_id, pir_handled),
                    daemon  = True
                ).start()
        else:
            if pir_handled:
                current_fan = self.mqtt.get_fan_state()
                decision = self.coordinator.normalize({"light": 1, "fan": current_fan, "confidence": 1.0, "source": "pir_only"}, user_id=user_id, pir_handled=True)
                self.pipeline.apply_and_log(
                    decision,
                    env_log_id=env_id,
                    user_id=user_id,
                    env_source="pir_only",
                    data_source="real",
                    scenario_type="background_noise",
                    override_flag=0,
                    override_reason="",
                    broadcast=False,
                )

    def _on_user_detected(self, user_id: str):
        with self._lock:
            old_user = self._current_user
            current_user = self.face.get_current_user() if hasattr(self, "face") else old_user
            if (
                user_id == "guest"
                and old_user not in ("unknown", "guest")
                and current_user == old_user
            ):
                logger.debug(f"[Main] Bỏ qua guest tạm thời vì đang giữ user ổn định: {old_user}")
                return
            stable_user = self._get_stable_detected_user(user_id)
            self._current_user          = stable_user
            self._last_user_change_time = time.time()
            user_id = stable_user

        if old_user == user_id:
            return

        if hasattr(self.ai, 'reset_context'):
            self.ai.reset_context(old_user)

        with self._lock:
            is_manual = self._manual_mode
            is_shadow = self._shadow_mode
            env_id = self._last_env_id

        if is_manual or is_shadow:
            current_mode = "manual" if is_manual else "shadow"
            logger.info(f"[Main] User đổi sang {user_id} nhưng đang ở {current_mode} mode – bỏ qua AI can thiệp phần cứng.")
            return

        logger.info(f"[Main] 👤 Phát hiện thay đổi user: {user_id}")
        last = self.mqtt.get_last_sensor_data()

        if last:
            received_at_str = last.get("received_at")
            if received_at_str:
                try:
                    received_at = datetime.fromisoformat(received_at_str)
                    data_age    = (datetime.now() - received_at).total_seconds()
                    if data_age > 300:
                        logger.warning(f"[Main] Dữ liệu cảm biến đã cũ {data_age:.0f}s, bỏ qua AI trigger để an toàn.")
                        return
                except Exception:
                    pass

            now    = datetime.now()
            result = self.ai.predict(
                hour        = now.hour,
                day_of_week = now.weekday(),
                temp        = last["temp"],
                hum         = last["hum"],
                light       = last["light"],
                motion      = last.get("motion", False),
                user_id     = user_id
            )
            decision = self.coordinator.normalize(result, user_id=user_id, shadow=False, pir_handled=False)
            if result.get("needs_review"):
                if hasattr(self.ai, "clear_user_memory"):
                    self.ai.clear_user_memory(user_id)

            def execute_commands():
                with self._lock:
                    if self._shadow_mode or not self.coordinator.should_execute(decision, self._manual_mode):
                        logger.info("[Main] Shadow/manual mode bật ngay sau khi AI predict, hủy lệnh hardware.")
                        return

                self.pipeline.apply_and_log(
                    decision,
                    env_log_id=env_id,
                    user_id=user_id,
                    env_source="auto",
                    data_source="real",
                    scenario_type="background_noise",
                    override_flag=0,
                    override_reason="",
                    broadcast=False,
                )

            threading.Thread(target=execute_commands, daemon=True).start()

    def _on_user_left(self, old_user_id: str):
        """ [YOLO MODE] Xóa bỏ PIR Session Hold, phản ứng ngay lập tức khi trống phòng. """
        with self._lock:
            if self._current_user == old_user_id:
                logger.info(
                    f"[Main] 🚪 Camera/YOLO xác nhận {old_user_id} đã rời phòng. "
                    f"Reset ngữ cảnh về 'unknown' ngay lập tức."
                )
                self._current_user = "unknown"  # Ngắt ngữ cảnh tức thì
                self._last_hysteresis_meta["stable_user"] = "unknown"
                self._last_hysteresis_meta["candidate_user"] = None
                self._last_hysteresis_meta["candidate_count"] = 0
                self._last_hysteresis_meta["last_seen"] = time.time()
                is_manual = self._manual_mode
                is_shadow = self._shadow_mode
                env_id    = self._last_env_id
                session_id = self._active_face_sessions.pop(old_user_id, None)
            else:
                return

        if session_id:
            try:
                self.db.close_face_session(session_id, end_reason="user_left")
                logger.info(f"[Main] Close face session: {session_id}")
            except Exception as e:
                logger.debug(f"[Main] Không đóng được face session {session_id}: {e}")

        if is_manual or is_shadow:
            current_mode = "manual" if is_manual else "shadow"
            logger.info(f"[Main] Đang ở chế độ {current_mode} – bỏ qua AI can thiệp khi trống phòng.")
            return

        last = self.mqtt.get_last_sensor_data()
        if last:
            if env_id is None:
                env_id = self.db.insert_sensor_data(
                    last["temp"], last["hum"], last["light"], last.get("motion", False)
                )
                with self._lock:
                    self._last_env_id = env_id

            # Đưa ngữ cảnh "unknown" vào AI để AI tự động đóng cắt thiết bị khi phòng trống
            threading.Thread(
                target = self._run_ai_prediction,
                args   = (env_id, last["temp"], last["hum"], last["light"],
                          last.get("motion", False), "unknown", False, "auto"),
                daemon = True
            ).start()

    def _on_motion_detected(self, env_id: int):
        logger.info("[Main] 🚶‍♂️ Phát hiện chuyển động (PIR) trong bóng tối")

        with self._lock:
            disable_mins = getattr(config, 'PIR_DISABLE_AI_MINUTES', 3)
            self._pir_override_until = time.time() + disable_mins * 60
            is_shadow                = self._shadow_mode

            if self._pir_light_timer:
                self._pir_light_timer.cancel()
            self._pir_light_timer = threading.Timer(
                getattr(config, 'PIR_TIMEOUT_SECONDS', 60), self._pir_auto_turnoff_light
            )
            self._pir_light_timer.daemon = True
            self._pir_light_timer.start()

        if not is_shadow:
            logger.info("[Main] 💡 Kích hoạt luồng An toàn: Bật đèn tự động")
            sensor_snapshot = self.mqtt.get_last_sensor_data() or {}
            decision_dict = self.rule_engine.pir_motion_decide(
                temp=float(sensor_snapshot.get("temp", 0.0)),
                light=float(sensor_snapshot.get("light", 0.0)),
                user_id=self._current_user,
                fan_state=self.mqtt.get_fan_state(),
            )
            decision = self.coordinator.normalize(decision_dict, user_id=self._current_user, pir_handled=True)
            self.pipeline.apply_and_log(
                decision,
                env_log_id=env_id,
                user_id=self._current_user,
                env_source="pir_motion",
                data_source="real",
                scenario_type="sequence_behavior",
                override_flag=0,
                override_reason="",
                broadcast=False,
            )
        else:
            logger.info("[Main] 🌑 Shadow Mode: PIR phát hiện chuyển động nhưng KHÔNG can thiệp phần cứng.")

    def _pir_auto_turnoff_light(self):
        """Timeout PIR nhưng chỉ tắt khi cả PIR lẫn camera đều không còn người."""
        logger.info("[Main] ⏱️ Hết timeout PIR. Đánh giá trạng thái để tắt thiết bị.")
        with self._lock:
            self._pir_override_until  = 0.0
            self._pir_light_timer     = None
            is_shadow                 = self._shadow_mode
            is_manual                 = self._manual_mode
            current_user              = self._current_user
            camera_person_present     = getattr(self, "_camera_person_present", False)
            env_id                    = self._last_env_id

        if is_shadow or is_manual:
            return

        last = self.mqtt.get_last_sensor_data()
        pir_idle = bool(last) and not last.get("motion", False)

        if pir_idle and not camera_person_present:
            # [ECO MODE] Chỉ tắt khi PIR không thấy chuyển động và YOLO cũng không còn thấy người
            decision_dict = self.rule_engine.pir_timeout_decide(
                temp=last.get("temp", last.get("temperature", 0.0)) if last else 0.0,
                light=last.get("light", last.get("light_level", 0.0)) if last else 0.0,
                user_id=current_user,
            )
            decision = self.coordinator.normalize(decision_dict, user_id=current_user)
            self.pipeline.apply_and_log(
                decision,
                env_log_id=env_id,
                user_id=current_user,
                env_source="pir_timeout_total_off",
                data_source="real",
                scenario_type="background_noise",
                override_flag=0,
                override_reason="",
                broadcast=False,
            )
            logger.info("[Main] 🌙 Đã tắt toàn bộ đèn và quạt do hết timeout PIR và camera xác nhận phòng trống.")
        else:
            logger.info("[Main] ⏸️ Bỏ qua auto-off vì camera vẫn thấy người hoặc PIR chưa đủ điều kiện.")

    def _run_ai_prediction(self, env_id: int, temp: float, hum: float, light: float,
                           motion: bool, user_id: str, pir_handled: bool = False,
                           source_override: str = None):
        now = datetime.now()
        result = self.ai.predict(
            hour=now.hour,
            day_of_week=now.weekday(),
            temp=temp,
            hum=hum,
            light=light,
            motion=motion,
            user_id=user_id,
        )

        with self._lock:
            pir_active = time.time() < self._pir_override_until
            is_shadow = self._shadow_mode
            current_user = self._current_user

        if user_id not in (None, "unknown") and result["light"] == 0 and light < config.ENV_THRESHOLDS.get("light_bright", 300):
            logger.debug("[Main] Camera vẫn thấy người trong môi trường đủ tối - chặn AI tắt đèn để tránh tối đột ngột.")
            result["light"] = 1

        if (pir_active or pir_handled) and result["light"] == 0:
            logger.debug("[Main] AI nhường quyền cho Rule-based (PIR) - Giữ đèn BẬT.")
            result["light"] = 1

        if user_id == "unknown":
            result["source"] = source_override or "auto"

        decision = self.coordinator.normalize(result, user_id=user_id, shadow=is_shadow, pir_handled=pir_handled)
        scenario_type = "manual_override" if int(getattr(decision, "override_flag", 0)) else "background_noise"
        if user_id not in (None, "unknown") and (current_user in (user_id, "guest") or result.get("needs_review")):
            scenario_type = "profile_personalization"
        if result.get("needs_review"):
            scenario_type = "context_exception"

        manual_lock_active = self.ai._is_manual_override_active(user_id)
        if manual_lock_active:
            logger.info(f"[Main] Manual override cooldown đang active cho user={user_id} -> chỉ ghi nhận shadow, không gửi MQTT.")
            self.pipeline.sync_from_decision(decision, user_id=user_id, source="manual_override_cooldown")
            if env_id:
                self.db.log_event_bundle(
                    env_log_id=env_id,
                    user_id=user_id,
                    light=decision.light,
                    fan=decision.fan,
                    source="manual_override",
                    data_source="real",
                    scenario_type="manual_override",
                    override_flag=1,
                    override_reason="cooldown_lockout",
                    confidence=float(getattr(decision, "confidence", 0.0)),
                    session_id=getattr(decision, "session_id", None),
                    presence_type=getattr(decision, "presence_type", "unknown"),
                    face_count=int(getattr(decision, "face_count", 0) or 0),
                    feedback_source="manual_override_cooldown",
                )
            return

        if is_shadow:
            self.pipeline.sync_from_decision(decision, user_id=user_id, source="shadow_pir" if pir_handled else "shadow")
            if env_id:
                self.db.log_event_bundle(
                    env_log_id=env_id,
                    user_id=user_id,
                    light=decision.light,
                    fan=decision.fan,
                    source="shadow_pir" if pir_handled else "shadow",
                    data_source="simulated",
                    scenario_type="background_noise",
                    override_flag=0,
                    override_reason="shadow_mode",
                    confidence=float(getattr(decision, "confidence", 0.0)),
                    session_id=getattr(decision, "session_id", None),
                    presence_type=getattr(decision, "presence_type", "unknown"),
                    face_count=int(getattr(decision, "face_count", 0) or 0),
                    feedback_source="shadow_mode",
                )
            return

        source_log = source_override if source_override else "auto"
        self.pipeline.apply_and_log(
            decision,
            env_log_id=env_id,
            user_id=user_id,
            env_source=source_log,
            data_source="real",
            scenario_type=scenario_type,
            override_flag=0,
            override_reason="",
            broadcast=False,
        )


    def _safe_train(self, data_source: str | None = None, eval_source: str | None = None, manage_flag: bool = True):
        if manage_flag and self._is_training.is_set():
            return {"success": False, "message": "Tiến trình Train đang chạy, vui lòng chờ..."}

        if manage_flag:
            self._is_training.set()
        try:
            kwargs = {
                "data_source": data_source or self._train_data_source,
                "eval_source": eval_source or self._train_eval_source,
            }
            return self.ai.train(**kwargs)
        finally:
            if manage_flag:
                self._is_training.clear()

    def _choose_train_eval_mode(self):
        options = {
            "1": ("simulated", "real", "Train simulated, eval real"),
            "2": ("hybrid", "real", "Train hybrid, eval real"),
            "3": ("real", "real", "Train real, eval real"),
            "4": ("simulated", "simulated", "Train simulated, eval simulated"),
            "5": ("hybrid", "hybrid", "Train hybrid, eval hybrid"),
        }
        print("\n  === Chọn mode Train/Eval ===")
        print("  [1] Train simulated / Eval real")
        print("  [2] Train hybrid / Eval real")
        print("  [3] Train real / Eval real")
        print("  [4] Train simulated / Eval simulated")
        print("  [5] Train hybrid / Eval hybrid")
        print("  [0] Giữ nguyên mode hiện tại")
        choice = input("  Chọn (0-5): ").strip()
        if choice == "0":
            print(f"  ✅ Giữ nguyên: train={self._train_data_source}, eval={self._train_eval_source}")
            return
        selected = options.get(choice)
        if not selected:
            print("  ❌ Lựa chọn không hợp lệ!")
            return
        self._train_data_source, self._train_eval_source, label = selected
        config.TRAIN_DATA_SOURCE = self._train_data_source
        config.TRAIN_EVAL_SOURCE = self._train_eval_source
        print(f"  ✅ Đã chọn mode: {label}")
        confirm = input("  Train ngay bây giờ? (y/N): ").strip().lower()
        if confirm == "y":
            print("  ⏳ Đang khởi chạy Train AI theo mode đã chọn...")
            result = self._safe_train(self._train_data_source, self._train_eval_source)
            if result and result.get("success"):
                print(f"  ✅ Train xong! Đèn: {result['accuracy_light']}%, Quạt: {result['accuracy_fan']}%")
            elif result:
                print(f"  ⚠️  {result['message']}")

    def _should_run_retrain(self) -> bool:
        now = datetime.now()
        retrain_every = max(900, int(getattr(config, "RETRAIN_INTERVAL", 86400)))
        batch_min = max(50, int(getattr(config, "RETRAIN_BATCH_MIN", 200)))
        forced = bool(getattr(config, "FORCE_RETRAIN_ON_START", False))
        last_retrain = getattr(self, "_last_retrain_time", None)
        if forced and last_retrain is None:
            return True
        if last_retrain is None:
            return True
        return (now - last_retrain).total_seconds() >= retrain_every and self.db.count_training_data() >= batch_min

    def _run_retrain_cycle(self, mode: str = "scheduled"):
        if self._is_training.is_set():
            logger.info(f"[Main] Bỏ qua retrain {mode} vì một phiên train khác đang chạy.")
            return None
        if mode == "scheduled" and not self._should_run_retrain():
            logger.info("[Main] Chưa tới ngưỡng retrain định kỳ, bỏ qua.")
            return None

        try:
            count = self.db.count_training_data()
            logger.info(f"[Main] Tự động train lại AI ({count} mẫu dữ liệu)...")
            result = self._safe_train(manage_flag=False)
            if result and result.get("success"):
                self._last_retrain_time = datetime.now()
                self._last_retrain_result = result
                logger.info(f"[Main] Train xong! Đèn: {result['accuracy_light']}%, Quạt: {result['accuracy_fan']}%")
                self._check_shadow_mode_exit(count)
                deleted_rows = self.db.prune_old_logs(days_to_keep=30)
                if deleted_rows > 0:
                    logger.info(f"[Auto] Đã dọn dẹp {deleted_rows} dòng log cũ (>30 ngày) để tối ưu Database.")
                if hasattr(self.ai, "warmup_buffer"):
                    try:
                        recent_logs = self.db.get_recent_logs(n=getattr(config, "LSTM_TIME_STEPS", 8) * 4)
                        self.ai.warmup_buffer(recent_logs)
                    except Exception as e:
                        logger.debug(f"[Main] Không warm-up lại sau retrain: {e}")
            return result
        except Exception as e:
            logger.error(f"[Retrain] Lỗi nghiêm trọng khi train: {e}", exc_info=True)
            return None
        finally:
            self._is_training.clear()

    def _schedule_retrain(self):
        def retrain_job():
            try:
                self._run_retrain_cycle(mode="scheduled")
            finally:
                self._schedule_retrain()

        self._retrain_timer = threading.Timer(config.RETRAIN_INTERVAL, retrain_job)
        self._retrain_timer.daemon = True
        self._retrain_timer.start()

    def _schedule_active_learning(self):
        def active_learning_job():
            try:
                feedback_window = max(50, int(getattr(config, "ACTIVE_LEARNING_FEEDBACK_WINDOW", 150)))
                recent_logs = self.db.get_recent_logs(n=max(getattr(config, "LSTM_TIME_STEPS", 5) * 12, feedback_window))
                if hasattr(self.ai, "warmup_buffer"):
                    self.ai.warmup_buffer(recent_logs)
                    logger.info(f"[Main] Đã làm mới buffer ngữ cảnh từ {len(recent_logs)} log gần đây.")

                review_rows = []
                manual_rows = []
                for r in recent_logs:
                    source = str(r.get("source", "")).strip().lower()
                    override_flag = int(r.get("override_flag", 0) or 0)
                    review_flag = int(r.get("needs_review", 0) or 0)
                    overridden = int(r.get("overridden", 0) or 0)
                    if source in ("manual_override", "manual", "active_learning_feedback", "needs_review") or override_flag == 1 or overridden == 1:
                        manual_rows.append(r)
                    if review_flag == 1 or source in ("auto", "needs_review"):
                        review_rows.append(r)

                retrain_threshold = max(20, int(getattr(config, "ACTIVE_LEARNING_RETRAIN_THRESHOLD", 30)))
                review_threshold = max(10, int(getattr(config, "ACTIVE_LEARNING_REVIEW_THRESHOLD", 15)))
                review_hit_count = sum(1 for r in recent_logs if int(r.get("needs_review", 0) or 0) == 1 or int(r.get("overridden", 0) or 0) == 1 or str(r.get("source", "")).strip().lower() == "needs_review")

                if len(manual_rows) >= retrain_threshold or review_hit_count >= review_threshold:
                    logger.info(
                        f"[Main] Active learning trigger: manual={len(manual_rows)}, review={review_hit_count} -> retrain nhanh."
                    )
                    self._run_retrain_cycle(mode="active_learning")
            except Exception as e:
                logger.debug(f"[Main] Active learning maintenance lỗi: {e}")
            finally:
                self._schedule_active_learning()

        interval = max(300, int(getattr(config, "ACTIVE_LEARNING_INTERVAL", 1800)))
        self._active_learning_timer = threading.Timer(interval, active_learning_job)
        self._active_learning_timer.daemon = True
        self._active_learning_timer.start()

    def _schedule_daily_cleanup(self):
        def cleanup_job():
            try:
                deleted_rows = self.db.prune_old_logs(days_to_keep=60)
                if deleted_rows > 0:
                    logger.info(f"[Auto] Đã xóa {deleted_rows} dòng log cũ hơn 60 ngày.")
            except Exception as e:
                logger.debug(f"[Main] Daily cleanup lỗi: {e}")
            finally:
                self._schedule_daily_cleanup()

        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        now = datetime.now()
        next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run + timedelta(days=1)
        interval = max(1, int((next_run - now).total_seconds()))
        self._cleanup_timer = threading.Timer(interval, cleanup_job)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _check_shadow_mode_exit(self, count: int):
        elapsed       = (datetime.now() - self._shadow_start_time).total_seconds()
        required_rows = getattr(config, 'MIN_TRAINING_ROWS', 500)
        min_elapsed   = getattr(config, 'MIN_SHADOW_ELAPSED_SECONDS', 259200)

        with self._lock:
            if self._shadow_mode:
                logger.info(
                    f"[Shadow Mode] Tiến độ: {count}/{required_rows} mẫu dữ liệu | "
                    f"Thời gian: {elapsed/3600:.1f}/{min_elapsed/3600:.1f} giờ"
                )
                if count >= required_rows and elapsed > min_elapsed:
                    self._shadow_mode  = False
                    should_notify      = True
                else:
                    should_notify = False
            else:
                should_notify = False

        if should_notify:
            self.db.set_meta('shadow_mode', 'false')
            logger.info("[Main] 🌑 TỰ ĐỘNG THOÁT SHADOW MODE. AI đã đủ thông minh và bắt đầu điều khiển!")

    # ─────────────────────────────────────────────────────
    # Khởi động & Dừng
    # ─────────────────────────────────────────────────────
    def start(self):
        logger.info("[Main] Đang khởi động hệ thống...")

        threading.Thread(target=self._command_worker, daemon=True).start()
        threading.Thread(target=self._sensor_worker,  daemon=True).start()

        self.mqtt.connect()

        deadline = time.time() + 30.0
        while not getattr(self.mqtt, 'is_connected', False) and time.time() < deadline:
            time.sleep(1.0)

        if not getattr(self.mqtt, 'is_connected', False):
            logger.warning("[Main] MQTT chưa kết nối sau 30 giây. Vẫn tiếp tục khởi động hệ thống.")

        logger.info("[Main] Đang nạp lịch sử ngữ cảnh (Warm-up LSTM)...")
        warmup_n    = getattr(config, "LSTM_TIME_STEPS", 8) * 10
        recent_logs = self.db.get_recent_logs(n=warmup_n * 3)

        if len(recent_logs) >= getattr(config, "LSTM_TIME_STEPS", 8):
            self.ai.warmup_buffer(recent_logs)
            logger.info("[Main] ✅ Nạp Warm-up LSTM hoàn tất.")
        else:
            logger.info("[Main] Chưa đủ dữ liệu Training để Warm-up, bỏ qua.")

        self.face.start()
        self._schedule_retrain()
        self._schedule_active_learning()
        self._schedule_daily_cleanup()
        try:
            deleted_rows = self.db.prune_old_logs(days_to_keep=60)
            if deleted_rows > 0:
                logger.info(f"[Auto] Đã xóa {deleted_rows} dòng log cũ hơn 60 ngày ngay khi khởi động.")
        except Exception as e:
            logger.debug(f"[Main] Không thể chạy cleanup khi khởi động: {e}")
        logger.info("[Main] ✅ Hệ thống đã sẵn sàng!\n")

        self._cli_thread = threading.Thread(target=self._idle_mode, daemon=True)
        self._cli_thread.start()

        while self._running:
            time.sleep(0.5)

    def stop(self):
        logger.info("[Main] Đang tắt hệ thống...")
        self._running = False

        deadline = time.time() + 3.0
        while not self._cmd_queue.empty() and time.time() < deadline:
            time.sleep(0.1)

        if self._retrain_timer: self._retrain_timer.cancel()
        if self._active_learning_timer: self._active_learning_timer.cancel()
        if self._cleanup_timer: self._cleanup_timer.cancel()
        if self._pir_light_timer: self._pir_light_timer.cancel()
        self.face.stop()
        self.mqtt.disconnect()
        logger.info("[Main] Đã tắt. Tạm biệt!")

    # ─────────────────────────────────────────────────────
    # CLI Interface (Dual Mode Logic)
    # ─────────────────────────────────────────────────────
    def _idle_mode(self):
        while self._running:
            console_handler.setLevel(logging.INFO)
            print("\n" + "═" * 60)
            print(" 🟢 HỆ THỐNG ĐANG HOẠT ĐỘNG - LOG ĐƯỢC CẬP NHẬT LIÊN TỤC")
            print(" ⌨️  BẤM phím [ENTER] bất cứ lúc nào để mở MENU điều khiển")
            print("═" * 60 + "\n")

            try:
                input()
            except EOFError:
                break

            console_handler.setLevel(logging.WARNING)
            action = self._show_menu()

            if action == "stop":
                self.stop()
                break
            elif action is None:
                continue

    def _show_menu(self):
        while self._running:
            print("\n" + "═" * 55)

            with self._lock:
                user      = self._current_user
                is_manual = self._manual_mode
                is_shadow = self._shadow_mode
                elapsed   = (datetime.now() - self._shadow_start_time).total_seconds()

            mode       = "THỦ CÔNG" if is_manual else "TỰ ĐỘNG (AI)"
            count      = self.db.count_training_data()
            last       = self.mqtt.get_last_sensor_data()
            sensor_str = f"{last['temp']}°C, {last['hum']}%, {last['light']} lux" if last else "Chưa có dữ liệu"

            shadow_status = (
                f"SHADOW MODE (Dữ liệu: {count} | Giờ: {elapsed/3600:.1f}h)"
                if is_shadow else "AI MODE (Đang điều khiển thật)"
            )

            print(f"  🏠 SMART HOME │ Chế độ: {mode}")
            print(f"  ☁️  Kết nối  : {logger_prefix} {'✅' if getattr(self.mqtt, 'is_connected', True) else '❌'}")
            print(f"  👤 Người dùng : {user}")
            print(f"  🌡  Cảm biến  : {sensor_str}")
            print(f"  📊 Mẫu huấn luyện: {count}/{getattr(config, 'MIN_TRAINING_ROWS', 500)}")
            print(f"  🌑 Trạng thái AI: {shadow_status}")
            print("═" * 55)
            print("  [1] Bật/Tắt chế độ Tự động AI")
            print("  [2] Điều khiển thủ công (Đèn / Quạt)")
            print("  [3] Ghi nhãn thủ công (tạo training data)")
            print("  [4] Train AI ngay bây giờ")
            print("  [5] Chọn mode Train/Eval")
            print("  [6] Đăng ký khuôn mặt người dùng mới")
            print("  [7] Xem log gần đây")
            print("  [8] Xem danh sách người dùng")
            print("  [9] Bật/Tắt Shadow Mode")
            print("  [0] Tắt toàn bộ hệ thống")
            print("─" * 55)

            choice = input("  Chọn (0-9): ").strip()

            if choice == "1":
                with self._lock:
                    self._manual_mode = not self._manual_mode
                    is_manual         = self._manual_mode
                    if is_manual and self._pir_light_timer:
                        self._pir_light_timer.cancel()
                        self._pir_light_timer    = None
                        self._pir_override_until = 0.0

                mode_str = "THỦ CÔNG" if is_manual else "TỰ ĐỘNG (AI)"
                print(f"  ✅ Đã chuyển sang chế độ: {mode_str}")

            elif choice == "2":
                self._manual_control()
            elif choice == "3":
                self._label_current_state()
            elif choice == "4":
                print("  ⏳ Đang khởi chạy Train AI...")
                result = self._safe_train(getattr(config, "TRAIN_DATA_SOURCE", None), getattr(config, "TRAIN_EVAL_SOURCE", "real"))
                if result and result.get("success"):
                    print(f"  ✅ Train xong! Độ chính xác Đèn: {result['accuracy_light']}%, Quạt: {result['accuracy_fan']}%")
                    count = self.db.count_training_data()
                    self._check_shadow_mode_exit(count)
                elif result:
                    print(f"  ⚠️  {result['message']}")
            elif choice == "5":
                self._choose_train_eval_mode()
            elif choice == "6":
                self._register_new_user()
            elif choice == "7":
                self._show_recent_logs()
            elif choice == "8":
                self._show_users()
            elif choice == "9":
                with self._lock:
                    self._shadow_mode = not self._shadow_mode
                    is_shadow         = self._shadow_mode
                self.db.set_meta('shadow_mode', 'true' if is_shadow else 'false')
                shadow_status = "SHADOW MODE" if is_shadow else "AI MODE"
                print(f"  ✅ Đã chuyển sang: {shadow_status}")
            elif choice == "0":
                return "stop"
            elif choice == "10":
                return "continue"
            elif choice == "0":
                return "stop"
            else:
                print("  ❌ Lựa chọn không hợp lệ!")

    # [CHUẨN HÓA DỮ LIỆU HUẤN LUYỆN] Đóng gói Snapshot cho Manual Control CLI với Label chuẩn
    def _create_cli_snapshot_log(self, user_id, source_label, target_light=None, target_fan=None):
        last_sensor = self.mqtt.get_last_sensor_data()

        if last_sensor:
            env_id = self.db.insert_sensor_data(
                last_sensor["temp"], last_sensor["hum"],
                last_sensor["light"], last_sensor.get("motion", False)
            )
        else:
            with self._lock:
                env_id = self._last_env_id

        if env_id:
            actual_light = target_light if target_light is not None else self.mqtt.get_light_state()
            actual_fan = target_fan if target_fan is not None else self.mqtt.get_fan_state()
            payload = {
                "user_id": user_id,
                "dominant_user_id": user_id,
                "light_state": actual_light,
                "fan_state": actual_fan,
                "source": source_label,
                "data_source": "real",
                "scenario_type": "manual_override",
                "sequence_phase": "override_response",
                "override_flag": 1,
                "override_reason": source_label,
                "session_id": "",
                "presence_type": "human",
                "face_count": 1,
                "confidence": 1.0,
                
            }
            self.db.log_event_bundle(
                env_log_id=env_id,
                user_id=user_id,
                light=actual_light,
                fan=actual_fan,
                source=source_label,
                data_source="real",
                scenario_type="manual_override",
                override_flag=1,
                override_reason=source_label,
                confidence=1.0,
                session_id="",
                presence_type="human",
                face_count=1,
                feedback_source=source_label,
            )
            logger.info(f"[Manual CLI] Snapshot lưu DB: Đèn={actual_light}, Quạt={actual_fan}, Source={source_label}")

    def _manual_control(self):
        print("\n  === Điều khiển thủ công ===")
        print("  [1] Đèn BẬT  [2] Đèn TẮT")
        print("  [3] Quạt BẬT [4] Quạt TẮT")
        c = input("  Chọn: ").strip()
        target_light, target_fan = None, None

        with self._lock:
            current_user = self._current_user

        if   c == "1": self._enqueue_command('light', (True,));  target_light = 1; print("  💡 Đèn BẬT")
        elif c == "2": self._enqueue_command('light', (False,)); target_light = 0; print("  💡 Đèn TẮT")
        elif c == "3": self._enqueue_command('fan',   (True,));  target_fan   = 1; print("  🌀 Quạt BẬT")
        elif c == "4": self._enqueue_command('fan',   (False,)); target_fan   = 0; print("  🌀 Quạt TẮT")
        else:
            print("  ❌ Lựa chọn không hợp lệ!")

        deadline = time.time() + 5.0
        while not self._cmd_queue.empty() and time.time() < deadline:
            time.sleep(0.1)
        if not self._cmd_queue.empty():
            print("  ⚠️ Cảnh báo: Lệnh chưa được gửi hết hoặc hàng đợi MQTT đang kẹt!")

        if target_light is not None or target_fan is not None:
            self._create_cli_snapshot_log(current_user, "manual_override", target_light, target_fan)

    def _label_current_state(self):
        last = self.mqtt.get_last_sensor_data()
        if not last:
            return print("  ❌ Chưa nhận được dữ liệu cảm biến nào!")

        with self._lock:
            current_user = self._current_user

        print(f"\n  === Ghi nhãn dữ liệu ===")
        print(f"  Môi trường: {last['temp']}°C, {last['hum']}%, {last['light']} lux\n  Người dùng: {current_user}")
        try:
            light = int(input("  Trạng thái Đèn (0=TẮT / 1=BẬT): "))
            fan   = int(input("  Trạng thái Quạt (0=TẮT / 1=BẬT): "))
            if light not in (0, 1) or fan not in (0, 1):
                raise ValueError

            # Snapshot thủ công
            env_id = self.db.insert_sensor_data(
                last["temp"], last["hum"], last["light"],
                motion=last.get("motion", False)
            )
            self.db.log_event_bundle(
                env_log_id=env_id,
                user_id=current_user,
                light=light,
                fan=fan,
                source="manual",
                data_source="real",
                scenario_type="manual_override",
                override_flag=1,
                override_reason="manual_label",
                confidence=1.0,
                session_id="",
                presence_type="human",
                face_count=1,
                feedback_source="manual",
            )
            print("  ✅ Đã lưu nhãn thành công!")
        except ValueError:
            print("  ❌ Giá trị không hợp lệ (chỉ nhập 0 hoặc 1)")

    def _register_new_user(self):
        print("\n  === Đăng ký người dùng mới ===")
        user_id      = input("  Nhập user ID (vd: user_a): ").strip().lower()
        display_name = input("  Nhập tên hiển thị: ").strip()
        if not user_id or not display_name:
            return print("  ❌ Vui lòng nhập đầy đủ thông tin")

        print("  ⏳ Đang trích xuất khung hình (không làm gián đoạn luồng Camera)...")

        try:
            success = self.face.register_user(user_id, display_name)
            if success:
                print(f"  ✅ Đã đăng ký thành công: {display_name}")
            else:
                print("  ❌ Đăng ký thất bại. Kiểm tra camera hoặc ánh sáng.")
        except Exception as e:
            logger.error(f"[Main] Lỗi ngoại lệ khi đăng ký user: {e}")
            print(f"  ❌ Đăng ký thất bại: {e}")

    def _show_recent_logs(self):
        logs = self.db.get_recent_logs(10)
        print(
            f"\n  {'─'*65}\n"
            f"  {'Thời gian':<20} {'Nhiệt':<8} {'Ẩm':<6} {'Sáng':<8} {'User':<12} {'Đèn':<6} {'Quạt'}\n"
            f"  {'─'*65}"
        )
        for r in logs:
            ts   = str(r.get("timestamp", ""))[:19]
            temp = f"{r.get('temperature', 0):.1f}°C"
            hum  = f"{r.get('humidity', 0):.0f}%"
            lux  = f"{r.get('light_level', 0):.0f} lx"
            user = str(r.get("user_id", "-"))[:10]
            ls   = "BẬT" if r.get("light_state") else "TẮT"
            fs   = "BẬT" if r.get("fan_state")   else "TẮT"
            print(f"  {ts:<20} {temp:<8} {hum:<6} {lux:<8} {user:<12} {ls:<6} {fs}")

    def _show_users(self):
        users = self.db.get_all_users()
        print(
            f"\n  {'─'*55}\n"
            f"  {'User ID':<15} {'Tên':<20} {'Lần cuối':<20} {'Lượt'}\n"
            f"  {'─'*55}"
        )
        for u in users:
            print(
                f"  {u['user_id']:<15} {u['display_name']:<20} "
                f"{str(u.get('last_seen', '-'))[:16]:<20} {u['visit_count']}"
            )


# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Home CLI / Debug runner")
    parser.add_argument(
        "--train-data-source",
        choices=["simulated", "real", "hybrid"],
        default=getattr(config, "TRAIN_DATA_SOURCE", "hybrid"),
        help="Nguồn dữ liệu dùng để train AI",
    )
    parser.add_argument(
        "--train-eval-source",
        choices=["simulated", "real", "hybrid"],
        default=getattr(config, "TRAIN_EVAL_SOURCE", "real"),
        help="Nguồn dữ liệu dùng để đánh giá sau train",
    )
    parser.add_argument(
        "--skip-auto-train",
        action="store_true",
        help="Không tự train khi khởi động hệ thống",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    config.TRAIN_DATA_SOURCE = args.train_data_source
    config.TRAIN_EVAL_SOURCE = args.train_eval_source
    config.SKIP_AUTO_TRAIN = args.skip_auto_train

    try:
        import fcntl
        LOCK_FILE = (
            "/tmp/smarthome.lock"
            if sys.platform != "win32"
            else os.path.join(os.environ.get("TEMP", "data"), "smarthome.lock")
        )
        lock_fd = open(LOCK_FILE, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            print("\n" + "═" * 60)
            print(" ❌ CẢNH BÁO XUNG ĐỘT HỆ THỐNG (Lỗi FLOW-01) ❌")
            print(" Phát hiện một instance khác đang chạy (main.py hoặc web_server.py)!")
            print(" Không thể chạy đồng thời để tránh tranh chấp Camera, MQTT và Database.")
            print(" Vui lòng tắt hệ thống đang chạy trước khi sử dụng giao diện CLI này.")
            print("═" * 60 + "\n")
            sys.exit(1)
    except ImportError:
        pass  # Bỏ qua hỗ trợ khóa tiến trình trên các hệ điều hành thuần Windows

    system = SmartHomeSystem()
    try:
        if not getattr(config, "SKIP_AUTO_TRAIN", False):
            try:
                system._safe_train(config.TRAIN_DATA_SOURCE, config.TRAIN_EVAL_SOURCE)
            except Exception as e:
                logger.error(f"[Main] Auto-train thất bại: {e}")
        system.start()
    except KeyboardInterrupt:
        logger.info("\n  [Ctrl+C] Nhận tín hiệu ngắt từ hệ thống. Đang thoát...")
        system.stop()
    except Exception as e:
        logger.error(f"Lỗi khởi động: {e}")