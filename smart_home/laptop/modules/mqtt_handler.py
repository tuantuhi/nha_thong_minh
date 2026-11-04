

import json
import time
import math
import logging
import threading
import paho.mqtt.client as mqtt
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.priority_policy import DECISION_MODE_FALLBACK, normalize_decision_mode

logger = logging.getLogger(__name__)


class MQTTHandler:
    def __init__(self, db, on_sensor_data=None):
        """
        db: Bắt buộc truyền object Database đang dùng chung của hệ thống vào (Dependency Injection).
        on_sensor_data: Hàm callback được gọi (trigger) mỗi khi nhận được bản tin sensor mới.
        """
        if db is None:
            raise ValueError("[Kiến trúc hệ thống] NGHIÊM TRỌNG: Bạn phải truyền object Database vào MQTTHandler để tránh lỗi SQLITE_BUSY (BUG-03).")
            
        self.db                = db 
        self.on_sensor_data    = on_sensor_data
        
        self.on_esp32_status   = None 
        
        self.client            = None
        self.is_connected      = False
        self._last_sensor_data = None
        self._lock             = threading.Lock() 
        
        # [FIX BUG-MQTT-01] Dùng Dictionary để lưu lệnh pending theo topic (Coalesce mechanism)
        # Đảm bảo khi mất kết nối, hệ thống chỉ lưu trạng thái mong muốn CÙNG CÙNG, chống tràn RAM.
        self._pending_commands = {}
        
        self._desired_light_state = 0
        self._desired_fan_state   = 0
        self._current_light_state  = 0
        self._current_fan_state    = 0
        
        # =========================================================================
        # Khôi phục trạng thái từ Database an toàn qua class Database dùng chung
        # =========================================================================
        try:
            with self.db.get_connection() as conn:
                row = conn.execute("SELECT light_state, fan_state FROM device_states ORDER BY timestamp DESC LIMIT 1").fetchone()
                if row:
                    self._current_light_state = int(row["light_state"])
                    self._current_fan_state   = int(row["fan_state"])
                    logger.info(f"[MQTT] Đã khôi phục trạng thái lịch sử từ DB: Đèn={self._current_light_state}, Quạt={self._current_fan_state}")
                
        except Exception as e:
            logger.warning(f"[MQTT] Không thể khôi phục trạng thái từ DB (Có thể DB chưa có dữ liệu): {e}")

        self._setup_client()

    def _setup_client(self):
        try:
            from paho.mqtt.enums import CallbackAPIVersion
            self.client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, 
                                      client_id="SmartHome_Laptop_AI_Core",
                                      protocol=mqtt.MQTTv5)
        except ImportError:
            self.client = mqtt.Client(client_id="SmartHome_Laptop_AI_Core",
                                      protocol=mqtt.MQTTv5)

        if config.MQTT_USERNAME:
            self.client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
                                         
        if getattr(config, 'MQTT_TLS', False):
            import ssl
            self.client.tls_set(
                ca_certs=getattr(config, 'MQTT_CA_CERTS', None),
                certfile=getattr(config, 'MQTT_CERTFILE', None),
                keyfile=getattr(config, 'MQTT_KEYFILE', None),
                tls_version=ssl.PROTOCOL_TLS_CLIENT if hasattr(ssl, 'PROTOCOL_TLS_CLIENT') else ssl.PROTOCOL_TLS
            )
            if getattr(config, 'MQTT_TLS_INSECURE', False):
                self.client.tls_insecure_set(True)
            logger.info("[MQTT] 🔒 Mã hóa TLS đã được kích hoạt.")

        # =========================================================================
        # Thiết lập LWT (Last Will Testament) trước khi Connect
        # =========================================================================
        if "server_status" not in getattr(config, 'TOPICS', {}):
            logger.warning("[MQTT] CẢNH BÁO: 'server_status' chưa được khai báo trong config.TOPICS. Hệ thống sẽ dùng topic fallback.")
            
        server_status_topic = config.TOPICS.get("server_status", "home/room1/server/status")
        lwt_payload = json.dumps({"status": "offline"})
        self.client.will_set(topic=server_status_topic, payload=lwt_payload, qos=1, retain=True)

        try:
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        except Exception:
            pass

        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message    = self._on_message
        self.client.enable_logger(logger)

    def _normalize_event_payload(self, payload: dict) -> dict:
        now = datetime.now()

        def _text(value, default=""):
            if value is None:
                return default
            if isinstance(value, str):
                value = value.strip()
                return value if value else default
            return str(value)

        def _lower_text(value, default=""):
            return _text(value, default).lower()

        user_id = _text(payload.get("user_id") or payload.get("dominant_user_id") or "unknown", "unknown")
        confidence = float(payload.get("confidence", 0.0) or 0.0)

        return {
            "timestamp": payload.get("timestamp") or now.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": int(payload.get("hour", now.hour) or now.hour),
            "minute": int(payload.get("minute", now.minute) or now.minute),
            "day_of_week": int(payload.get("day_of_week", now.weekday()) or now.weekday()),
            "temperature": float(payload.get("temp", payload.get("temperature", 0.0)) or 0.0),
            "humidity": float(payload.get("hum", payload.get("humidity", 0.0)) or 0.0),
            "light_level": float(payload.get("light", payload.get("light_level", 0.0)) or 0.0),
            "motion_detected": int(bool(payload.get("motion", payload.get("motion_detected", False)))),
            "user_id": user_id,
            "dominant_user_id": _text(payload.get("dominant_user_id") or user_id, user_id),
            "light_state": int(payload.get("light_state", payload.get("light", 0)) or 0),
            "fan_state": int(payload.get("fan_state", payload.get("fan", 0)) or 0),
            "source": _lower_text(payload.get("source") or "real", "real"),
            "data_source": _lower_text(payload.get("data_source") or "real", "real"),
            "scenario_type": _lower_text(payload.get("scenario_type") or "background_noise", "background_noise"),
            "sequence_phase": _lower_text(payload.get("sequence_phase") or "background_noise", "background_noise"),
            "override_flag": int(payload.get("override_flag", 0) or 0),
            "override_reason": _text(payload.get("override_reason") or "", ""),
            "session_id": _text(payload.get("session_id"), ""),
            "presence_type": _lower_text(payload.get("presence_type") or "unknown", "unknown"),
            "face_count": int(payload.get("face_count", 0) or 0),
            "confidence": confidence,
            "decision_mode": normalize_decision_mode(payload.get("decision_mode") or DECISION_MODE_FALLBACK),
            
        }

    def publish_event_log(self, payload: dict):
        try:
            normalized = self._normalize_event_payload(payload)
            if self.client and self.client.is_connected():
                topic = config.TOPICS.get("event_log", "home/room1/events")
                self.client.publish(topic, json.dumps(normalized, ensure_ascii=False), qos=1)
        except Exception as e:
            logger.debug(f"[MQTT] Không publish được event log: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        is_success = False
        if hasattr(rc, 'value'):
            is_success = (rc.value == 0)
        else:
            is_success = (rc == 0)

        if is_success:
            with self._lock:
                self.is_connected = True
            logger.info(f"[MQTT] Đã thiết lập kết nối tới Broker {config.MQTT_BROKER}")
            
            server_status_topic = config.TOPICS.get("server_status", "home/room1/server/status")
            self.client.publish(server_status_topic, json.dumps({"status": "online"}), qos=1, retain=True)
            
            pending = []
            with self._lock:
                pending = list(self._pending_commands.items())
                self._pending_commands.clear()
            
            if pending:
                logger.info(f"[MQTT] Phục hồi {len(pending)} lệnh pending sau reconnect.")
                for topic, payload in pending:
                    try:
                        res = self.client.publish(topic, payload, qos=1)
                        rc_val = res.rc.value if hasattr(res.rc, 'value') else res.rc
                        if rc_val != mqtt.MQTT_ERR_SUCCESS:
                            with self._lock:
                                self._pending_commands[topic] = payload
                            logger.warning(f"[MQTT] Không flush được lệnh pending tới {topic}, sẽ thử lại ở lần reconnect sau.")
                    except Exception as e:
                        with self._lock:
                            self._pending_commands[topic] = payload
                        logger.warning(f"[MQTT] Flush pending lỗi với topic={topic}: {e}")
            
            if "sensor" in config.TOPICS:
                client.subscribe(config.TOPICS["sensor"], qos=1)
            if "status" in config.TOPICS:
                client.subscribe(config.TOPICS["status"], qos=0)
            if "event_log" in config.TOPICS:
                client.subscribe(config.TOPICS["event_log"], qos=1)
            else:
                client.subscribe("home/room1/events", qos=1)
        else:
            logger.error(f"[MQTT] Kết nối Broker thất bại. Mã lỗi (Return Code): {rc}")

    def _on_disconnect(self, client, userdata, disconnect_flags_or_rc, *args):
        with self._lock:
            self.is_connected = False
            
        if len(args) > 0:
            rc = args[0]
        else:
            rc = disconnect_flags_or_rc

        rc_val = rc.value if hasattr(rc, 'value') else rc
        logger.warning(f"[MQTT] Đứt kết nối (code={rc_val}). Hệ thống thư viện Paho đang tự động thử lại...")
        if hasattr(self, 'on_esp32_status') and callable(self.on_esp32_status):
            try:
                self.on_esp32_status('offline')
            except Exception as e:
                logger.error(f"[MQTT] Lỗi khi gọi callback on_esp32_status (offline): {e}")
        
    def _on_message(self, client, userdata, msg):
        topic = msg.topic

        try:
            payload = msg.payload.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                f"[MQTT] Payload không phải UTF-8 hợp lệ ở topic {topic}; bỏ qua gói tin nhị phân hoặc dữ liệu bị lỗi."
            )
            return

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning(f"[MQTT] Bỏ qua payload lỗi định dạng JSON: {payload}")
            return

        if topic == config.TOPICS.get("sensor"):
            with self._lock:
                fallback_t = self._last_sensor_data["temp"] if self._last_sensor_data else 25.0
                fallback_h = self._last_sensor_data["hum"] if self._last_sensor_data else 60.0
                fallback_l = self._last_sensor_data["light"] if self._last_sensor_data else 0.0

            def _safe_float(value, fallback):
                try:
                    if value is None:
                        return fallback, False
                    parsed = float(value)
                    if not math.isfinite(parsed):
                        return fallback, False
                    return parsed, True
                except Exception:
                    return fallback, False

            # ESP32 có thể gửi theo 2 schema khác nhau:
            # - temp / hum / light
            # - temperature / humidity / light_level
            raw_t = data.get("temp", data.get("temperature"))
            raw_h = data.get("hum", data.get("humidity"))
            raw_l = data.get("light", data.get("light_level"))

            raw_m = data.get("motion", data.get("motion_detected", False))
            if isinstance(raw_m, str):
                motion = (raw_m.strip().lower() in ['true', '1', 'yes', 'on'])
            else:
                motion = bool(raw_m)

            temp, ok_t = _safe_float(raw_t, fallback_t)
            hum, ok_h = _safe_float(raw_h, fallback_h)
            light, ok_l = _safe_float(raw_l, fallback_l)

            if not ok_t or temp < -50 or temp > 100:
                temp = fallback_t
            if not ok_h or hum < 0 or hum > 100:
                hum = fallback_h
            if not ok_l or light < 0:
                light = fallback_l

            with self._lock:
                self._last_sensor_data = {
                    "temp": temp, "hum": hum, "light": light, "motion": motion,
                    "received_at": datetime.now().isoformat()
                }

            if (not ok_t) or (not ok_h) or (temp == 0) or (hum == 0):
                logger.warning(
                    f"[MQTT] Phát hiện Sensor Glitch. Payload={data} | Đã tự động khôi phục: T={temp}, H={hum}, L={light}"
                )
            else:
                logger.debug(f"[MQTT-Stream] T: {temp}°C | H: {hum}% | L: {light}lx | PIR: {motion}")

            if self.on_sensor_data:
                try:
                    self.on_sensor_data(temp, hum, light, motion)
                except Exception as e:
                    logger.error(f"[MQTT] Lỗi ngoại lệ khi gọi on_sensor_data callback: {e}", exc_info=True)

        elif topic == config.TOPICS.get("status"):
            status = data.get("status", "unknown")
            logger.info(f"[MQTT] Trạng thái từ ESP32 Edge Node: {status.upper()}")
            
            if status == "online":
                logger.info("[MQTT] ESP32 vừa khởi động. Khôi phục trạng thái để chống Lệch đồng bộ!")
                
                if hasattr(self, 'on_esp32_status') and callable(self.on_esp32_status):
                    try:
                        self.on_esp32_status('online')
                    except Exception as e:
                        logger.error(f"[MQTT] Lỗi khi gọi callback on_esp32_status: {e}")
            return
        elif topic in {
            config.TOPICS.get("light_state"),
            config.TOPICS.get("fan_state"),
            config.TOPICS.get("light_cmd"),
            config.TOPICS.get("fan_cmd"),
        }:
            self.update_ack_state(topic, data)
            logger.info(f"[MQTT] ACK trạng thái thiết bị: {topic} -> {data}")
            return
        elif topic == config.TOPICS.get("event_log", "home/room1/events"):
            try:
                normalized = self._normalize_event_payload(data)
                row_id = self.db.insert_event_log(normalized)
                logger.info(
                    f"[MQTT] Đã lưu event log vào DB (id={row_id}): "
                    f"user={normalized.get('user_id', 'unknown')}, "
                    f"source={normalized.get('source', 'n/a')}, scenario={normalized.get('scenario_type', 'n/a')}"
                )
            except Exception as e:
                logger.error(f"[MQTT] Lỗi lưu event log vào DB: {e}")
            return

    # ─────────────────────────────────────────────────────
    # API GIAO TIẾP VỚI MAIN.PY VÀ WEB_SERVER.PY
    # ─────────────────────────────────────────────────────
    def connect(self):
        try:
            self.client.connect_async(config.MQTT_BROKER, config.MQTT_PORT, keepalive=60)
            self.client.loop_start()   
            logger.info(f"[MQTT] Đang tiến hành kết nối tới {config.MQTT_BROKER}:{config.MQTT_PORT}...")
        except Exception as e:
            logger.error(f"[MQTT] Lỗi Socket/Mạng khi khởi tạo: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("[MQTT] Đã ngắt kết nối an toàn.")

    def get_last_sensor_data(self) -> dict | None:
        with self._lock:
            return self._last_sensor_data.copy() if self._last_sensor_data else None

    def get_light_state(self) -> int:
        with self._lock: return self._current_light_state

    def get_fan_state(self) -> int:
        with self._lock: return self._current_fan_state

    def _normalize_device_state(self, value) -> int:
        if isinstance(value, bool):
            return 1 if value else 0
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return 1 if int(value) != 0 else 0
        text = str(value).strip().lower()
        if text in {"on", "1", "true", "yes", "enable", "enabled"}:
            return 1
        if text in {"off", "0", "false", "no", "disable", "disabled"}:
            return 0
        return 1 if text else 0

    def update_ack_state(self, topic: str, data: dict):
        with self._lock:
            if topic == config.TOPICS.get("light_state"):
                self._current_light_state = self._normalize_device_state(data.get("state", data.get("value")))
            elif topic == config.TOPICS.get("fan_state"):
                self._current_fan_state = self._normalize_device_state(data.get("state", data.get("value")))

    def get_desired_state(self) -> dict:
        with self._lock:
            return {
                "light": self._desired_light_state,
                "fan": self._desired_fan_state,
            }

    def get_actual_state(self) -> dict:
        with self._lock:
            return {
                "light": self._current_light_state,
                "fan": self._current_fan_state,
            }

    # ─────────────────────────────────────────────────────
    # API ĐIỀU KHIỂN THIẾT BỊ EDGE (ESP32)
    # ─────────────────────────────────────────────────────
    
    def control_light(self, state: bool) -> bool:
        light_cmd_topic = config.TOPICS.get("light_cmd", "home/room1/light/command")
        payload = json.dumps({"state": "ON" if state else "OFF"})
        success = self._publish(light_cmd_topic, payload)
        
        if success:
            with self._lock:
                self._desired_light_state = 1 if state else 0
        return success

    def control_fan(self, state: bool) -> bool:
        fan_cmd_topic = config.TOPICS.get("fan_cmd", "home/room1/fan/command")
        payload = json.dumps({"state": "ON" if state else "OFF"})
        success = self._publish(fan_cmd_topic, payload)
        
        if success:
            with self._lock:
                self._desired_fan_state = 1 if state else 0
        return success

    def _publish(self, topic: str, payload: str) -> bool:
        with self._lock:
            if not self.is_connected:
                logger.warning(f"[MQTT] Tạm ngưng publish tới {topic} do mất mạng. Đã lưu trạng thái mới nhất vào hàng đợi.")
                self._pending_commands[topic] = payload
                return False
                
        try:
            result = self.client.publish(topic, payload, qos=1)
            
            rc_val = result.rc.value if hasattr(result.rc, 'value') else result.rc
            success = (rc_val == mqtt.MQTT_ERR_SUCCESS)
            
            if success:
                logger.debug(f"[MQTT-Out] {topic} ➔ {payload}")
                with self._lock:
                    self._pending_commands.pop(topic, None)
            else:
                logger.error(f"[MQTT-Out] Lỗi gửi gói tin tới {topic}. Error Code: {rc_val}")
                with self._lock:
                    self._pending_commands[topic] = payload
                
            return success
        except Exception as e:
            logger.error(f"[MQTT-Out] Exception TCP/IP khi gửi gói tin tới {topic}: {e}")
            with self._lock:
                self._pending_commands[topic] = payload
            return False
            return False