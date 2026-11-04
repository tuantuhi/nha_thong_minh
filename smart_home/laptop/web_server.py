
import os, sys, json, time, threading, queue, logging, cv2, io, csv, secrets, hashlib, hmac, sqlite3
import numpy as np
import math
from datetime import datetime, timedelta
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, Response, stream_with_context, send_from_directory, make_response, render_template_string
from flask_cors import CORS

import config
from database import Database
from ai_model import SmartHomeAI

from modules.face_recognition_module import FaceRecognitionSystem
from modules.mqtt_handler import MQTTHandler
from modules.decision_coordinator import DecisionCoordinator
from modules.device_service import DeviceService
from modules.control_pipeline import ControlPipeline
from modules.rule_engine import RuleEngine
from modules.event_schema import canonicalize_event
from modules.priority_policy import (
    DECISION_MODE_FALLBACK,
    DECISION_MODE_MANUAL_OVERRIDE,
    DECISION_MODE_MULTI_USER,
    DECISION_MODE_SINGLE_USER,
    map_scenario_type,
    normalize_decision_mode,
)
from digital_twin import DigitalTwinEngine, SimulationMode, build_twin_adapter, normalize_readout

logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("WebServer")

app = Flask(__name__, template_folder="templates", static_folder="static")

# [FIX BUG-WEB-03] Bổ sung Security Headers chống XSS
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:;"
    return response

# Cấu hình CORS
CORS_ORIGINS_ENV = os.getenv('CORS_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000')
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_ENV.split(',') if origin.strip()]
if not CORS_ORIGINS:
    CORS_ORIGINS = ['http://localhost:5000']
CORS(app, supports_credentials=True, origins=CORS_ORIGINS)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
db  = Database(config.DATABASE_PATH)
ai  = SmartHomeAI(db)
mqtt_handler = None
face_system = None  

digital_twin_engine = None
if getattr(config, "ENABLE_DIGITAL_TWIN", True):
    digital_twin_engine = DigitalTwinEngine(mode=SimulationMode.DEMO)

digital_twin_adapter = build_twin_adapter(digital_twin_engine)

_state_lock  = threading.Lock()
_sse_lock    = threading.Lock()
_sse_clients: list[queue.Queue] = []

_ai_time_lock = threading.Lock()
_last_ai_run: float = 0.0
_manual_override_until: float = 0.0
_MANUAL_COOLDOWN_SEC = 120

_user_change_until: float = 0.0
_USER_CHANGE_COOLDOWN_SEC = 30

_pir_override_until: float = 0.0
_pir_timeout_timer: threading.Timer | None = None
_pir_timeout_token: int = 0

_esp32_watchdog_timer: threading.Timer | None = None
_ESP32_TIMEOUT_SEC = 150

_active_face_sessions: dict[str, str] = {}

_sm = db.get_meta("shadow_mode")
_initial_shadow = (_sm or "true").lower() == "true"

shared_state = {
    "sensor":  {"temp": 28.5, "hum": 72.0, "light": 120.0, "motion": False, "ts": ""},
    "devices": {"light": 0, "fan": 0,
                 "light_actual": 0, "fan_actual": 0},
    "system":  {
        "shadow_mode": _initial_shadow, "manual_mode": False, "ai_enabled": True,
        "mqtt_connected": False, "esp32_online": False, "esp32_last_seen": None,
        "current_user": "unknown",
        "training_count": 0, "ai_accuracy_light": None, "ai_accuracy_fan": None,
        "last_prediction": None,
        "safe_mode": False, "safe_mode_reason": "", "anomalies": []
    },
    "digital_twin": {
        "enabled": False,
        "source": "disabled",
        "timestamp": "",
        "scenario": "off",
        "environment": {},
        "occupancy": {},
        "control": {},
        "devices": {},
        "stream": [],
        "presets": {},
    }
}

def _dashboard_metric_bundle(*, predicted_light=None, predicted_fan=None, actual_light=None, actual_fan=None, executed_light=None, executed_fan=None, **extra) -> dict:
    payload = dict(extra)
    if predicted_light is not None:
        payload["predicted_light"] = predicted_light
    if predicted_fan is not None:
        payload["predicted_fan"] = predicted_fan
    if actual_light is not None:
        payload["actual_light"] = actual_light
    if actual_fan is not None:
        payload["actual_fan"] = actual_fan
    if executed_light is not None:
        payload["executed_light"] = executed_light
    if executed_fan is not None:
        payload["executed_fan"] = executed_fan
    return payload


def _current_data_source() -> str:
    return "simulated" if digital_twin_adapter.has_engine() and bool(shared_state.get("digital_twin", {}).get("enabled", False)) else "real"


def _broadcast_event(event_type: str, data: dict):
    msg  = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    dead = []
    with _sse_lock:
        clients_copy = list(_sse_clients)
        for q in clients_copy:
            try: q.put_nowait(msg)
            except queue.Full: dead.append(q)
        for q in dead:
            if q in _sse_clients: _sse_clients.remove(q)


def _sync_digital_twin_state(readout):
    normalized = normalize_readout(readout)
    with _state_lock:
        shared_state["digital_twin"] = {**normalized, "enabled": True}
        env = normalized.get("environment", {})
        occ = normalized.get("occupancy", {})
        dev = normalized.get("devices", {})
        if env:
            shared_state["sensor"]["temp"] = env.get("temperature", shared_state["sensor"].get("temp"))
            shared_state["sensor"]["hum"] = env.get("humidity", shared_state["sensor"].get("hum"))
            shared_state["sensor"]["light"] = env.get("light", shared_state["sensor"].get("light"))
            shared_state["sensor"]["motion"] = bool(env.get("motion", shared_state["sensor"].get("motion")))
        if dev:
            shared_state["devices"]["light"] = int(bool(dev.get("light", shared_state["devices"].get("light", 0))))
            shared_state["devices"]["fan"] = int(bool(dev.get("fan", shared_state["devices"].get("fan", 0))))
        if occ:
            selected_users = occ.get("users") or []
            if isinstance(selected_users, str):
                selected_users = [u.strip() for u in selected_users.split(",") if u.strip()]
            selected_users = [u for u in selected_users if u and u not in {"unknown", "guest"}]
            current_user = shared_state["system"].get("current_user", "unknown")
            if selected_users:
                if current_user in selected_users:
                    next_user = current_user
                else:
                    next_user = selected_users[0]
                shared_state["system"]["current_user"] = next_user
            elif not occ.get("present"):
                shared_state["system"]["current_user"] = "unknown"
    _broadcast_event("digital_twin", normalized)


def _log_digital_twin_change(action: str, readout, *, source: str = "digital_twin", user_id: str = "showcase"):
    normalized = normalize_readout(readout)
    env = normalized.get("environment", {}) or {}
    dev = normalized.get("devices", {}) or {}
    occ = normalized.get("occupancy", {}) or {}
    control = normalized.get("control", {}) or {}

    light_state = int(bool(dev.get("light", 0)))
    fan_state = int(bool(dev.get("fan", 0)))
    confidence = float(control.get("confidence", 1.0) or 1.0)
    env_log_id = db.insert_sensor_data(
        float(env.get("temperature", shared_state["sensor"].get("temp", 0.0)) or 0.0),
        float(env.get("humidity", shared_state["sensor"].get("hum", 0.0)) or 0.0),
        float(env.get("light", shared_state["sensor"].get("light", 0.0)) or 0.0),
        bool(env.get("motion", shared_state["sensor"].get("motion", False))),
    )
    db.log_event_bundle(
        env_log_id=env_log_id,
        user_id=user_id,
        light=light_state,
        fan=fan_state,
        source=source,
        data_source="simulated",
        scenario_type="digital_twin_showcase",
        override_flag=1,
        override_reason=action,
        confidence=confidence,
        feedback_source="digital_twin",
        prediction_id=None,
        actual_light=light_state,
        actual_fan=fan_state,
        predicted_light=light_state,
        predicted_fan=fan_state,
        event_type="digital_twin_change",
        actor_type="user",
        trigger_source=action,
        hysteresis_meta=json.dumps({
            "action": action,
            "occupancy": occ,
            "control": control,
        }, ensure_ascii=False),
    )

coordinator = DecisionCoordinator()
rule_engine = RuleEngine()
device_service = DeviceService(mqtt_handler=None, state_lock=_state_lock, state_ref=shared_state, event_broadcaster=None)
pipeline = ControlPipeline(db=db, device_service=device_service, state_lock=_state_lock, shared_state=shared_state, broadcaster=_broadcast_event)

# ══════════════════════════════════════════════════════════
# ① AUTHENTICATION SYSTEM & SECURITY
# ══════════════════════════════════════════════════════════

_WEB_USERNAME = os.getenv("WEB_USERNAME", "")
_WEB_PASSWORD = os.getenv("WEB_PASSWORD", "")

_SECRET_KEY = os.getenv("SECRET_KEY")
if not _SECRET_KEY:
    _SECRET_KEY = db.get_meta("SECRET_KEY")
    if not _SECRET_KEY:
        _SECRET_KEY = secrets.token_hex(32)
        db.set_meta("SECRET_KEY", _SECRET_KEY)

_active_tokens: dict = {}
_token_lock = threading.Lock()
TOKEN_TTL_HOURS = 1

_login_attempts = {}
_login_lock = threading.Lock()

def _hash_token(token: str) -> str:
    return hashlib.sha256(f"{_SECRET_KEY}{token}".encode()).hexdigest()

def issue_token(username: str) -> str:
    raw = secrets.token_hex(32)
    h   = _hash_token(raw)
    expires = time.time() + TOKEN_TTL_HOURS * 3600
    with _token_lock:
        expired = [k for k, v in _active_tokens.items() if v["expires"] < time.time()]
        for k in expired: del _active_tokens[k]
        _active_tokens[h] = {"user": username, "expires": expires}
    try:
        db.set_meta(f"session:{h}", json.dumps({"user": username, "expires": expires}))
    except Exception: pass
    return raw

def revoke_token(raw: str):
    h = _hash_token(raw)
    with _token_lock:
        _active_tokens.pop(h, None)
    try: db.set_meta(f"session:{h}", "")
    except Exception: pass

def _validate_token(raw: str) -> dict | None:
    if not raw: return None
    h = _hash_token(raw)
    with _token_lock:
        entry = _active_tokens.get(h)
    if not entry:
        try:
            saved = db.get_meta(f"session:{h}")
            if saved:
                entry = json.loads(saved)
                with _token_lock: _active_tokens[h] = entry
        except Exception: pass
    if not entry: return None
    if entry["expires"] < time.time():
        with _token_lock: _active_tokens.pop(h, None)
        try: db.set_meta(f"session:{h}", "")
        except Exception: pass
        return None
    return entry

def _get_token_from_request() -> str:
    token = request.cookies.get("auth_token")
    if token: return token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "): return auth[7:]
    return request.args.get("token", "")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = _get_token_from_request()
        payload = _validate_token(token)
        if payload is None:
            return jsonify({"ok": False, "error": "Unauthorized", "code": 401}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    ip = request.remote_addr
    now = time.time()
    with _login_lock:
        attempts = _login_attempts.get(ip, [])
        attempts = [t for t in attempts if now - t < 60]
        if len(attempts) >= 5:
            return jsonify({"ok": False, "error": "Quá nhiều yêu cầu sai. Vui lòng thử lại sau 1 phút."}), 429
        _login_attempts[ip] = attempts

    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")
    
    if hmac.compare_digest(username, _WEB_USERNAME) and hmac.compare_digest(password, _WEB_PASSWORD):
        token = issue_token(username)
        logger.info(f"[Auth] Đăng nhập thành công: {username} từ IP: {ip}")
        with _login_lock:
            _login_attempts.pop(ip, None)
            
        # Gọi dọn dẹp các session rác trong DB để tối ưu dung lượng
        threading.Thread(target=db.cleanup_expired_sessions, daemon=True).start()
            
        resp = make_response(jsonify({"ok": True, "token": token, "user": username, "expires_in": TOKEN_TTL_HOURS * 3600}))
        cookie_secure = config.USE_HTTPS if hasattr(config, 'USE_HTTPS') else False
        max_age = TOKEN_TTL_HOURS * 3600
        resp.set_cookie("auth_token", token, httponly=True, samesite="Strict", secure=cookie_secure, max_age=max_age)
        resp.set_cookie("stream_token", token, httponly=True, samesite="Strict", secure=cookie_secure, max_age=max_age)
        return resp
    
    logger.warning(f"[Auth] Đăng nhập sai: user={username} từ IP={ip}")
    with _login_lock:
        _login_attempts[ip].append(now)
    return jsonify({"ok": False, "error": "Sai tên đăng nhập hoặc mật khẩu"}), 401

@app.route("/api/auth/refresh", methods=["POST"])
@require_auth
def auth_refresh():
    token = _get_token_from_request()
    payload = _validate_token(token)
    if payload:
        username = payload["user"]
        revoke_token(token)
        new_token = issue_token(username)
        resp = make_response(jsonify({"ok": True, "token": new_token, "user": username, "expires_in": TOKEN_TTL_HOURS * 3600}))
        cookie_secure = config.USE_HTTPS if hasattr(config, 'USE_HTTPS') else False
        max_age = TOKEN_TTL_HOURS * 3600
        resp.set_cookie("auth_token", new_token, httponly=True, samesite="Strict", secure=cookie_secure, max_age=max_age)
        resp.set_cookie("stream_token", new_token, httponly=True, samesite="Strict", secure=cookie_secure, max_age=max_age)
        return resp
    return jsonify({"ok": False}), 401

@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    token = _get_token_from_request()
    revoke_token(token)
    resp = make_response(jsonify({"ok": True}))
    resp.set_cookie("auth_token", "", expires=0)
    resp.set_cookie("stream_token", "", expires=0)
    return resp

@app.route("/api/auth/verify")
def auth_verify():
    token   = _get_token_from_request()
    payload = _validate_token(token)
    if payload:
        return jsonify({"ok": True, "user": payload["user"],
                        "expires_at": datetime.fromtimestamp(payload["expires"]).isoformat()})
    return jsonify({"ok": False}), 401

# ══════════════════════════════════════════════════════════
# ② SHARED STATE & THREAD LOCKS
# ══════════════════════════════════════════════════════════

def _esp32_mark_offline():
    with _state_lock:
        shared_state["system"]["esp32_online"] = False
    _broadcast_event("node_status", {"esp32_online": False, "reason": "timeout"})
    logger.warning("[LWT] ESP32 không phản hồi – đánh dấu Offline")

def _esp32_heartbeat():
    global _esp32_watchdog_timer
    with _state_lock:
        shared_state["system"]["esp32_online"]    = True
        shared_state["system"]["esp32_last_seen"] = datetime.now().isoformat()
    _broadcast_event("node_status", {"esp32_online": True})
    if _esp32_watchdog_timer:
        _esp32_watchdog_timer.cancel()
    _esp32_watchdog_timer = threading.Timer(_ESP32_TIMEOUT_SEC, _esp32_mark_offline)
    _esp32_watchdog_timer.daemon = True
    _esp32_watchdog_timer.start()

_training_count_cache = 0
_training_count_ts = 0.0

def _get_training_count() -> int:
    global _training_count_cache, _training_count_ts
    now = time.time()
    if now - _training_count_ts > 60:
        _training_count_cache = db.count_training_data()
        _training_count_ts = now
    return _training_count_cache

def _force_update_training_count() -> int:
    global _training_count_cache, _training_count_ts
    _training_count_cache = db.count_training_data()
    _training_count_ts = time.time()
    return _training_count_cache

def check_hardware_availability():
    """Hàm phụ trợ chặn luồng nếu phần cứng mất kết nối."""
    with _state_lock:
        esp_online = shared_state["system"].get("esp32_online", False)
        mqtt_conn  = shared_state["system"].get("mqtt_connected", False)
    return esp_online and mqtt_conn

# ══════════════════════════════════════════════════════════
# ③ AI TRIGGER CALLBACKS (FACE RECOGNITION INTEGRATION)
# ══════════════════════════════════════════════════════════

def _on_user_detected_callback(user_id: str):
    global _user_change_until
    with _state_lock:
        old_user = shared_state["system"]["current_user"]
        if old_user == user_id:
            return
        shared_state["system"]["current_user"] = user_id
        is_manual = shared_state["system"]["manual_mode"]
        is_shadow = shared_state["system"]["shadow_mode"]
        s = shared_state["sensor"].copy()

    with _ai_time_lock:
        _user_change_until = time.time() + _USER_CHANGE_COOLDOWN_SEC

    if hasattr(ai, 'reset_context'):
        ai.reset_context(old_user)

    if is_manual or is_shadow:
        logger.info(f"[Web] User đổi sang {user_id} nhưng đang ở manual/shadow mode – bỏ qua AI can thiệp.")
        return

    logger.info(f"[Web] 👤 Phát hiện thay đổi user: {user_id} -> Kích hoạt AI dự đoán ngay lập tức!")
    now = datetime.now()

    result = ai.predict(
        hour=now.hour, day_of_week=now.weekday(),
        temp=s["temp"], hum=s["hum"], light=s["light"],
        motion=s.get("motion", False), user_id=user_id
    )

    def execute_user_change():
        decision = coordinator.normalize(result, user_id=user_id)
        env_id = db.insert_sensor_data(s["temp"], s["hum"], s["light"], s.get("motion", False))
        scenario_type = _infer_web_scenario(
            decision_mode=getattr(decision, "decision_mode", DECISION_MODE_SINGLE_USER),
            source="auto",
            override_flag=int(getattr(decision, "override_flag", 0)),
            review_reason=getattr(decision, "review_reason", ""),
            presence_type="human" if user_id not in (None, "", "unknown", "guest") else "unknown",
            target_user=user_id,
        )
        apply_result = _apply_and_log_device_decision(
            decision,
            env_source="auto",
            user_id=user_id,
            env_log_id=env_id,
            broadcast=False,
        )
        if not apply_result.get("ok"):
            logger.warning(f"[Web] Không áp dụng được quyết định user-change: {apply_result.get('error')}")
            return

        last_prediction = _dashboard_metric_bundle(
            predicted_light=decision.light,
            predicted_fan=decision.fan,
            executed_light=decision.light,
            executed_fan=decision.fan,
            confidence=decision.confidence,
            source='auto',
            ts=datetime.now().isoformat(),
            user_id=user_id,
            auto=True,
            shadow=False,
            scenario_type=scenario_type,
            decision_mode=getattr(decision, 'decision_mode', 'single_user'),
            presence_type='human' if user_id not in (None, '', 'unknown', 'guest') else 'unknown',
            dominant_user_id=user_id,
            data_source='real',
            override_flag=int(getattr(decision, 'override_flag', 0)),
            override_reason=getattr(decision, 'review_reason', ''),
            session_id='',
        )
        with _state_lock:
            shared_state["system"]["last_prediction"] = last_prediction

        _broadcast_event('prediction', last_prediction)

    threading.Thread(target=execute_user_change, daemon=True).start()

def _on_user_left_callback(old_user_id: str):
    is_manual = False
    is_shadow = False 
    is_ai = True
    s = {}
    trigger_ai = False

    try:
        with _state_lock:
            if shared_state["system"]["current_user"] == old_user_id:
                logger.info(f"[Web] 🚪 Camera/YOLO xác nhận {old_user_id} đã rời phòng. Reset ngữ cảnh về 'unknown' ngay lập tức.")
                shared_state["system"]["current_user"] = "unknown"
                is_manual = shared_state["system"].get("manual_mode", False)
                is_shadow = shared_state["system"].get("shadow_mode", False) 
                is_ai = shared_state["system"].get("ai_enabled", True)
                s = shared_state["sensor"].copy()
                trigger_ai = True
            else:
                return
    except Exception as e:
        logger.error(f"[Web] Lỗi đọc biến shared_state khi user left: {e}")

    if trigger_ai and is_ai and not is_shadow and not is_manual:
        now = datetime.now()
        # [YOLO MODE] Đưa ngữ cảnh "unknown" (trống phòng) vào AI
        result = ai.predict(
            hour=now.hour, day_of_week=now.weekday(),
            temp=s["temp"], hum=s["hum"], light=s["light"], 
            motion=s.get("motion", False), user_id="unknown"
        )

        def execute_user_left():
            decision = coordinator.normalize(result, user_id='unknown')
            env_id = db.insert_sensor_data(s["temp"], s["hum"], s["light"], s.get("motion", False))
            scenario_type = _infer_web_scenario(
                decision_mode=getattr(decision, "decision_mode", DECISION_MODE_SINGLE_USER),
                source="auto",
                override_flag=int(getattr(decision, "override_flag", 0)),
                review_reason=getattr(decision, "review_reason", ""),
                presence_type="unknown",
                target_user='unknown',
            )
            apply_result = _apply_and_log_device_decision(
                decision,
                env_source="auto",
                user_id='unknown',
                env_log_id=env_id,
                broadcast=False,
            )
            if not apply_result.get("ok"):
                logger.warning(f"[Web] Không áp dụng được quyết định user-left: {apply_result.get('error')}")

            last_prediction = _dashboard_metric_bundle(
                predicted_light=decision.light,
                predicted_fan=decision.fan,
                executed_light=decision.light,
                executed_fan=decision.fan,
                confidence=decision.confidence,
                source='auto',
                ts=datetime.now().isoformat(),
                user_id='unknown',
                auto=True,
                shadow=False,
                scenario_type=scenario_type,
                decision_mode=getattr(decision, 'decision_mode', 'single_user'),
                presence_type='unknown',
                dominant_user_id='unknown',
                data_source='real',
                override_flag=int(getattr(decision, 'override_flag', 0)),
                override_reason=getattr(decision, 'review_reason', ''),
                session_id='',
            )
            with _state_lock:
                shared_state["system"]["last_prediction"] = last_prediction

            _broadcast_event('prediction', last_prediction)

        threading.Thread(target=execute_user_left, daemon=True).start()

def _broadcast_face_update(user_id, confidence, face_count, display_name=''):
    _broadcast_event('face', {
        'user_id': user_id, 'confidence': round(confidence * 100, 1),
        'face_count': face_count, 'display_name': display_name
    })

# ══════════════════════════════════════════════════════════
# ④ LUỒNG ĐIỀU PHỐI AI & MQTT CHUNG
# ══════════════════════════════════════════════════════════

def _pir_auto_turnoff(token: int):
    global _pir_override_until, _pir_timeout_timer, _pir_timeout_token
    with _ai_time_lock:
        if token != _pir_timeout_token:
            return
        _pir_override_until = 0.0
        _pir_timeout_timer = None
        _pir_timeout_token += 1

    with _state_lock:
        shadow = shared_state["system"]["shadow_mode"]
        manual = shared_state["system"]["manual_mode"]
        curr_motion = shared_state["sensor"].get("motion", False)
        curr_temp = shared_state["sensor"].get("temp", 25.0)
        final_user = shared_state["system"]["current_user"]

    if not shadow and not manual and not curr_motion:
        rule_decision = rule_engine.pir_timeout_decide(
            temp=curr_temp,
            light=shared_state["sensor"].get("light", 0.0),
            user_id=final_user,
        )
        decision = coordinator.normalize(rule_decision, user_id=final_user)
        result = _apply_and_log_device_decision(
            decision,
            env_source="pir_timeout_total_off",
            user_id=final_user,
            env_log_id=None,
            broadcast=False,
        )
        if not result.get("ok"):
            logger.warning(f"[Web] Không áp dụng được PIR timeout decision: {result.get('error')}")
        with _state_lock:
            dev_copy = shared_state["devices"].copy()
            temp = shared_state["sensor"]["temp"]
            hum = shared_state["sensor"]["hum"]
            slux = shared_state["sensor"]["light"]
            motion = shared_state["sensor"].get("motion", False)

        logger.info("[Web] 🌙 Đã tắt đèn và quạt do hết thời gian PIR timeout. Phòng trống.")

        _broadcast_event("devices", dev_copy)
        with _state_lock:
            sys_copy = shared_state["system"].copy()
        _broadcast_event("system", sys_copy)

        env_id = db.insert_sensor_data(temp, hum, slux, motion)
        db.log_event_bundle(
            env_log_id=env_id,
            user_id=final_user,
            light=0,
            fan=dev_copy["fan"],
            source="pir_timeout_total_off",
            data_source="real",
            scenario_type="background_noise",
            override_flag=0,
            override_reason="",
            confidence=0.0,
            session_id="",
            presence_type="unknown",
            face_count=0,
            feedback_source="pir_timeout_total_off",
        )


def _coerce_motion_value(motion):
    if isinstance(motion, str):
        return motion.strip().lower() in ['true', '1', 'yes', 'on']
    return bool(motion)


def _is_valid_sensor_number(value, *, min_v=-9999, max_v=9999):
    try:
        v = float(value)
        return math.isfinite(v) and min_v <= v <= max_v
    except Exception:
        return False


def _sensor_issue_details(temp, hum, light):
    issues = []
    checks = {
        "temp": temp,
        "hum": hum,
        "light": light,
    }
    for key, value in checks.items():
        if not _is_valid_sensor_number(value):
            issues.append({"field": key, "value": value, "reason": "invalid_number"})
    return issues


def _detect_anomalies(temp, hum, light, motion, user, mqtt_connected, esp32_online, camera_running, camera_faces, pred=None):
    anomalies = []
    safe_mode = False
    reasons = []
    sensor_issues = _sensor_issue_details(temp, hum, light)
    if not mqtt_connected:
        anomalies.append({"type": "mqtt_offline", "severity": "high", "message": "MQTT offline"})
        safe_mode = True; reasons.append("mqtt_offline")
    if not esp32_online:
        anomalies.append({"type": "esp32_offline", "severity": "high", "message": "ESP32 offline"})
        safe_mode = True; reasons.append("esp32_offline")
    if sensor_issues:
        anomalies.append({
            "type": "invalid_sensor",
            "severity": "high",
            "message": "Dữ liệu sensor không hợp lệ",
            "details": sensor_issues,
        })
        safe_mode = True; reasons.append("invalid_sensor")
    if _is_valid_sensor_number(light, min_v=-1, max_v=5000) and light > 4000:
        anomalies.append({"type": "abnormal_light_change", "severity": "medium", "message": "Ánh sáng thay đổi bất thường"})
    if camera_running and (not camera_faces) and motion:
        anomalies.append({"type": "pir_without_presence", "severity": "high", "message": "Camera mất người nhưng PIR vẫn báo động"})
        safe_mode = True; reasons.append("pir_without_presence")
    if pred and pred.get("confidence", 0) < 0.35:
        anomalies.append({"type": "low_confidence", "severity": "medium", "message": "Độ tin cậy thấp"})
    if any(a["severity"] == "high" for a in anomalies):
        safe_mode = True
    return anomalies, safe_mode, ",".join(reasons)


def _snapshot_sensor_state(temp, hum, light, motion):
    now = datetime.now()
    with _state_lock:
        if face_system:
            shared_state["system"]["current_user"] = face_system.get_current_user()
        shared_state["sensor"] = {
            "temp": temp,
            "hum": hum,
            "light": light,
            "motion": motion,
            "ts": now.isoformat(),
        }
        shared_state["system"]["training_count"] = _get_training_count()
        return {
            "now": now,
            "temp": temp,
            "hum": hum,
            "light": light,
            "motion": motion,
            "user": shared_state["system"].get("current_user", "unknown"),
            "ai_enabled": shared_state["system"].get("ai_enabled", True),
            "shadow_mode": shared_state["system"].get("shadow_mode", True),
            "manual_mode": shared_state["system"].get("manual_mode", False),
            "curr_light": shared_state["devices"].get("light", 0),
            "curr_fan": shared_state["devices"].get("fan", 0),
            "sensor": shared_state["sensor"].copy(),
        }


def _handle_pir_motion(env_id, user, curr_fan, sensor_light, shadow_mode):
    if not getattr(config, 'PIR_AUTO_CONTROL_LIGHT', True):
        return False
    with _state_lock:
        motion_now = shared_state["sensor"].get("motion", False)
    if motion_now and sensor_light < config.ENV_THRESHOLDS.get("light_dark", 100):
        with _ai_time_lock:
            global _pir_override_until, _pir_timeout_timer, _pir_timeout_token
            _pir_override_until = time.time() + getattr(config, 'PIR_DISABLE_AI_MINUTES', 3) * 60
            _pir_timeout_token += 1
            token = _pir_timeout_token
            if _pir_timeout_timer:
                _pir_timeout_timer.cancel()
            _pir_timeout_timer = threading.Timer(getattr(config, 'PIR_TIMEOUT_SECONDS', 60), _pir_auto_turnoff, args=(token,))
            _pir_timeout_timer.daemon = True
            _pir_timeout_timer.start()
        if not shadow_mode:
            sensor_snapshot = shared_state["sensor"].copy()
            rule_decision = rule_engine.pir_motion_decide(
                temp=sensor_snapshot.get("temp", 0.0),
                light=sensor_snapshot.get("light", sensor_light),
                user_id=user,
                fan_state=curr_fan,
            )
            decision = coordinator.normalize(rule_decision, user_id=user)
            result = _apply_and_log_device_decision(decision, env_source="pir_motion", user_id=user, env_log_id=env_id, broadcast=False)
            if not result.get("ok"):
                logger.warning(f"[Web] Không áp dụng được PIR motion decision: {result.get('error')}")
            session_id = _open_face_session_if_needed(
                user_id=user,
                dominant_user_id=user,
                active_users=[user] if user not in (None, "", "unknown", "guest") else [],
                presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown",
                decision_mode=DECISION_MODE_SINGLE_USER,
                source="pir_motion",
            )
            db.log_event_bundle(
                env_log_id=env_id,
                user_id=user,
                light=shared_state["devices"].get("light", 0),
                fan=shared_state["devices"].get("fan", 0),
                source="pir_motion",
                data_source="real",
                scenario_type="sequence_behavior",
                override_flag=0,
                override_reason="",
                confidence=float(getattr(rule_decision, "confidence", 1.0)),
                session_id=session_id,
                presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown",
                face_count=1 if user not in (None, "", "unknown", "guest") else 0,
                feedback_source="pir_motion",
            )
            _touch_face_session(session_id, user_id=user, dominant_user_id=user, active_users=[user] if user not in (None, "", "unknown", "guest") else [], presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown", decision_mode="single_user")
            _broadcast_event("devices", shared_state["devices"].copy())
        return True
    return False


def process_sensor_data(temp, hum, light, motion):
    global _last_ai_run, _manual_override_until, _user_change_until
    is_digital_twin_active = digital_twin_adapter.has_engine() and bool(shared_state.get("digital_twin", {}).get("enabled", False))
    _esp32_heartbeat()
    motion = _coerce_motion_value(motion)

    ctx = _snapshot_sensor_state(temp, hum, light, motion)
    now = ctx["now"]
    user = ctx["user"]
    ai_enabled = ctx["ai_enabled"]
    shadow_mode = ctx["shadow_mode"]
    manual_mode = ctx["manual_mode"]
    curr_fan = ctx["curr_fan"]
    camera_running = bool(shared_state.get("camera", {}).get("running", False))
    camera_faces = list(shared_state.get("camera", {}).get("faces", []) or [])
    anomalies, force_safe_mode, anomaly_reason = _detect_anomalies(
        temp, hum, light, motion, user,
        shared_state["system"].get("mqtt_connected", False),
        shared_state["system"].get("esp32_online", False),
        camera_running, camera_faces
    )
    with _state_lock:
        shared_state["system"]["anomalies"] = anomalies
        shared_state["system"]["safe_mode"] = bool(force_safe_mode)
        shared_state["system"]["safe_mode_reason"] = anomaly_reason
    _broadcast_event("system", {
        "anomalies": anomalies,
        "safe_mode": bool(force_safe_mode),
        "safe_mode_reason": anomaly_reason,
        "mqtt_connected": shared_state["system"].get("mqtt_connected", False),
        "esp32_online": shared_state["system"].get("esp32_online", False),
    })
    if anomalies:
        try:
            for a in anomalies:
                db.insert_anomaly_event(
                    anomaly_type=a.get("type", "unknown"),
                    message=a.get("message", "Anomaly detected"),
                    sensor_json={
                        **ctx["sensor"],
                        "issues": a.get("details", []),
                    },
                    system_json={
                        "mqtt_connected": shared_state["system"].get("mqtt_connected", False),
                        "esp32_online": shared_state["system"].get("esp32_online", False),
                        "safe_mode": bool(force_safe_mode),
                        "safe_mode_reason": anomaly_reason,
                        "current_user": user,
                    },
                    severity=a.get("severity", "medium"),
                    action_taken="safe_mode" if force_safe_mode else "warn",
                    safe_mode=int(bool(force_safe_mode)),
                    user_id=user,
                )
        except Exception as e:
            logger.warning(f"[Web] Không ghi được anomaly event: {e}")

    env_id = db.insert_sensor_data(temp, hum, light, motion)
    _broadcast_event("sensor", ctx["sensor"])

    session_id = _open_face_session_if_needed(
        user_id=user,
        dominant_user_id=user,
        active_users=[user] if user not in (None, "", "unknown", "guest") else [],
        presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown",
        decision_mode="single_user",
        source="periodic_heartbeat",
    )
    db.log_event_bundle(
        env_log_id=env_id,
        user_id=user,
        light=ctx["curr_light"],
        fan=curr_fan,
        source="periodic_heartbeat",
        data_source=_current_data_source(),
        scenario_type="background_noise",
        override_flag=0,
        override_reason="",
        confidence=0.0,
        session_id=session_id,
        presence_type="unknown",
        face_count=0,
        feedback_source="periodic_heartbeat",
    )

    if force_safe_mode:
        with _state_lock:
            shared_state["system"]["shadow_mode"] = True
            shared_state["system"]["manual_mode"] = False
            shared_state["system"]["ai_enabled"] = False
        _broadcast_event("system", {
            "safe_mode": True,
            "safe_mode_reason": anomaly_reason,
            "shadow_mode": True,
            "manual_mode": False,
            "ai_enabled": False,
            "anomalies": anomalies,
        })
        db.log_event_bundle(
            env_log_id=env_id,
            user_id=user,
            light=ctx["curr_light"],
            fan=curr_fan,
            source="safe_mode",
            data_source=_current_data_source(),
            scenario_type="context_exception",
            override_flag=1,
            override_reason=anomaly_reason or "safe_mode",
            confidence=0.0,
            session_id=session_id,
            presence_type="unknown",
            face_count=0,
            feedback_source="safe_mode",
            decision_evidence={
                "sensor_json": ctx["sensor"],
                "evidence_json": {"anomalies": anomalies, "safe_mode": True},
                "reason": anomaly_reason or "safe_mode",
                "active_users": [user] if user not in (None, "", "unknown", "guest") else [],
            },
        )
        return

    if manual_mode:
        return

    pir_handled = _handle_pir_motion(env_id, user, curr_fan, light, shadow_mode)

    if not ai_enabled:
        return

    now_ts = time.time()
    with _ai_time_lock:
        in_cooldown = now_ts < _manual_override_until
        can_infer = (now_ts - _last_ai_run) >= getattr(config, "INFERENCE_INTERVAL_SEC", 120)
        in_user_change_cd = now_ts < _user_change_until
        pir_is_disabling_ai = now_ts < _pir_override_until

    if in_user_change_cd or pir_is_disabling_ai:
        return

    if in_cooldown or not can_infer:
        if pir_handled:
            rule_decision = rule_engine.decide(
                temp=temp,
                light=light,
                motion=motion,
                user_id=user,
                allow_fan_when_hot=False,
                empty_room_safe=False,
            )
            rule_decision["light"] = 1
            rule_decision["fan"] = curr_fan
            session_id = _open_face_session_if_needed(
                user_id=user,
                dominant_user_id=user,
                active_users=[user] if user not in (None, "", "unknown", "guest") else [],
                presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown",
                decision_mode=DECISION_MODE_SINGLE_USER,
                source="pir_only",
            )
            db.log_event_bundle(
                env_log_id=env_id,
                user_id=user,
                light=rule_decision["light"],
                fan=rule_decision["fan"],
                source="pir_only",
                data_source="real",
                scenario_type="background_noise",
                override_flag=0,
                override_reason="",
                confidence=float(getattr(rule_decision, "confidence", 1.0)),
                session_id=session_id,
                presence_type="unknown",
                face_count=0,
                feedback_source="pir_only",
            )
        return

    with _ai_time_lock:
        _last_ai_run = now_ts

    pred = ai.predict(hour=now.hour, day_of_week=now.weekday(), temp=temp, hum=hum,
                      light=light, motion=motion, user_id=user)
    if force_safe_mode:
        pred["light"] = 0
        pred["fan"] = 0
        pred["confidence"] = 0.0
        pred["source"] = f"safe_mode+{pred.get('source','ai')}"
        pred["needs_review"] = True
        pred["review_reason"] = anomaly_reason or "safe_mode"

    with _ai_time_lock:
        pir_active = time.time() < _pir_override_until
    if (pir_handled or pir_active) and pred["light"] == 0:
        pred["light"] = 1

    prediction_payload = {
        "light": pred["light"],
        "fan": pred["fan"],
        "confidence": pred["confidence"],
        "source": pred["source"],
        "scenario_type": _infer_web_scenario(decision_mode=DECISION_MODE_SINGLE_USER, source=pred["source"], override_flag=int(pred.get("override_flag", 0)), review_reason=pred.get("review_reason", ""), presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown", target_user=user),
        "sequence_phase": _infer_web_scenario(decision_mode=DECISION_MODE_SINGLE_USER, source=pred["source"], override_flag=int(pred.get("override_flag", 0)), review_reason=pred.get("review_reason", ""), presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown", target_user=user),
        "override_flag": int(pred.get("override_flag", 0)),
        "override_reason": pred.get("review_reason", ""),
        "session_id": "",
        "decision_mode": "single_user",
        "user_id": user,
        "dominant_user_id": user,
        "presence_type": "human" if user not in (None, "", "unknown", "guest") else "unknown",
        "data_source": "real",
        
        "auto": True,
        "shadow": shadow_mode,
        "safe_mode": force_safe_mode,
        "safe_mode_reason": anomaly_reason,
        "anomalies": anomalies,
        "ts": now.isoformat(),
    }
    with _state_lock:
        shared_state["system"]["last_prediction"] = prediction_payload

    _broadcast_event("prediction", prediction_payload)

    if pred.get("needs_review"):
        try:
            review_reason = pred.get("review_reason", "low_confidence")
            db.log_event_bundle(
                env_log_id=env_id,
                user_id=user,
                light=pred["light"],
                fan=pred["fan"],
                source="needs_review",
                data_source="real",
                scenario_type="context_exception",
                override_flag=1,
                override_reason=review_reason,
                confidence=float(pred.get("confidence", 0.0)),
                session_id="",
                presence_type="human" if user not in ("unknown", "guest") else "unknown",
                face_count=1 if user not in ("unknown", "guest") else 0,
                feedback_source="needs_review",
            )
        except Exception as e:
            logger.debug(f"[Web] Không ghi được needs_review event: {e}")

    if shadow_mode:
        db.log_event_bundle(
            env_log_id=env_id,
            user_id=user,
            light=pred["light"],
            fan=pred["fan"],
            source="shadow",
            data_source="simulated",
            scenario_type="background_noise",
            override_flag=0,
            override_reason="shadow_mode",
            confidence=float(pred.get("confidence", 0.0)),
            session_id="",
            presence_type="human" if user not in ("unknown", "guest") else "unknown",
            face_count=1 if user not in ("unknown", "guest") else 0,
            feedback_source="shadow",
        )
        with _state_lock:
            shared_state["devices"]["light"] = int(bool(pred["light"]))
            shared_state["devices"]["fan"] = int(bool(pred["fan"]))
        _broadcast_event("devices", shared_state["devices"].copy())
        return

    def execute_ai_commands():
        with _state_lock:
            is_shadow_now = shared_state['system']['shadow_mode']
        if is_shadow_now:
            return
        decision = coordinator.normalize({"light": pred["light"], "fan": pred["fan"],
                                          "confidence": pred.get("confidence", 0.0), "source": pred.get("source", "ai")},
                                         user_id=user)
        result = _apply_and_log_device_decision(decision, env_source="auto",
                                                user_id=user, env_log_id=env_id, broadcast=True)
        if not result.get("ok"):
            logger.warning(f"[Web] Không áp dụng được AI decision: {result.get('error')}")

    threading.Thread(target=execute_ai_commands, daemon=True).start()

def _try_init_mqtt():
    global mqtt_handler
    try:
        
        from modules.mqtt_handler import MQTTHandler    
        mqtt_handler = MQTTHandler(db=db, on_sensor_data=process_sensor_data)
        device_service.mqtt_handler = mqtt_handler
        
        def _on_esp32_status_callback(status):
            if status == 'online':
                _esp32_heartbeat()
        mqtt_handler.on_esp32_status = _on_esp32_status_callback
        
        mqtt_handler.connect()
        
        deadline = time.time() + 10.0
        while not getattr(mqtt_handler, 'is_connected', False) and time.time() < deadline:
            time.sleep(0.5)
            
        with _state_lock:
            shared_state["system"]["mqtt_connected"] = getattr(mqtt_handler, "is_connected", False)
            
        if shared_state["system"]["mqtt_connected"]:
            with _state_lock:
                shared_state["devices"]["light"] = mqtt_handler.get_light_state()
                shared_state["devices"]["fan"]   = mqtt_handler.get_fan_state()
            logger.info("[Web] MQTT kết nối thành công, đã đồng bộ trạng thái thiết bị.")
        else:
            logger.warning("[Web] MQTT kết nối timeout sau 10s")
            
    except Exception as e:
        logger.warning(f"[Web] MQTT không khởi động: {e}")

def _demo_sensor_simulator():
    import math, random; t = 0
    while True:
        time.sleep(5); t += 1; h = datetime.now().hour
        temp   = round(28 + 3*math.sin(h*math.pi/12) + random.uniform(-0.5,0.5), 1)
        hum    = round(65 + 10*math.sin(t*0.1) + random.uniform(-2,2), 1)
        light  = round(max(0, 200*math.sin(max(0,h-6)*math.pi/12)+random.uniform(-20,20)), 1)
        motion = random.random() < 0.3
        process_sensor_data(temp, hum, light, motion)

# ══════════════════════════════════════════════════════════
# ⑤ SSE BROADCAST & OTHERS
# ══════════════════════════════════════════════════════════

def _query_history(hours: int) -> dict:
    if   hours <= 1:  bucket_min = 5
    elif hours <= 24: bucket_min = 30
    else:             bucket_min = 120
    since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with db.get_connection() as conn:
            rows = conn.execute("""
                SELECT strftime('%Y-%m-%d %H:', timestamp) || printf('%02d', (cast(strftime('%M', timestamp) as integer) / ?) * ?) AS bucket,
                    AVG(temperature) AS avg_temp, AVG(humidity) AS avg_hum, AVG(light_level) AS avg_light,
                    SUM(motion_detected) AS motion_cnt, COUNT(*) AS cnt
                FROM environmental_logs WHERE timestamp >= ? GROUP BY bucket ORDER BY bucket ASC
            """, (bucket_min, bucket_min, since)).fetchall()
            
        temp_series = [round(r[1], 1) if r[1] else 0 for r in rows]
        hum_series = [round(r[2], 1) if r[2] else 0 for r in rows]
        light_series = [round(r[3], 1) if r[3] else 0 for r in rows]
        motion_series = [int(r[4] or 0) for r in rows]
        return {
            "labels": [r[0] for r in rows],
            "executed_temp": temp_series,
            "executed_hum": hum_series,
            "executed_light": light_series,
            "executed_motion": motion_series,
            "temp": temp_series,
            "hum": hum_series,
            "light": light_series,
            "motion": motion_series,
            "bucket_min": bucket_min,
        }
    except Exception as e:
        logger.error(f"[History] Query lỗi: {e}")
        return {"labels":[],"executed_temp":[],"executed_hum":[],"executed_light":[],"executed_motion":[],"temp":[],"hum":[],"light":[],"motion":[],"bucket_min":bucket_min}

# ══════════════════════════════════════════════════════════
# ⑦ ROUTES – Public
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")

@app.route("/api/camera/stream")
def camera_stream():
    token = _get_token_from_request()
    if not token or not _validate_token(token): 
        return Response("Unauthorized", status=401)
    
    def generate():
        try:
            while True:
                if face_system and face_system._running:
                    if hasattr(face_system, 'get_annotated_frame_bytes'):
                        blob = face_system.get_annotated_frame_bytes()
                    else:
                        blob = _make_placeholder_jpeg("Face System khong co get_annotated_frame")
                else:
                    blob = _make_placeholder_jpeg("Camera chua bat")
                    
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + blob + b"\r\n"
                time.sleep(1/15)
        except GeneratorExit:
            logger.info("[Web] Client ngắt kết nối Camera stream. Đã dọn dẹp Generator.")
        finally:
            pass
            
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-cache"})

def _make_placeholder_jpeg(text="Camera chua bat"):
    img = np.zeros((480,640,3), dtype=np.uint8); img[:] = (26,32,55)
    cv2.putText(img,text,(150,225),cv2.FONT_HERSHEY_SIMPLEX,0.8,(96,165,250),2)
    _,buf = cv2.imencode(".jpg",img); return buf.tobytes()

# ══════════════════════════════════════════════════════════
# ⑧ ROUTES – Protected
# ══════════════════════════════════════════════════════════

@app.route("/api/state")
@require_auth
def get_state():
    with _state_lock: state = json.loads(json.dumps(shared_state))
    
    state["system"]["training_count"] = _get_training_count()
    
    if mqtt_handler and getattr(mqtt_handler, "is_connected", False):
        state["system"]["mqtt_connected"] = True

    if digital_twin_adapter.has_engine():
        state["digital_twin"] = normalize_readout(digital_twin_adapter.read())

    last_pred = state["system"].get("last_prediction") or {}
    state["system"]["last_prediction"] = {
        "light": last_pred.get("executed_light", last_pred.get("predicted_light", last_pred.get("light"))),
        "fan": last_pred.get("executed_fan", last_pred.get("predicted_fan", last_pred.get("fan"))),
        "predicted_light": last_pred.get("predicted_light", last_pred.get("light")),
        "predicted_fan": last_pred.get("predicted_fan", last_pred.get("fan")),
        "actual_light": last_pred.get("actual_light"),
        "actual_fan": last_pred.get("actual_fan"),
        "executed_light": last_pred.get("executed_light", last_pred.get("predicted_light", last_pred.get("light"))),
        "executed_fan": last_pred.get("executed_fan", last_pred.get("predicted_fan", last_pred.get("fan"))),
        "confidence": last_pred.get("confidence"),
        "source": last_pred.get("source"),
        "scenario_type": last_pred.get("scenario_type"),
        "sequence_phase": last_pred.get("sequence_phase", last_pred.get("scenario_type")),
        "override_flag": last_pred.get("override_flag", 0),
        "override_reason": last_pred.get("override_reason", ""),
        "session_id": last_pred.get("session_id", ""),
        "decision_mode": last_pred.get("decision_mode", "single_user"),
        "user_id": last_pred.get("user_id"),
        "dominant_user_id": last_pred.get("dominant_user_id", last_pred.get("user_id")),
        "presence_type": last_pred.get("presence_type", "unknown"),
        "data_source": last_pred.get("data_source", "real"),
        "ts": last_pred.get("ts"),
    }

    if face_system:
        state["camera"] = {"running": face_system._running, "user_id": face_system.get_current_user()}
    else:
        state["camera"] = {"running": False}
        
    return jsonify(state)

@app.route("/api/sensor/latest")
@require_auth
def get_sensor():
    with _state_lock: return jsonify(shared_state["sensor"])

@app.route("/api/digital-twin/state")
@require_auth
def digital_twin_state():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    return jsonify({"ok": True, **normalize_readout(digital_twin_adapter.read())})

@app.route("/api/digital-twin/users", methods=["POST"])
@require_auth
def digital_twin_users():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    data = request.get_json(silent=True) or {}
    users = data.get("users") or []
    readout = digital_twin_adapter.set_showcase_users(users=users)
    _sync_digital_twin_state(readout)
    _log_digital_twin_change("set_users", readout)
    return jsonify({"ok": True, **normalize_readout(readout)})

@app.route("/api/digital-twin/datetime", methods=["POST"])
@require_auth
def digital_twin_datetime():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    data = request.get_json(silent=True) or {}
    when = data.get("datetime") or data.get("timestamp")
    readout = digital_twin_adapter.set_showcase_datetime(when)
    _sync_digital_twin_state(readout)
    _log_digital_twin_change("set_datetime", readout)
    return jsonify({"ok": True, **normalize_readout(readout)})

@app.route("/api/digital-twin/preset", methods=["POST"])
@require_auth
def digital_twin_apply_preset():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    data = request.get_json(silent=True) or {}
    preset = (data.get("preset") or getattr(config, "DIGITAL_TWIN_DEFAULT_PRESET", "comfortable")).strip().lower()
    try:
        readout = digital_twin_adapter.apply_preset(preset)
        _sync_digital_twin_state(readout)
        _log_digital_twin_change(f"preset:{preset}", readout)
        return jsonify({"ok": True, **normalize_readout(readout)})
    except KeyError:
        return jsonify({"ok": False, "error": f"Unknown preset: {preset}"}), 400

@app.route("/api/digital-twin/slider", methods=["POST"])
@require_auth
def digital_twin_slider():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    data = request.get_json(silent=True) or {}
    readout = digital_twin_adapter.update_sliders(
        temperature=data.get("temperature"),
        humidity=data.get("humidity"),
        light=data.get("light"),
    )
    _sync_digital_twin_state(readout)
    _log_digital_twin_change("slider", readout)
    return jsonify({"ok": True, **normalize_readout(readout)})

@app.route("/api/digital-twin/person/enter", methods=["POST"])
@require_auth
def digital_twin_person_enter():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    data = request.get_json(silent=True) or {}
    count = int(data.get("count", 1))
    label = (data.get("label") or "visitor").strip() or "visitor"
    users = data.get("users") or []
    readout = digital_twin_adapter.person_enter(count=count, label=label, users=users)
    _sync_digital_twin_state(readout)
    _log_digital_twin_change("person_enter", readout)
    return jsonify({"ok": True, **normalize_readout(readout)})

@app.route("/api/digital-twin/person/leave", methods=["POST"])
@require_auth
def digital_twin_person_leave():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    readout = digital_twin_adapter.person_leave()
    _sync_digital_twin_state(readout)
    _log_digital_twin_change("person_leave", readout)
    return jsonify({"ok": True, **normalize_readout(readout)})

@app.route("/api/digital-twin/manual", methods=["POST"])
@require_auth
def digital_twin_manual():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    data = request.get_json(silent=True) or {}
    device = (data.get("device") or "").strip().lower()
    state_value = data.get("state")
    duration_sec = int(data.get("duration_sec", 15))
    if device not in {"light", "fan", "ac", "door"}:
        return jsonify({"ok": False, "error": "Invalid device"}), 400
    readout = digital_twin_engine.set_manual_override(device, state_value, duration_sec=duration_sec) if digital_twin_engine else None
    if readout is None:
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    normalized = normalize_readout(readout)
    _sync_digital_twin_state(readout)
    _log_digital_twin_change(f"manual:{device}", readout)
    return jsonify({"ok": True, **normalized})

@app.route("/api/digital-twin/step", methods=["POST"])
@require_auth
def digital_twin_step():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    data = request.get_json(silent=True) or {}
    seconds = int(data.get("seconds", getattr(config, "DIGITAL_TWIN_STEP_SECONDS", 1)))
    readout = digital_twin_adapter.step_and_read(seconds=seconds)
    _sync_digital_twin_state(readout)
    _log_digital_twin_change(f"step:{seconds}", readout)
    return jsonify({"ok": True, **normalize_readout(readout)})

@app.route("/api/camera/autostart", methods=["POST"])
@require_auth
def camera_autostart():
    if face_system and not face_system._running:
        face_system.start()
        return jsonify({"ok": True, "started": True, "msg": "Camera đã bật"})
    return jsonify({"ok": True, "started": False, "msg": "Camera đã đang chạy"})

def _infer_web_scenario(decision_mode: str = "", source: str = "", override_flag: int = 0, review_reason: str = "", event_type: str = "", presence_type: str = "", target_user: str = "") -> str:
    return map_scenario_type(
        decision_mode=decision_mode,
        source=source,
        override_flag=override_flag,
        review_reason=review_reason,
        event_type=event_type,
        presence_type=presence_type,
        target_user=target_user,
    )


def _canonical_event_payload(*, timestamp: datetime, user_id: str, light_state: int, fan_state: int, temp: float, hum: float, light: float, motion: bool, source: str, scenario_type: str, override_flag: int = 0, override_reason: str = "", decision_mode: str = "single_user", presence_type: str = "unknown", face_count: int = 0, dominant_user_id: str | None = None, session_id: str | None = None, data_source: str = "real", confidence: float = 0.0, device_type: str = "web", device_uptime_ms: int = 0, sequence_phase: str | None = None) -> dict:
    return canonicalize_event({
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "day_of_week": timestamp.weekday(),
        "temperature": float(temp),
        "humidity": float(hum),
        "light_level": float(light),
        "motion_detected": int(bool(motion)),
        "user_id": user_id,
        "dominant_user_id": dominant_user_id or user_id,
        "light_state": int(light_state),
        "fan_state": int(fan_state),
        "source": source,
        "data_source": data_source,
        "scenario_type": scenario_type,
        "sequence_phase": sequence_phase or _infer_web_scenario(source=source, override_flag=override_flag, review_reason=override_reason, presence_type=presence_type),
        "override_flag": int(override_flag),
        "override_reason": override_reason,
        "session_id": session_id or "",
        "presence_type": presence_type,
        "face_count": int(face_count),
        "decision_mode": decision_mode,
        "confidence": float(confidence),
        
        "device_type": device_type,
        "device_uptime_ms": int(device_uptime_ms),
    })


def _log_manual_state(light=None, fan=None, source="manual"):
    try:
        with _state_lock:
            s = shared_state["sensor"]
            d = shared_state["devices"]
            user = shared_state["system"]["current_user"]
            temp, hum, slux, motion = s["temp"], s["hum"], s["light"], s.get("motion", False)
            al = light if light is not None else d["light"]
            af = fan if fan is not None else d["fan"]

        env_id = db.insert_sensor_data(temp, hum, slux, motion)
        return env_id, user, al, af, temp, hum, slux, motion
    except Exception as e:
        logger.warning(f"[Web] Ghi manual state lỗi: {e}")
        return None, None, None, None, None, None, None, None


def _open_face_session_if_needed(*, user_id: str, dominant_user_id: str, active_users: list[str], presence_type: str, decision_mode: str, source: str, metadata: dict | None = None) -> str:
    session_user = user_id or "unknown"
    session_id = _active_face_sessions.get(session_user)
    if session_id:
        db.touch_face_session(
            session_id,
            user_id=session_user,
            dominant_user_id=dominant_user_id or session_user,
            active_users=active_users,
            presence_type=presence_type,
            decision_mode=decision_mode,
        )
        return session_id

    session_id = f"sess_{session_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.create_face_session(
        session_id=session_id,
        user_id=session_user,
        dominant_user_id=dominant_user_id or session_user,
        active_users=active_users,
        presence_type=presence_type,
        decision_mode=decision_mode,
        metadata=metadata or {"source": source},
    )
    _active_face_sessions[session_user] = session_id
    return session_id


def _touch_face_session(session_id: str, *, user_id: str, dominant_user_id: str, active_users: list[str], presence_type: str, decision_mode: str):
    if not session_id:
        return
    db.touch_face_session(
        session_id,
        user_id=user_id,
        dominant_user_id=dominant_user_id,
        active_users=active_users,
        presence_type=presence_type,
        decision_mode=decision_mode,
    )


def _close_face_session(user_id: str, *, end_reason: str = "user_left"):
    session_id = _active_face_sessions.pop(user_id, None)
    if session_id:
        try:
            db.close_face_session(session_id, end_reason=end_reason)
        except Exception as e:
            logger.debug(f"[Web] Không đóng được face session {session_id}: {e}")


def _canonical_device_event(*, timestamp: datetime, user_id: str, light: int, fan: int, temp: float, hum: float, light_level: float, motion: bool, source: str, scenario_type: str, override_flag: int = 0, override_reason: str = "", decision_mode: str = "single_user", presence_type: str = "unknown", face_count: int = 0, dominant_user_id: str | None = None, session_id: str | None = None, confidence: float = 0.0, data_source: str = "real", sequence_phase: str | None = None) -> dict:
    return _canonical_event_payload(
        timestamp=timestamp,
        user_id=user_id,
        light_state=light,
        fan_state=fan,
        temp=temp,
        hum=hum,
        light=light_level,
        motion=motion,
        source=source,
        scenario_type=scenario_type,
        override_flag=override_flag,
        override_reason=override_reason,
        decision_mode=decision_mode,
        presence_type=presence_type,
        face_count=face_count,
        dominant_user_id=dominant_user_id,
        session_id=session_id,
        confidence=confidence,
        data_source=data_source,
        sequence_phase=sequence_phase,
    )


def _apply_manual_decision(decision, source: str, log_feedback: bool = False):
    result = device_service.apply_decision(decision)
    if not result.get("ok"):
        return result
    if log_feedback:
        env_id, user, al, af, temp, hum, slux, motion = _log_manual_state(light=decision.light, fan=decision.fan, source=source)
        if env_id is not None:
            try:
                active_users = [user] if user not in (None, "", "unknown", "guest") else []
                session_id = _open_face_session_if_needed(
                    user_id=user or "unknown",
                    dominant_user_id=user or "unknown",
                    active_users=active_users,
                    presence_type="human" if active_users else "unknown",
                    decision_mode=DECISION_MODE_MANUAL_OVERRIDE,
                    source=source,
                    metadata={"reason": source},
                )
                payload = _canonical_device_event(
                    timestamp=datetime.now(),
                    user_id=user,
                    light=al,
                    fan=af,
                    temp=temp,
                    hum=hum,
                    light_level=slux,
                    motion=motion,
                    source=source,
                    scenario_type="manual_override",
                    override_flag=1,
                    override_reason=source,
                    decision_mode=DECISION_MODE_MANUAL_OVERRIDE,
                    presence_type="human" if active_users else "unknown",
                    face_count=len(active_users),
                    dominant_user_id=user,
                    session_id=session_id,
                    confidence=float(getattr(decision, "confidence", 1.0)),
                    data_source="real",
                    sequence_phase="override_response",
                )
                db.log_event_bundle(
                    env_log_id=env_id,
                    user_id=user or "unknown",
                    light=al,
                    fan=af,
                    source=source,
                    data_source="real",
                    scenario_type="manual_override",
                    override_flag=1,
                    override_reason=source,
                    confidence=float(getattr(decision, "confidence", 1.0)),
                    session_id=session_id,
                    presence_type="human" if active_users else "unknown",
                    face_count=len(active_users),
                    feedback_source=source,
                    actual_light=al,
                    actual_fan=af,
                    predicted_light=int(bool(getattr(decision, "light", al))),
                    predicted_fan=int(bool(getattr(decision, "fan", af))),
                )
                db.set_manual_override_cooldown(
                    user_id=user or "unknown",
                    expires_at=(datetime.now() + timedelta(seconds=_MANUAL_COOLDOWN_SEC)).strftime("%Y-%m-%d %H:%M:%S"),
                    reason=source,
                    source=source,
                )
                _touch_face_session(session_id, user_id=user or "unknown", dominant_user_id=user or "unknown", active_users=active_users, presence_type="human" if active_users else "unknown", decision_mode="manual")
            except Exception as e:
                logger.warning(f"[Web] Không ghi được manual device state: {e}")
    return result


def _sync_device_state_from_result(decision, result):
    """Đồng bộ shared_state sau khi DeviceService thực thi thành công."""
    if not result or not result.get("ok"):
        return
    with _state_lock:
        shared_state["devices"]["light"] = int(bool(decision.light))
        shared_state["devices"]["fan"] = int(bool(decision.fan))
        shared_state["system"]["last_prediction"] = {
            "light": int(bool(decision.light)),
            "fan": int(bool(decision.fan)),
            "confidence": float(getattr(decision, "confidence", 0.0)),
            "source": getattr(decision, "source", "unknown"),
            "user_id": getattr(decision, "user_id", "unknown"),
            "ts": datetime.now().isoformat(),
        }


def _collect_decision_evidence_snapshot(user_id: str, decision, env_source: str, override_reason: str = "", sensor_snapshot: dict | None = None):
    with _state_lock:
        sensor = (sensor_snapshot or shared_state.get("sensor", {}) or {}).copy()
        system = shared_state.get("system", {}).copy()
        devices = shared_state.get("devices", {}).copy()
    evidence = {
        "sensor_json": {
            "temperature": sensor.get("temp", sensor.get("temperature")),
            "humidity": sensor.get("hum", sensor.get("humidity")),
            "light_level": sensor.get("light", sensor.get("light_level")),
            "motion_detected": sensor.get("motion", sensor.get("motion_detected")),
            "timestamp": sensor.get("ts", sensor.get("timestamp")),
        },
        "evidence_json": {
            "current_user": user_id,
            "dominant_user_id": getattr(decision, "dominant_user_id", user_id),
            "active_users": list(getattr(decision, "active_users", ()) or []),
            "decision_mode": getattr(decision, "decision_mode", "single_user"),
            "priority_reason": getattr(decision, "priority_reason", ""),
            "source": env_source,
            "override_reason": override_reason,
            "current_devices": devices,
            "ai_enabled": system.get("ai_enabled", True),
            "shadow_mode": system.get("shadow_mode", True),
        },
        "reason": getattr(decision, "priority_reason", "") or env_source or "ai",
        "frame_url": "",
        "frame_path": "",
        "active_users": list(getattr(decision, "active_users", ()) or []),
    }
    if face_system and getattr(face_system, "_running", False):
        try:
            if hasattr(face_system, "get_last_frame_path"):
                evidence["frame_path"] = face_system.get_last_frame_path() or ""
            if hasattr(face_system, "get_last_frame_url"):
                evidence["frame_url"] = face_system.get_last_frame_url() or ""
            if not evidence["frame_url"] and hasattr(face_system, "get_annotated_frame_bytes"):
                evidence["frame_url"] = "/api/camera/stream"
        except Exception:
            logger.debug("[Evidence] Không lấy được frame snapshot", exc_info=True)
    return evidence


def _apply_and_log_device_decision(decision, env_source: str, user_id: str, env_log_id: int = None, broadcast: bool = True):
    """Thực thi một quyết định thiết bị theo cùng một luồng chuẩn."""
    override_reason = getattr(decision, "review_reason", "") or ""
    sensor_snapshot = _snapshot_sensor_state(
        shared_state["sensor"].get("temp"),
        shared_state["sensor"].get("hum"),
        shared_state["sensor"].get("light"),
        shared_state["sensor"].get("motion"),
    )["sensor"]
    evidence = _collect_decision_evidence_snapshot(
        user_id,
        decision,
        env_source,
        override_reason=override_reason,
        sensor_snapshot=sensor_snapshot,
    )
    result = pipeline.apply_and_log(
        decision,
        env_log_id=env_log_id,
        user_id=user_id,
        env_source=env_source,
        data_source=_current_data_source(),
        scenario_type=_infer_web_scenario(env_source, override_flag=int(getattr(decision, "override_flag", 0))),
        override_flag=int(getattr(decision, "override_flag", 0)),
        override_reason=override_reason,
        broadcast=broadcast,
        sensor_snapshot=sensor_snapshot,
        evidence=evidence,
    )
    if result and result.get("ok"):
        _sync_device_state_from_result(decision, result)
    return result

@app.route("/api/device/light", methods=["POST"])
@require_auth
def control_light():
    # [FIX BUG-WEB-01] Ngăn chặn cập nhật trạng thái ảo khi thiết bị offline
    if not check_hardware_availability():
        return jsonify({"ok": False, "error": "Thiết bị hiện đang mất kết nối (Offline).", "code": 503}), 503

    data = request.get_json(silent=True) or {}
    s = bool(data.get("state", False))

    decision = coordinator.normalize(
        {
            "light": 1 if s else 0,
            "fan": shared_state["devices"]["fan"],
            "confidence": 1.0,
            "source": "manual",
        },
        user_id=shared_state["system"]["current_user"],
    )
    result = _apply_manual_decision(decision, "manual", log_feedback=True)
    if not result.get("ok"):
        return jsonify({"ok": False, "error": result.get("error", "Không gửi được lệnh MQTT.")}), 502

    global _manual_override_until, _last_ai_run
    with _ai_time_lock:
        _manual_override_until = time.time() + _MANUAL_COOLDOWN_SEC
        _last_ai_run = time.time()

    with _state_lock:
        current_light = shared_state["devices"].get("light", 0)
        current_user = shared_state["system"].get("current_user", "unknown")
    cooldown_meta = db.get_manual_override_cooldown(current_user) or {}
    return jsonify({"ok": True, "light": current_light, "manual_override_cooldown_until": cooldown_meta.get("expires_at"), "manual_override_reason": cooldown_meta.get("reason", "manual")})

@app.route("/api/device/fan", methods=["POST"])
@require_auth
def control_fan():
    # [FIX BUG-WEB-01] Kiểm tra kết nối trước
    if not check_hardware_availability():
        return jsonify({"ok": False, "error": "Thiết bị hiện đang mất kết nối (Offline).", "code": 503}), 503

    data = request.get_json(silent=True) or {}
    s = bool(data.get("state", False))

    decision = coordinator.normalize(
        {
            "light": shared_state["devices"]["light"],
            "fan": 1 if s else 0,
            "confidence": 1.0,
            "source": "manual",
        },
        user_id=shared_state["system"]["current_user"],
    )
    result = _apply_manual_decision(decision, "manual", log_feedback=True)
    if not result.get("ok"):
        return jsonify({"ok": False, "error": result.get("error", "Không gửi được lệnh MQTT.")}), 502

    global _manual_override_until, _last_ai_run
    with _ai_time_lock:
        _manual_override_until = time.time() + _MANUAL_COOLDOWN_SEC
        _last_ai_run = time.time()

    with _state_lock:
        current_fan = shared_state["devices"].get("fan", 0)
        current_user = shared_state["system"].get("current_user", "unknown")
    cooldown_meta = db.get_manual_override_cooldown(current_user) or {}
    return jsonify({"ok": True, "fan": current_fan, "manual_override_cooldown_until": cooldown_meta.get("expires_at"), "manual_override_reason": cooldown_meta.get("reason", "manual_web")})

@app.route('/api/ai/predict')
@require_auth
def api_predict():
    with _state_lock:
        s = shared_state['sensor']
        user = shared_state['system']['current_user']
    now = datetime.now()
    result = ai.predict(hour=now.hour, day_of_week=now.weekday(), temp=s['temp'],
                        hum=s['hum'], light=s['light'], motion=s.get('motion',False), user_id=user)
    return jsonify(result)

@app.route('/api/ai/explain')
@require_auth
def api_explain_ai():
    with _state_lock:
        s = shared_state['sensor']
        user = shared_state['system']['current_user']
    now = datetime.now()
    explanation = ai.get_feature_importance(
        hour=now.hour,
        day_of_week=now.weekday(),
        temp=s['temp'],
        hum=s['hum'],
        light=s['light'],
        motion=s.get('motion', False),
        user_id=user,
    )
    return jsonify({"ok": True, **explanation})

@app.route('/api/stats/training')
@require_auth
def stats_training():
    return jsonify({'count': _get_training_count(), 'required': getattr(config, 'MIN_TRAINING_ROWS', 500)})

@app.route('/api/stats/history')
@require_auth
def stats_history():
    range_str = request.args.get('range', '24h')
    hours = 1 if range_str == '1h' else (24 if range_str == '24h' else 168)
    return jsonify(_query_history(hours))

@app.route('/api/ai/label', methods=['POST'])
@require_auth
def ai_label():
    data = request.get_json() or {}
    light = int(data.get('light', 0))
    fan = int(data.get('fan', 0))
    reason = data.get('reason', 'manual')
    with _state_lock:
        s = shared_state['sensor']
        user = shared_state['system']['current_user']
        last_pred = shared_state['system'].get('last_prediction') or {}
    env_id = db.insert_sensor_data(s['temp'], s['hum'], s['light'], s.get('motion', False))
    try:
        session_id = _open_face_session_if_needed(
            user_id=user or "unknown",
            dominant_user_id=user or "unknown",
            active_users=[user] if user not in (None, "", "unknown", "guest") else [],
            presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown",
            decision_mode="manual",
            source="manual",
            metadata={"reason": reason},
        )
        payload = _canonical_device_event(
            timestamp=datetime.now(),
            user_id=user,
            light=light,
            fan=fan,
            temp=s['temp'],
            hum=s['hum'],
            light_level=s['light'],
            motion=bool(s.get('motion', False)),
            source='manual',
            scenario_type='manual_override',
            override_flag=1,
            override_reason=reason,
            decision_mode=DECISION_MODE_MANUAL_OVERRIDE,
            presence_type='human' if user not in (None, "", "unknown", "guest") else 'unknown',
            face_count=1 if user not in (None, "", "unknown", "guest") else 0,
            dominant_user_id=user,
            session_id=session_id,
            confidence=float(last_pred.get('confidence', 0.0)),
            data_source='real',
            sequence_phase='override_response',
        )
        db.log_event_bundle(
            env_log_id=env_id,
            user_id=user,
            light=light,
            fan=fan,
            source='manual',
            data_source='real',
            scenario_type='manual_override',
            override_flag=1,
            override_reason=reason,
            confidence=float(last_pred.get('confidence', 0.0)),
            session_id=session_id,
            presence_type='human' if user not in (None, "", "unknown", "guest") else 'unknown',
            face_count=1 if user not in (None, "", "unknown", "guest") else 0,
            feedback_source='manual',
            actual_light=light,
            actual_fan=fan,
            predicted_light=int(bool(last_pred.get('light', light))),
            predicted_fan=int(bool(last_pred.get('fan', fan))),
        )
    except Exception as e:
        logger.debug(f"[Web] Không ghi được prediction override log: {e}")
    
    count = _force_update_training_count()
    return jsonify({
        'ok': True, 
        'count': count, 
        'required': getattr(config, 'MIN_TRAINING_ROWS', 500)
    })

@app.route('/api/active-learning/feedback', methods=['POST'])
@require_auth
def active_learning_feedback():
    data = request.get_json(silent=True) or {}
    with _state_lock:
        s = shared_state['sensor']
        user = shared_state['system']['current_user']
        last_pred = shared_state['system'].get('last_prediction') or {}
    light = int(data.get('light', 0))
    fan = int(data.get('fan', 0))
    reason = data.get('reason', 'active_learning_feedback')
    env_id = db.insert_sensor_data(s['temp'], s['hum'], s['light'], s.get('motion', False))
    decision = coordinator.normalize({"light": light, "fan": fan, "confidence": 1.0, "source": reason}, user_id=user)
    result = _apply_manual_decision(decision, reason, log_feedback=False)
    if not result.get('ok'):
        return jsonify({'ok': False, 'error': result.get('error', 'Không gửi được lệnh MQTT.')}), 502
    try:
        session_id = _open_face_session_if_needed(
            user_id=user or "unknown",
            dominant_user_id=user or "unknown",
            active_users=[user] if user not in (None, "", "unknown", "guest") else [],
            presence_type="human" if user not in (None, "", "unknown", "guest") else "unknown",
            decision_mode="manual",
            source=reason,
            metadata={"reason": reason},
        )
        db.log_event_bundle(
            env_log_id=env_id,
            user_id=user,
            light=light,
            fan=fan,
            source=reason,
            data_source='real',
            scenario_type='manual_override',
            override_flag=1,
            override_reason=reason,
            confidence=1.0,
            session_id=session_id,
            presence_type='human' if user not in (None, "", "unknown", "guest") else 'unknown',
            face_count=1 if user not in (None, "", "unknown", "guest") else 0,
            feedback_source='active_learning_feedback',
            actual_light=light,
            actual_fan=fan,
        )
        ai.record_override_feedback(
            user_id=user,
            predicted={"light": int(last_pred.get('light', 0)), "fan": int(last_pred.get('fan', 0)), "confidence": float(last_pred.get('confidence', 0.0)), "source": last_pred.get('source', reason)},
            actual={
                "light": light,
                "fan": fan,
                "temperature": s['temp'],
                "humidity": s['hum'],
                "light_level": s['light'],
                "motion_detected": int(bool(s.get('motion', False))),
                "hour": datetime.now().hour,
                "day_of_week": datetime.now().weekday(),
                "presence_type": "human" if user not in ("unknown", "guest") else "unknown",
                "face_count": 1 if user not in ("unknown", "guest") else 0,
                "dominant_user_id": user,
                "session_id": session_id,
                "decision_mode": "manual",
                
            },
            source="active_learning_feedback",
            reason=reason,
        )
    except Exception as e:
        logger.debug(f"[Web] Không ghi được active-learning prediction feedback: {e}")
    count = _force_update_training_count()
    return jsonify({'ok': True, 'count': count})

@app.route('/api/users')
@require_auth
def get_users():
    return jsonify(db.get_all_users())

@app.route("/api/logs/predictions")
@require_auth
def get_prediction_logs():
    limit = min(100, int(request.args.get("limit", 20)))
    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    timestamp as ts,
                    id as prediction_id,
                    env_log_id,
                    user_id,
                    pred_light as predicted_light,
                    pred_fan as predicted_fan,
                    actual_light,
                    actual_fan,
                    COALESCE(actual_light, pred_light) as light,
                    COALESCE(actual_fan, pred_fan) as fan,
                    confidence,
                    executed as auto,
                    overridden,
                    override_reason,
                    feedback_source
                FROM predictions
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
        evidence_rows = {r["prediction_id"]: r for r in db.get_decision_evidence(limit=limit * 2)}
        data = []
        for r in rows:
            item = dict(r)
            item.setdefault("scenario_type", "sequence_behavior")
            item.setdefault("predicted_light", item.get("predicted_light", item.get("light")))
            item.setdefault("predicted_fan", item.get("predicted_fan", item.get("fan")))
            item.setdefault("sequence_phase", item.get("scenario_type", "sequence_behavior"))
            item.setdefault("override_flag", int(item.get("overridden", 0) or 0))
            item.setdefault("override_reason", item.get("override_reason", "") or "")
            item.setdefault("session_id", "")
            item.setdefault("decision_mode", "single_user")
            item.setdefault("dominant_user_id", item.get("user_id", "unknown"))
            item.setdefault("presence_type", "unknown")
            item.setdefault("data_source", "real")
            ev = evidence_rows.get(item.get("prediction_id"))
            if ev:
                item["evidence"] = {
                    "sensor": json.loads(ev.get("sensor_json") or "{}"),
                    "meta": json.loads(ev.get("evidence_json") or "{}"),
                    "frame_url": ev.get("frame_url") or "",
                    "frame_path": ev.get("frame_path") or "",
                    "reason": ev.get("reason") or "",
                    "confidence": ev.get("confidence") or item.get("confidence") or 0,
                    "override_reason": ev.get("override_reason") or item.get("override_reason") or "",
                    "active_users": json.loads(ev.get("active_users") or "[]"),
                }
            else:
                item["evidence"] = {
                    "sensor": {}, "meta": {}, "frame_url": "", "frame_path": "", "reason": "",
                    "confidence": item.get("confidence") or 0, "override_reason": item.get("override_reason") or "", "active_users": [],
                }
            data.append(item)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        logger.error(f"[Logs] Lỗi tải lịch sử dự đoán: {e}")
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/logs/<int:row_id>", methods=["DELETE"])
@require_auth
def delete_log_row(row_id: int):
    try:
        deleted = db.delete_device_state(row_id)
        return jsonify({"ok": True, "deleted": deleted})
    except Exception as e:
        logger.error(f"[Logs] Lỗi xóa dòng log {row_id}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/logs/export")
@require_auth
def export_logs():
    fmt = request.args.get('format', 'csv')
    hours = int(request.args.get('hours', 24))
    since = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
    
    rows = []
    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT e.timestamp, e.temperature, e.humidity, e.light_level, 
                       d.user_id, d.dominant_user_id, d.light_state AS executed_light, d.fan_state AS executed_fan, d.source,
                       d.data_source, d.scenario_type, d.sequence_phase, d.override_flag,
                       d.override_reason, d.presence_type, d.face_count, d.session_id,
                       d.decision_mode, d.confidence,
                       p.pred_light AS predicted_light, p.pred_fan AS predicted_fan,
                       p.actual_light AS actual_light, p.actual_fan AS actual_fan,
                       p.executed AS executed
                FROM environmental_logs e 
                LEFT JOIN device_states d ON e.id = d.env_log_id 
                LEFT JOIN predictions p ON p.env_log_id = e.id AND (p.user_id = d.user_id OR p.user_id IS NULL)
                WHERE e.timestamp >= ?
                ORDER BY e.timestamp DESC
            """
            rows = conn.execute(query, (since,)).fetchall()
    except Exception as e:
        logger.error(f"[Export] Lỗi truy vấn DB: {e}")

    if fmt == 'xlsx':
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Smart Home Logs"
            ws.append(['Thời gian', 'Nhiệt độ (°C)', 'Độ ẩm (%)', 'Độ sáng (lux)', 'User ID', 'Presence', 'Face Count', 'Session ID', 'Event Type', 'Actor', 'Trigger', 'Override Reason', 'Sequence Phase', 'Override Flag', 'Scenario Type', 'Data Source', 'Prediction Confidence', 'Pred Light', 'Pred Fan', 'Actual Light', 'Actual Fan', 'Executed', 'Nguồn lệnh'])
            
            for r in rows:
                ws.append([r['timestamp'], r['temperature'], r['humidity'], r['light_level'], 
                           r['user_id'], r.get('dominant_user_id', r.get('user_id', '')), r.get('presence_type', ''), r.get('face_count', ''), r.get('session_id', ''),
                           r.get('decision_mode', 'single_user'), r.get('override_reason', ''), r.get('override_flag', ''), r.get('scenario_type', ''),
                           r.get('sequence_phase', ''), r.get('data_source', ''), r.get('confidence', ''),
                           r.get('predicted_light', ''), r.get('predicted_fan', ''), r.get('actual_light', ''), r.get('actual_fan', ''), r.get('executed', ''), r['source']])
            
            from io import BytesIO
            output = BytesIO()
            wb.save(output)
            output.seek(0)
            
            return Response(
                output.getvalue(), 
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-disposition": f"attachment; filename=smarthome_logs_{hours}h.xlsx"}
            )
        except ImportError:
            logger.error("[Export] Thư viện openpyxl chưa được cài đặt. Hệ thống sẽ tự động xuất file CSV.")
            fmt = 'csv' 
    
    if fmt == 'csv':
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['Thời gian', 'Nhiệt độ', 'Độ ẩm', 'Độ sáng', 'User ID', 'Dominant User ID', 'Presence', 'Face Count', 'Session ID', 'Decision Mode', 'Override Reason', 'Override Flag', 'Scenario Type', 'Sequence Phase', 'Data Source', 'Payload Version', 'Confidence', 'Pred Light', 'Pred Fan', 'Actual Light', 'Actual Fan', 'Executed', 'Nguồn lệnh'])
        for r in rows:
            cw.writerow([r['timestamp'], r['temperature'], r['humidity'], r['light_level'], 
                         r['user_id'], r.get('dominant_user_id', r.get('user_id', '')), r.get('presence_type', ''), r.get('face_count', ''), r.get('session_id', ''),
                         r.get('decision_mode', 'single_user'), r.get('override_reason', ''), r.get('override_flag', ''), r.get('scenario_type', ''),
                         r.get('sequence_phase', ''), r.get('data_source', ''), r.get('confidence', ''),
                         r.get('predicted_light', ''), r.get('predicted_fan', ''), r.get('actual_light', ''), r.get('actual_fan', ''), r.get('executed', ''), r['source']])
        return Response(si.getvalue(), mimetype="text/csv", headers={"Content-disposition": f"attachment; filename=smarthome_logs_{hours}h.csv"})

@app.route("/api/system/mode", methods=["POST"])
@require_auth
def toggle_mode():
    data = request.get_json(silent=True) or {}
    with _state_lock:
        system = shared_state["system"]
        prev_manual = bool(system.get("manual_mode", False))
        prev_ai = bool(system.get("ai_enabled", True))
        prev_shadow = bool(system.get("shadow_mode", True))

        if "manual_mode" in data:
            next_manual = bool(data["manual_mode"])
            system["manual_mode"] = next_manual
            if next_manual:
                system["ai_enabled"] = False
                data.setdefault("ai_enabled", False)

        if "ai_enabled" in data:
            next_ai = bool(data["ai_enabled"])
            system["ai_enabled"] = next_ai
            if next_ai:
                system["manual_mode"] = False
                data.setdefault("manual_mode", False)

        if "shadow_mode" in data:
            system["shadow_mode"] = bool(data["shadow_mode"])

        if prev_shadow != system.get("shadow_mode", prev_shadow):
            db.set_meta("shadow_mode", "true" if system["shadow_mode"] else "false")

        db.set_meta("manual_mode", "true" if system.get("manual_mode", False) else "false")
        db.set_meta("ai_enabled", "true" if system.get("ai_enabled", True) else "false")
        sys_copy = system.copy()

    _broadcast_event("system", sys_copy)
    return jsonify({"ok": True, "system": sys_copy, "changed": {
        "manual_mode": prev_manual != sys_copy.get("manual_mode", prev_manual),
        "ai_enabled": prev_ai != sys_copy.get("ai_enabled", prev_ai),
        "shadow_mode": prev_shadow != sys_copy.get("shadow_mode", prev_shadow),
    }})

_train_lock = threading.Lock()
_is_training_web = False

@app.route("/api/ai/train", methods=["POST"])
@require_auth
def train_ai():
    global _is_training_web
    data = request.get_json(silent=True) or {}
    data_source = (data.get("data_source") or getattr(config, "TRAIN_DATA_SOURCE", "hybrid")).strip().lower()
    eval_source = (data.get("eval_source") or getattr(config, "TRAIN_EVAL_SOURCE", "real")).strip().lower()
    if data_source not in ("hybrid", "real", "simulated"):
        data_source = "hybrid"
    if eval_source not in ("hybrid", "real", "simulated"):
        eval_source = "real"

    with _train_lock:
        if _is_training_web:
            return jsonify({"ok": False, "msg": "Hệ thống đang trong quá trình huấn luyện, vui lòng chờ."})
        _is_training_web = True
        
    def _do():
        global _is_training_web
        try:
            r = ai.train(data_source=data_source, eval_source=eval_source)
            if r and r.get("success"):
                with _state_lock:
                    shared_state["system"]["ai_accuracy_light"] = r.get("accuracy_light")
                    shared_state["system"]["ai_accuracy_fan"]   = r.get("accuracy_fan")
                    shared_state["system"]["training_data_source"] = r.get("data_source", data_source)
                    shared_state["system"]["training_report"] = r.get("report")
                    shared_state["system"]["training_quality_audit"] = r.get("quality_audit")
                    shared_state["system"]["training_quality_status"] = (r.get("quality_audit") or {}).get("status")
                
                count = _force_update_training_count()
                shadow_start = db.get_meta('shadow_start_time')
                if shadow_start:
                    elapsed = (datetime.now() - datetime.fromisoformat(shadow_start)).total_seconds()
                    
                    required_rows = getattr(config, 'MIN_TRAINING_ROWS', 500)
                    min_elapsed = getattr(config, 'MIN_SHADOW_ELAPSED_SECONDS', 3600)
                    
                    if count >= required_rows and elapsed > min_elapsed:
                        db.set_meta('shadow_mode', 'false')
                        shared_state["system"]["shadow_mode"] = False
                        _broadcast_event("system", shared_state["system"].copy())
                        logger.info("[Web] 🌑 TỰ ĐỘNG THOÁT SHADOW MODE (Được kích hoạt từ Web Server).")

            r = r or {"success":False}
            r["data_source"] = data_source
            r["eval_source"] = eval_source
            with _state_lock:
                shared_state["system"]["training_data_source"] = data_source
                shared_state["system"]["training_eval_source"] = eval_source
                if r.get("report"):
                    shared_state["system"]["training_report"] = r["report"]
                if r.get("quality_audit"):
                    shared_state["system"]["training_quality_audit"] = r["quality_audit"]
                    shared_state["system"]["training_quality_status"] = r["quality_audit"].get("status")
            _broadcast_event("training", r)
        except Exception as e:
            _broadcast_event("training",{"success":False,"msg":str(e),"data_source":data_source})
        finally:
            with _train_lock: _is_training_web = False
            
    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"ok":True,"msg":f"Bắt đầu khởi chạy huấn luyện AI ({data_source} -> eval {eval_source})...","data_source":data_source,"eval_source":eval_source})

@app.route("/api/anomalies")
@require_auth
def get_anomalies():
    limit = min(200, max(1, int(request.args.get("limit", 50))))
    try:
        return jsonify({"ok": True, "data": db.get_anomaly_events(limit=limit)})
    except Exception as e:
        logger.error(f"[Anomaly] Lỗi tải anomaly log: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/stats/training_report")
@require_auth
def get_training_report():
    report_path = os.path.join(getattr(config, 'DATA_DIR', 'data'), 'training_report.json')
    audit_path = os.path.join(getattr(config, 'DATA_DIR', 'data'), 'training_quality_audit.json')
    if not os.path.exists(report_path) and not os.path.exists(audit_path):
        return jsonify({"ok": False, "error": "No report available"}), 404
    try:
        report = {}
        audit = {}
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        if os.path.exists(audit_path):
            with open(audit_path, 'r', encoding='utf-8') as f:
                audit = json.load(f)
        return jsonify({"ok": True, "report": report, "quality_audit": audit})
    except Exception as e:
        logger.error(f"[Web] Lỗi đọc training report: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/sessions/active")
@require_auth
def get_active_sessions():
    limit = min(200, max(1, int(request.args.get("limit", 20))))
    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT session_id, user_id, dominant_user_id, active_users, presence_type, decision_mode,
                       started_at, ended_at, end_reason, event_count, last_event_at, is_active, metadata_json
                FROM face_sessions
                WHERE is_active = 1
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        data = []
        for r in rows:
            item = dict(r)
            item = _dashboard_metric_bundle(
                predicted_light=None,
                predicted_fan=None,
                actual_light=None,
                actual_fan=None,
                executed_light=item.get("is_active", 0),
                executed_fan=None,
                session_id=item.get("session_id"),
                user_id=item.get("user_id"),
                dominant_user_id=item.get("dominant_user_id", item.get("user_id")),
                active_users=json.loads(item.get("active_users") or "[]") if isinstance(item.get("active_users"), str) else item.get("active_users", []),
                presence_type=item.get("presence_type", "unknown"),
                decision_mode=item.get("decision_mode", "single_user"),
                started_at=item.get("started_at"),
                ended_at=item.get("ended_at"),
                end_reason=item.get("end_reason", ""),
                event_count=item.get("event_count", 0),
                last_event_at=item.get("last_event_at"),
                is_active=item.get("is_active", 0),
                metadata_json=item.get("metadata_json", "{}"),
            )
            data.append(item)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        logger.error(f"[Sessions] Lỗi lấy session đang mở: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/sessions/history")
@require_auth
def get_session_history():
    limit = min(500, max(1, int(request.args.get("limit", 50))))
    user = (request.args.get("user") or "all").strip().lower()
    status = (request.args.get("status") or "all").strip().lower()
    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            where = []
            params = []
            if user != "all":
                where.append("user_id = ?")
                params.append(user)
            if status == "active":
                where.append("is_active = 1")
            elif status == "closed":
                where.append("is_active = 0")
            sql = "SELECT session_id, user_id, dominant_user_id, active_users, presence_type, decision_mode, started_at, ended_at, end_reason, event_count, last_event_at, is_active, metadata_json FROM face_sessions"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        return jsonify({"ok": True, "data": [dict(r) for r in rows]})
    except Exception as e:
        logger.error(f"[Sessions] Lỗi lấy lịch sử session: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/sessions")
@require_auth
def sessions_view():
    html = """
    <!doctype html>
    <html lang="vi">
    <head><meta charset="utf-8"><title>Face Sessions</title></head>
    <body style="font-family:Arial,sans-serif;padding:20px;background:#0f172a;color:#e2e8f0">
      <h1>Face Sessions</h1>
      <p>Trang xem session đang mở và lịch sử session.</p>
      <h2>API</h2>
      <ul>
        <li><code>/api/sessions/active</code></li>
        <li><code>/api/sessions/history</code></li>
      </ul>
    </body></html>
    """
    return render_template_string(html)

@app.route("/api/sessions/export")
@require_auth
def export_sessions():
    fmt = (request.args.get("format") or "csv").strip().lower()
    user = (request.args.get("user") or "all").strip().lower()
    status = (request.args.get("status") or "all").strip().lower()
    limit = min(5000, max(1, int(request.args.get("limit", 1000))))
    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            where = []
            params = []
            if user != "all":
                where.append("user_id = ?")
                params.append(user)
            if status == "active":
                where.append("is_active = 1")
            elif status == "closed":
                where.append("is_active = 0")
            sql = """
                SELECT session_id, user_id, dominant_user_id, active_users, presence_type, decision_mode,
                       started_at, ended_at, end_reason, event_count, last_event_at, is_active, metadata_json
                FROM face_sessions
            """
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            data = [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[Sessions] Lỗi export session: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

    if fmt == "xlsx":
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Face Sessions"
            ws.append(["session_id", "user_id", "dominant_user_id", "active_users", "presence_type", "decision_mode", "started_at", "ended_at", "end_reason", "event_count", "last_event_at", "is_active", "metadata_json"])
            for r in data:
                ws.append([
                    r.get("session_id", ""), r.get("user_id", ""), r.get("dominant_user_id", ""), r.get("active_users", ""),
                    r.get("presence_type", ""), r.get("decision_mode", ""), r.get("started_at", ""), r.get("ended_at", ""),
                    r.get("end_reason", ""), r.get("event_count", 0), r.get("last_event_at", ""), r.get("is_active", 0),
                    r.get("metadata_json", ""),
                ])
            from io import BytesIO
            output = BytesIO()
            wb.save(output)
            output.seek(0)
            return Response(output.getvalue(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-disposition": "attachment; filename=face_sessions.xlsx"})
        except ImportError:
            fmt = "csv"

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["session_id", "user_id", "dominant_user_id", "active_users", "presence_type", "decision_mode", "started_at", "ended_at", "end_reason", "event_count", "last_event_at", "is_active", "metadata_json"])
    for r in data:
        cw.writerow([
            r.get("session_id", ""), r.get("user_id", ""), r.get("dominant_user_id", ""), r.get("active_users", ""),
            r.get("presence_type", ""), r.get("decision_mode", ""), r.get("started_at", ""), r.get("ended_at", ""),
            r.get("end_reason", ""), r.get("event_count", 0), r.get("last_event_at", ""), r.get("is_active", 0),
            r.get("metadata_json", ""),
        ])
    return Response(si.getvalue(), mimetype="text/csv", headers={"Content-disposition": "attachment; filename=face_sessions.csv"})

@app.route("/api/logs")
@require_auth
def get_logs_paginated():
    page  = max(1, int(request.args.get("page",  1)))
    limit = min(100, max(1, int(request.args.get("limit", 20))))
    user  = request.args.get("user", "all")
    offset = (page - 1) * limit
    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            where = "WHERE d.source IN ('manual_override','manual','periodic_heartbeat','simulated','pir_motion','pir_only','pir_timeout','shadow','shadow_pir','auto','ai_session_hold','pir_timeout_total_off')"
            params_count, params_data = [], []
            
            if user != "all":
                where += " AND d.user_id = ?"
                params_count.append(user); params_data.append(user)
                
            total = conn.execute(f"SELECT COUNT(*) FROM device_states d JOIN environmental_logs e ON d.env_log_id = e.id {where}", params_count).fetchone()[0]
            rows = conn.execute(f"""
                SELECT d.id as row_id, e.timestamp, e.temperature, e.humidity, e.light_level, e.motion_detected, 
                       d.user_id, d.light_state, d.fan_state, d.source, d.presence_type, d.face_count,
                       d.session_id, d.override_reason, d.sequence_phase
                FROM device_states d 
                JOIN environmental_logs e ON d.env_log_id = e.id 
                {where} ORDER BY d.timestamp DESC LIMIT ? OFFSET ?
            """, (*params_data, limit, offset)).fetchall()
            
        return jsonify({"data":[dict(r) for r in rows],"page":page,"limit":limit,"total":total,"total_pages":max(1,-(-total//limit))})
    except Exception as e:
        logger.error(f"[Logs] Lỗi truy xuất lịch sử thiết bị: {e}")
        return jsonify({"data":[],"page":page,"limit":limit,"total":0,"total_pages":1})

@app.route("/api/camera/start", methods=["POST"])
@require_auth
def camera_start():
    if face_system and not face_system._running:
        face_system.start()
        return jsonify({"ok":True, "started": True, "msg":"Đã bật camera"})
    return jsonify({"ok":True, "started": False, "msg":"Camera đã đang chạy"})

@app.route("/api/camera/stop", methods=["POST"])
@require_auth
def camera_stop():
    if face_system: face_system.stop()
    return jsonify({"ok": True})

@app.route("/api/users/<user_id>", methods=["DELETE"])
@require_auth
def delete_user(user_id):
    # [FIX BUG-WEB-02] Xóa triệt để user trong SQLite
    if db:
        db.delete_face_profile(user_id)
    
    # Xóa dữ liệu nhận diện trong file encodings.json
    if face_system:
        face_system.delete_user(user_id)
        
    # DỌN DẸP RÁC BỘ NHỚ AI (Khắc phục lỗi tích tụ)
    if ai:
        ai.clear_user_memory(user_id)
        
    return jsonify({"ok": True, "msg": "Đã xóa user và dọn dẹp metadata"})

@app.route('/api/users/register', methods=['POST'])
@require_auth
def register_user_web():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get('user_id', '')).strip().lower()
    display_name = str(data.get('display_name', '')).strip()
    num_samples = int(data.get('num_samples', 15))

    if not user_id or not display_name:
        return jsonify({'ok': False, 'error': 'Thiếu user_id hoặc display_name'}), 400
    if not face_system:
        return jsonify({'ok': False, 'error': 'Hệ thống khuôn mặt chưa sẵn sàng'}), 503
    if not getattr(face_system, '_running', False):
        return jsonify({'ok': False, 'error': 'Camera chưa được bật'}), 503

    def _job():
        try:
            face_system.register_user(user_id, display_name, num_samples=num_samples)
            default_profile = {
                'preferred': 'warm_white',
                'day': 'cool_white',
                'night': 'warm_white',
                'override': 'cool_white',
            }
        except Exception as e:
            logger.error(f"[Web] Lỗi đăng ký user qua web: {e}", exc_info=True)

    threading.Thread(target=_job, daemon=True).start()
    return jsonify({'ok': True, 'msg': 'Đang chụp 15 khung hình đăng ký...'} )

@app.route("/api/events")
def sse_stream():
    token = _get_token_from_request()
    if not token or not _validate_token(token): 
        return Response("Unauthorized", status=401)
        
    q = queue.Queue(maxsize=50)
    with _sse_lock: _sse_clients.append(q)
    def generate():
        with _state_lock: init = json.dumps(shared_state, ensure_ascii=False)
        yield f"event: init\ndata: {init}\n\n"
        try:
            while True:
                try: yield q.get(timeout=30)
                except queue.Empty: yield "event: ping\ndata: {}\n\n"
        except GeneratorExit: pass
        finally:
            with _sse_lock:
                if q in _sse_clients: _sse_clients.remove(q)
    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/digital-twin/start", methods=["POST"])
@require_auth
def digital_twin_start():
    if not digital_twin_adapter.has_engine():
        return jsonify({"ok": False, "error": "Digital twin is disabled"}), 503
    readout = digital_twin_adapter.read()
    _sync_digital_twin_state(readout)
    return jsonify({"ok": True, **normalize_readout(readout)})

def _schedule_retrain():
    def retrain_job():
        global _is_training_web
        try:
            count = _force_update_training_count()
            logger.info(f"[Web-Auto] Tự động train lại AI định kỳ ({count} mẫu)...")
            with _train_lock:
                if _is_training_web: return
                _is_training_web = True
            
            try:
                r = ai.train(
                    data_source=getattr(config, "TRAIN_DATA_SOURCE", "hybrid"),
                    eval_source=getattr(config, "TRAIN_EVAL_SOURCE", "real"),
                )
                if r and r.get("success"):
                    logger.info(f"[Web-Auto] Train xong! Đèn: {r.get('accuracy_light')}%, Quạt: {r.get('accuracy_fan')}%")
                    with _state_lock:
                        shared_state["system"]["ai_accuracy_light"] = r.get("accuracy_light")
                        shared_state["system"]["ai_accuracy_fan"]   = r.get("accuracy_fan")
                        
                        shadow_start = db.get_meta('shadow_start_time')
                        if shadow_start and shared_state["system"]["shadow_mode"]:
                            elapsed = (datetime.now() - datetime.fromisoformat(shadow_start)).total_seconds()
                            required_rows = getattr(config, 'MIN_TRAINING_ROWS', 500)
                            min_elapsed = getattr(config, 'MIN_SHADOW_ELAPSED_SECONDS', 3600)
                            if count >= required_rows and elapsed > min_elapsed:
                                db.set_meta('shadow_mode', 'false')
                                shared_state["system"]["shadow_mode"] = False
                                _broadcast_event("system", shared_state["system"].copy())
                                logger.info("[Web-Auto] 🌑 TỰ ĐỘNG THOÁT SHADOW MODE.")
                    
                    try:
                        days_to_keep = getattr(config, 'LOG_RETENTION_DAYS', 60)
                        deleted_rows = db.prune_old_logs(days_to_keep=days_to_keep)
                        if deleted_rows > 0:
                            logger.info(f"[Web-Auto] Đã dọn dẹp {deleted_rows} dòng log cũ (>{days_to_keep} ngày) để tối ưu SQLite.")
                        
                        # Tự động dọn dẹp session rác
                        db.cleanup_expired_sessions()
                    except Exception as e_prune:
                        logger.error(f"[Web-Auto] Lỗi khi chạy tự động dọn rác DB: {e_prune}")
                        
            finally:
                with _train_lock: _is_training_web = False
        except Exception as e:
            logger.error(f"[Web-Auto] Lỗi retrain định kỳ: {e}")
        finally:
            _schedule_retrain()

    interval = getattr(config, 'RETRAIN_INTERVAL', 86400)
    timer = threading.Timer(interval, retrain_job)
    timer.daemon = True
    timer.start()

# ══════════════════════════════════════════════════════════
# ⑨ MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        import fcntl
        LOCK_FILE = "/tmp/smarthome.lock" if sys.platform != "win32" else os.path.join(os.environ.get("TEMP", "data"), "smarthome.lock")
        lock_fd = open(LOCK_FILE, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            print("❌ LỖI: Một instance Smart Home (main.py hoặc web_server.py) đang hoạt động!")
            print(" Hệ thống không cho phép chạy đồng thời 2 tiến trình điều khiển. Vui lòng tắt server cũ trước.")
            sys.exit(1)
    except ImportError:
        pass 
        
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--live",   action="store_true")
    p.add_argument("--port",   type=int, default=5000)
    p.add_argument("--no-cam", action="store_true", help="Không tự bật camera khi khởi động")
    args = p.parse_args()

    logger.info("=" * 55)
    logger.info("  Smart Home AI Web Server v4.2 - ĐỒNG BỘ YOLO GATEKEEPER")
    logger.info(f"  Tài khoản: {_WEB_USERNAME} / (xem .env)")
    logger.info("=" * 55)

    face_system = FaceRecognitionSystem(
        db=db,
        on_user_detected=_on_user_detected_callback,
        on_user_left=_on_user_left_callback,
        on_face_update=_broadcast_face_update, 
        headless=True
    )

    if args.live:
        logger.info("[Web] Chế độ LIVE – MQTT + Camera")
        _try_init_mqtt()
        if not args.no_cam:
            face_system.start()
    else:
        logger.info("[Web] Chế độ DEMO – giả lập cảm biến")
        threading.Thread(target=_demo_sensor_simulator, daemon=True).start()

    logger.info("[Web] Đang nạp lịch sử ngữ cảnh (Warm-up) vào bộ đệm LSTM...")
    warmup_n = getattr(config, 'LSTM_TIME_STEPS', 8) * 10
    
    recent_logs = db.get_recent_logs(n=warmup_n * 3)
    if len(recent_logs) >= getattr(config, 'LSTM_TIME_STEPS', 8):
        ai.warmup_buffer(recent_logs)
        logger.info("[Web] ✅ Nạp Warm-up LSTM hoàn tất.")
    else:
        logger.info("[Web] Chưa đủ dữ liệu gần đây để Warm-up, bỏ qua.")
        
    _schedule_retrain()

    logger.info(f"[Web] Dashboard: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)