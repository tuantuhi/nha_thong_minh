
import os
from dotenv import load_dotenv

# Lấy đường dẫn tuyệt đối của thư mục chứa file config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nạp các biến môi trường từ file .env (Khắc phục điểm trừ Bảo mật)
load_dotenv()

# ─── Thư mục Data & Database ─────────────────────────────
# Khai báo DATA_DIR và đảm bảo thư mục tồn tại
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_PATH = os.path.join(DATA_DIR, "smart_home.db")

USE_HTTPS = os.getenv("USE_HTTPS", "False").lower() in ("true","1","yes") 

# ─── MQTT cục bộ (Mosquitto) ─────────────────────────────

# [BẢO MẬT] Đọc Cấu hình từ file .env thay vì ghi trực tiếp plaintext
MQTT_BROKER   = os.getenv("MQTT_BROKER", "192.168.77.106") 
MQTT_PORT     = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")           
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

# ==============================================================================
# Tích hợp cấu hình mã hóa TLS cho môi trường Production
# ==============================================================================
MQTT_TLS          = os.getenv("MQTT_TLS", "False").lower() in ("true", "1", "yes")
MQTT_CA_CERTS     = os.getenv("MQTT_CA_CERTS", None)
MQTT_CERTFILE     = os.getenv("MQTT_CERTFILE", None)
MQTT_KEYFILE      = os.getenv("MQTT_KEYFILE", None)
MQTT_TLS_INSECURE = os.getenv("MQTT_TLS_INSECURE", "False").lower() in ("true", "1", "yes")

# ==============================================================================
# Bổ sung server_status cho cơ chế LWT (Last Will Testament)
# ==============================================================================
TOPICS = {
    "sensor"        : "home/room1/sensors",
    "event_log"     : "home/room1/events",
    "light_cmd"     : "home/room1/light/command",
    "fan_cmd"       : "home/room1/fan/command",
    "status"        : "home/room1/status",
    "server_status" : "home/room1/server/status"
}

# ─── AI Model (LSTM + XGBoost Time-Series Hybrid) ────────
# MODEL_PATH đóng vai trò là "Base Path" (Tiền tố). 
# Hệ thống sẽ tự tạo ra các file _xgb_light.json, _xgb_fan.json, _lstm.weights.h5 và _meta.joblib
# Tạo thư mục models nếu chưa có
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
MODEL_PATH          = os.path.join(BASE_DIR, "models/ai_model_hybrid") 

# ─── Digital Twin Showcase / Demo ────────────────────────
ENABLE_DIGITAL_TWIN = os.getenv("ENABLE_DIGITAL_TWIN", "True").lower() in ("true", "1", "yes")
DIGITAL_TWIN_DEFAULT_PRESET = os.getenv("DIGITAL_TWIN_DEFAULT_PRESET", "comfortable")
DIGITAL_TWIN_STEP_SECONDS = int(os.getenv("DIGITAL_TWIN_STEP_SECONDS", "1"))

MIN_TRAINING_ROWS   = 150       # Số bản ghi tối thiểu để bắt đầu huấn luyện trong giai đoạn thử nghiệm

# ==============================================================================
# Bổ sung hằng số cấu hình thời gian Shadow Mode tập trung
# ==============================================================================
MIN_SHADOW_ELAPSED_SECONDS = 86400 # Thời gian tối thiểu phải chạy Shadow Mode trước khi thoát 24h

RETRAIN_INTERVAL    = 86400     # Tự động train lại mỗi 24 giờ 
LOG_RETENTION_DAYS = 60        # Giữ lại log trong 60 ngày, tự động dọn dẹp log cũ sau mỗi lần train để tối ưu SQLite
# =========================================================
# CÁC BIẾN NÂNG CẤP ĐỘ TIN CẬY & HYPERPARAMETERS
# =========================================================
#  Khai báo duy nhất LSTM_TIME_STEPS tại đây, không lặp lại
# 1. Cấu hình chuỗi thời gian (Time-Series)
LSTM_TIME_STEPS = 6         # Số lượng trạng thái lịch sử liên tiếp (Sliding Window) LSTM dùng để phân tích
INFERENCE_INTERVAL_SEC = 60 # Đồng bộ độ phân giải: AI predict mỗi 60s
MAX_SEQUENCE_GAP_SEC = 180  # Khoảng cách tối đa (3 phút) để nối chuỗi, nếu lớn hơn sẽ bỏ qua

# 2. Cấu hình huấn luyện mạng LSTM
LSTM_EPOCHS = 15        # Số vòng huấn luyện
LSTM_BATCH_SIZE = 16    # Kích thước lô dữ liệu

# 3. Cấu hình XGBoost Classifier
XGB_N_ESTIMATORS = 100  # Số lượng cây quyết định
XGB_MAX_DEPTH = 5       # Độ sâu tối đa của cây
XGB_LEARNING_RATE = 0.05 # Tốc độ học thuật toán

# 4. Ngưỡng tự tin (Confidence Threshold)
LIGHT_CONF_THRESH   = 0.65    # Đèn cần độ chắc chắn cao (tránh nhấp nháy)
FAN_CONF_THRESH     = 0.60    # Quạt có thể linh hoạt hơn do có Hysteresis

# ─── Face Recognition ────────────────────────────────────
FACES_DIR         = os.path.join(BASE_DIR, "faces") # Thư mục chứa ảnh mẫu của người dùng
os.makedirs(FACES_DIR, exist_ok=True)

CAMERA_INDEX      = 0                # 0 = webcam mặc định của laptop
FRAME_PROCESS_FPS = 1               # Quét AI nặng mỗi ~1 giây, còn tracking xử lý các frame xen kẽ
FACE_REDETECT_INTERVAL_SEC = 1.8     # Khoảng thời gian giữa 2 lần chạy DeepFace/YOLO
TRACKER_TYPE = "CSRT"               # CSRT chính xác hơn, KCF nhẹ hơn

# Cấu hình lưu trữ bộ đệm nhận diện khuôn mặt (Debounce)
FACE_DEBOUNCE_SECONDS = 20           # Giữ trạng thái nhận diện user trong 20s sau khi mất dấu
FACE_GUEST_GRACE_SECONDS = 6         # Không đổi sang guest nếu user thật vừa được thấy gần đây
FACE_GUEST_CONFIRM_HITS = 3          # Cần số lần guest liên tiếp để xác nhận
FACE_HOST_CONFIRM_HITS = 2           # Cần số lần user thật liên tiếp để xác nhận
FACE_RECOGNITION_CONF_MIN = 0.35     # Ngưỡng tự tin tối thiểu để nhận diện face

# ── Cấu hình DeepFace (thay thế face_recognition) ────────
# Model nhận diện khuôn mặt:
#   "Facenet512" → 512-D, chính xác nhất     ← mặc định
#   "ArcFace"    → 512-D, nhanh hơn
#   "Facenet"    → 128-D, nhẹ nhất
DEEPFACE_MODEL     = "Facenet512"

# Bộ phát hiện khuôn mặt:
#   "opencv"     → nhanh nhất   ← mặc định
#   "ssd"        → cân bằng
#   "retinaface" → chính xác nhất nhưng chậm
DEEPFACE_DETECTOR  = "opencv"

# Cosine distance threshold (0=giống hệt, 1=khác hoàn toàn)
# Facenet512 + cosine: ngưỡng mặc định = 0.30
DEEPFACE_THRESHOLD = 0.32


# ─── Cấu hình PIR ────────────────────────────────────────
PIR_AUTO_CONTROL_LIGHT = True   # Tự động bật đèn khi PIR phát hiện người
PIR_TIMEOUT_SECONDS    = 45    # Tắt đèn sau 45s không có chuyển động

# --- CẤU HÌNH YOLO PERSON DETECTION ---
USE_YOLO = True
YOLO_MODEL = "yolov8n.pt" # Sử dụng bản Nano (n) để chạy realtime mượt nhất trên Laptop
YOLO_CONF_THRESHOLD = 0.45 # Độ tin cậy tối thiểu để xác nhận là người

#  Tự động đồng bộ thời gian khóa AI theo Timeout của PIR
# Đảm bảo AI không can thiệp tắt đèn khi PIR vẫn đang chạy đếm ngược
PIR_DISABLE_AI_MINUTES = PIR_TIMEOUT_SECONDS / 60.0

PIR_AUTO_CONTROL_FAN   = False  # PIR không điều khiển quạt (để AI quyết định)

# ─── Ngưỡng Môi trường (Dành cho Rule-based Fallback & PIR) ────
ENV_THRESHOLDS = {
    "temp_high"   : 29.0,   # Nhiệt độ cao → bật quạt/AC (°C)
    "temp_low"    : 20.0,   # Nhiệt độ thấp
    "light_dark"  : 80,     # Ánh sáng thấp → bật đèn (lux) - Cũng dùng làm ngưỡng cho PIR
    "light_bright": 250,    # Ánh sáng đủ → tắt đèn
    "hum_high"    : 80,     # Độ ẩm cao
}

# ─── Hysteresis điều khiển thiết bị ──────────────────────
# Bật khi vượt ngưỡng ON, chỉ tắt khi tụt xuống dưới ngưỡng OFF.
FAN_HYSTERESIS_ON_OFFSET  = float(os.getenv("FAN_HYSTERESIS_ON_OFFSET", "0.8"))
FAN_HYSTERESIS_OFF_OFFSET = float(os.getenv("FAN_HYSTERESIS_OFF_OFFSET", "0.6"))
LIGHT_HYSTERESIS_ON_OFFSET  = float(os.getenv("LIGHT_HYSTERESIS_ON_OFFSET", "0.0"))
LIGHT_HYSTERESIS_OFF_OFFSET = float(os.getenv("LIGHT_HYSTERESIS_OFF_OFFSET", "20.0"))

# Vùng trễ tối thiểu để tránh nhấp nháy khi AI dao động quanh ngưỡng.
FAN_HYSTERESIS_MIN_ON_TEMP  = float(os.getenv("FAN_HYSTERESIS_MIN_ON_TEMP", "29.0"))
FAN_HYSTERESIS_MIN_OFF_TEMP = float(os.getenv("FAN_HYSTERESIS_MIN_OFF_TEMP", "27.4"))

# Hồ sơ hysteresis theo nhóm người dùng để cá nhân hóa an toàn.
USER_HYSTERESIS_PROFILES = {
    "user_a": {"fan_on_delta": 0.6, "fan_off_delta": 0.8, "light_off_delta": 18.0},
    "user_b": {"fan_on_delta": 0.7, "fan_off_delta": 0.7, "light_off_delta": 20.0},
    "guest": {"fan_on_delta": 1.0, "fan_off_delta": 0.5, "light_off_delta": 22.0},
    "unknown": {"fan_on_delta": 1.0, "fan_off_delta": 0.5, "light_off_delta": 22.0},
}

# Tỉ lệ sinh 4 kịch bản để dataset cân bằng hơn cho train.
SCENARIO_WEIGHTS = {
    "sequence_behavior": 0.25,
    "context_exception": 0.27,
    "manual_override": 0.28,
    "background_noise": 0.19,
}

# ─── Multi-user policy ───────────────────────────────────
# Quy tắc ưu tiên người quyết định khi có nhiều khuôn mặt cùng lúc.
# Số nhỏ hơn = ưu tiên cao hơn.
USER_PRIORITY = {
    "user_b": 1,  # Quyền lực tối cao (VD: Vợ/Người yêu)
    "user_a": 2,  # Ưu tiên số 2 (VD: Chồng/Bạn)
    "guest": 3,   # Khách
    "unknown": 4,
}
MULTI_USER_DECISION_POLICY = os.getenv("MULTI_USER_DECISION_POLICY", "host_priority").strip().lower()
MULTI_USER_FALLBACK_MODE = os.getenv("MULTI_USER_FALLBACK_MODE", "dominant_host").strip().lower()
MULTI_USER_MIN_FACE_COUNT = int(os.getenv("MULTI_USER_MIN_FACE_COUNT", "2"))
MULTI_USER_HOST_GRACE_SECONDS = int(os.getenv("MULTI_USER_HOST_GRACE_SECONDS", "8"))

# ─── Quality Audit Thresholds ────────────────────────────
QUALITY_MIN_ROWS = int(os.getenv("QUALITY_MIN_ROWS", "100"))
QUALITY_MIN_UNIQUE_USERS = int(os.getenv("QUALITY_MIN_UNIQUE_USERS", "2"))
QUALITY_MAX_MISSING_VALUES = int(os.getenv("QUALITY_MAX_MISSING_VALUES", "0"))
QUALITY_MAX_DUPLICATE_RATIO = float(os.getenv("QUALITY_MAX_DUPLICATE_RATIO", "0.1"))
QUALITY_MIN_CLASS_BALANCE_RATIO = float(os.getenv("QUALITY_MIN_CLASS_BALANCE_RATIO", "0.1"))
# Override hiện tại khá hiếm, nên ngưỡng kiểm tra phải mềm hơn để không chặn train quá sớm.
QUALITY_MIN_OVERRIDE_RATIO = float(os.getenv("QUALITY_MIN_OVERRIDE_RATIO", "0.01"))
QUALITY_MAX_DOMINANT_SCENARIO_RATIO = float(os.getenv("QUALITY_MAX_DOMINANT_SCENARIO_RATIO", "0.65"))
QUALITY_FAIL_MAX_WARNINGS = int(os.getenv("QUALITY_FAIL_MAX_WARNINGS", "2"))
QUALITY_WARN_MIN_SCORE = float(os.getenv("QUALITY_WARN_MIN_SCORE", "75.0"))
QUALITY_PASS_MIN_SCORE = float(os.getenv("QUALITY_PASS_MIN_SCORE", "85.0"))


# ─── Logging ─────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FILE   = os.path.join(DATA_DIR, "smart_home.log")