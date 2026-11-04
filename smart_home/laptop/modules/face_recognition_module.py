
import os
import cv2
import json
import logging
import threading
import time
from collections import deque
import numpy as np
from datetime import datetime

# ── [THUẬT TOÁN ƯU TIÊN] BẢNG QUYỀN LỰC GIA ĐÌNH ──────────
# Số càng nhỏ -> Quyền lực càng cao (Ưu tiên số 1)
# Khi có nhiều người trong phòng, AI sẽ phục vụ người có quyền lực cao nhất.
USER_PRIORITY = {
    "user_b": 1,  # Quyền lực tối cao (VD: Vợ/Người yêu)
    "user_a": 2,  # Ưu tiên số 2 (VD: Chồng/Bạn)
    "guest": 3,   # Khách
    "unknown": 4   # Người lạ
}
# ─────────────────────────────────────────────────────────

# ── Import DeepFace với xử lý lỗi ────────────────────────
try:
    from deepface import DeepFace
    from scipy.spatial.distance import cdist
    DEEPFACE_AVAILABLE = True
except ImportError as _import_err:
    DEEPFACE_AVAILABLE = False
    logging.warning(
        f"[Face] Thư viện DeepFace chưa được cài đặt ({_import_err}). "
        "Vui lòng chạy lệnh: pip install deepface tf-keras scipy"
    )

# ── Import YOLO với xử lý lỗi ────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("[Face] Thư viện Ultralytics chưa được cài. YOLO bị vô hiệu hóa.")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

ENCODINGS_FILE = os.path.join(config.FACES_DIR, "encodings.json")

DEEPFACE_MODEL    = getattr(config, "DEEPFACE_MODEL",    "Facenet512")
DEEPFACE_DETECTOR = getattr(config, "DEEPFACE_DETECTOR", "opencv")
COSINE_THRESHOLD  = getattr(config, "FACE_RECOGNITION_CONF_MIN", getattr(config, "DEEPFACE_THRESHOLD", 0.3))

# ─────────────────────────────────────────────────────────
# Hàm tiền xử lý ảnh và Trích xuất đặc trưng (Chuẩn chung cho hệ thống)
# ─────────────────────────────────────────────────────────

def enhance_low_light(img_bgr: np.ndarray) -> np.ndarray:
    try:
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    except Exception as e:
        logger.debug(f"[Face] Lỗi áp dụng CLAHE: {e}")
        return img_bgr 

def _frame_to_embeddings_with_areas(img_bgr: np.ndarray) -> list[tuple[np.ndarray, dict]]:
    img_bgr_enhanced = enhance_low_light(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr_enhanced, cv2.COLOR_BGR2RGB)
    
    try:
        results = DeepFace.represent(
            img_path          = img_rgb,
            model_name        = DEEPFACE_MODEL,
            detector_backend  = DEEPFACE_DETECTOR,
            enforce_detection = False,   
            align             = True,    
        )
        return [
            (np.array(r["embedding"], dtype=np.float32), r["facial_area"])
            for r in results
            if r.get("face_confidence", 1.0) > 0.6
        ]
    except Exception as e:
        logger.debug(f"[Face] Trích xuất đặc trưng thất bại hoặc không có khuôn mặt: {e}")
        return []

# ─────────────────────────────────────────────────────────
# System Class chính
# ─────────────────────────────────────────────────────────

class FaceRecognitionSystem:
    def __init__(self, db, on_user_detected=None, on_user_left=None, on_face_update=None, on_face_event=None, headless: bool = False):
        self.db               = db
        self.on_user_detected = on_user_detected
        self.on_user_left     = on_user_left
        self.on_face_update   = on_face_update
        self.on_face_event    = on_face_event
        self.headless         = headless
        self.known_embeddings = {}    
        self.current_user     = "unknown"
        self.cap              = None
        self._running         = False
        self._thread          = None
        self._last_detection  = {}
        
        self._current_frame   = None
        self._clean_frame     = None  
        self._last_raw_frame   = None
        self._tracker          = None
        self._tracker_bbox     = None
        self._tracker_user_id  = "unknown"
        self._last_ai_scan_ts  = 0.0
        self._last_tracking_ok = False
        
        self._lock            = threading.Lock()
        
        self._user_buffer     = {}  
        self._debounce_window = getattr(config, "FACE_DEBOUNCE_SECONDS", 30)
        self._guest_grace_window = getattr(config, "FACE_GUEST_GRACE_SECONDS", 6)
        self._guest_confirm_hits = int(getattr(config, "FACE_GUEST_CONFIRM_HITS", 3))
        self._host_confirm_hits = int(getattr(config, "FACE_HOST_CONFIRM_HITS", 2))
        self._re_detect_interval = getattr(config, "FACE_REDETECT_INTERVAL_SEC", 1.8)
        self._tracker_type = getattr(config, "TRACKER_TYPE", "CSRT").upper()
        self._last_host_seen_ts = 0.0
        self._stable_user = "unknown"
        self._stable_user_hits = 0
        self._pending_user = "unknown"
        self._pending_user_hits = 0
        self._face_count_history = deque(maxlen=15)
        
        self.yolo_model = None
        if getattr(config, 'USE_YOLO', False) and YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(getattr(config, 'YOLO_MODEL', 'yolov8n.pt'))
                logger.info("[Face] Đã nạp thành công mô hình YOLO Person Detection.")
            except Exception as e:
                logger.error(f"[Face] Lỗi khởi tạo YOLO: {e}")

        os.makedirs(config.FACES_DIR, exist_ok=True)
        self._load_encodings()

        if DEEPFACE_AVAILABLE:
            self._warmup_model()

    def _create_tracker(self):
        try:
            tracker_type = self._tracker_type
            if tracker_type == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            if tracker_type == "KCF" and hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "legacy"):
                if tracker_type == "CSRT" and hasattr(cv2.legacy, "TrackerCSRT_create"):
                    return cv2.legacy.TrackerCSRT_create()
                if tracker_type == "KCF" and hasattr(cv2.legacy, "TrackerKCF_create"):
                    return cv2.legacy.TrackerKCF_create()
        except Exception as e:
            logger.debug(f"[Face] Không tạo được tracker {self._tracker_type}: {e}")
        return None

    def _reset_tracker(self):
        self._tracker = None
        self._tracker_bbox = None
        self._tracker_user_id = "unknown"
        self._last_tracking_ok = False

    def _update_stable_user(self, candidate: str) -> str:
        if candidate not in ("unknown", self._stable_user):
            if candidate == self._pending_user:
                self._pending_user_hits += 1
            else:
                self._pending_user = candidate
                self._pending_user_hits = 1

            required_hits = self._guest_confirm_hits if candidate == "guest" else self._host_confirm_hits
            if self._pending_user_hits >= required_hits:
                self._stable_user = candidate
                self._stable_user_hits = self._pending_user_hits
                self._pending_user = "unknown"
                self._pending_user_hits = 0
                return self._stable_user
            return self._stable_user if self._stable_user != "unknown" else candidate if required_hits <= 1 else "unknown"

        if candidate == self._stable_user:
            self._stable_user_hits += 1
            self._pending_user = "unknown"
            self._pending_user_hits = 0
            return self._stable_user

        return self._stable_user

    def _resolve_priority_user(self, users: list[str]) -> tuple[str, str]:
        valid = [u for u in users if u not in (None, "", "unknown")]
        if not valid:
            return "unknown", "no_active_users"

        priority_map = USER_PRIORITY if isinstance(USER_PRIORITY, dict) else {}
        ranked = sorted(valid, key=lambda uid: (priority_map.get(uid, 999), uid))
        dominant = ranked[0]

        if len(valid) == 1:
            return dominant, "single_user"

        host_candidates = [u for u in valid if priority_map.get(u, 999) <= 2]
        guest_candidates = [u for u in valid if priority_map.get(u, 999) >= 3]
        if host_candidates:
            dominant = sorted(host_candidates, key=lambda uid: (priority_map.get(uid, 999), uid))[0]
            return dominant, f"host_priority:{dominant}"

        if guest_candidates:
            return sorted(guest_candidates, key=lambda uid: uid)[0], "guest_only"

        return dominant, f"multi_user_priority:{dominant}"

    def _build_face_event_payload(self, *, disp_user: str, raw_user_id: str, stable_user_id: str,
                                  dominant_user_id: str, active_users: list[str], decision_reason: str,
                                  decision_mode: str, presence_type: str, face_count: int,
                                  face_count_raw: int, event_type: str, actor_type: str,
                                  source: str, trigger_source: str, scenario_type: str,
                                  confidence: float, raw_confidence: float) -> dict:
        stable_user_id = stable_user_id or disp_user or "unknown"
        raw_user_id = raw_user_id or stable_user_id
        dominant_user_id = dominant_user_id or stable_user_id
        scenario_type = scenario_type if scenario_type in {"background_noise", "sequence_behavior", "context_exception", "manual_override", "profile_personalization"} else "sequence_behavior"
        if decision_mode == "multi_user_priority" or len([u for u in active_users if u not in (None, "", "unknown")]) > 1:
            scenario_type = "profile_personalization"
        if presence_type in {"khach", "guest"} and stable_user_id not in ("unknown", "guest"):
            scenario_type = "profile_personalization"
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": disp_user,
            "raw_user_id": raw_user_id,
            "stable_user_id": stable_user_id,
            "dominant_user_id": dominant_user_id,
            "active_users": active_users,
            "decision_reason": decision_reason,
            "decision_mode": decision_mode,
            "presence_type": presence_type,
            "face_count": int(face_count),
            "face_count_raw": int(face_count_raw),
            "event_type": event_type,
            "actor_type": actor_type,
            "source": source,
            "trigger_source": trigger_source,
            "scenario_type": scenario_type,
            "override_flag": 0,
            "override_reason": "",
            "data_source": "real",
            "prediction_confidence": float(confidence),
            "confidence": float(raw_confidence),
        }

    def _start_tracker(self, frame, bbox, user_id):
        tracker = self._create_tracker()
        if tracker is None:
            return False
        try:
            ok = tracker.init(frame, tuple(map(float, bbox)))
            if ok:
                self._tracker = tracker
                self._tracker_bbox = bbox
                self._tracker_user_id = user_id
                self._last_tracking_ok = True
                return True
        except Exception as e:
            logger.debug(f"[Face] Tracker init lỗi: {e}")
        return False

    def _update_tracker(self, frame):
        if self._tracker is None:
            return False, None
        try:
            ok, bbox = self._tracker.update(frame)
            if ok:
                self._tracker_bbox = bbox
                self._last_tracking_ok = True
                return True, bbox
        except Exception as e:
            logger.debug(f"[Face] Tracker update lỗi: {e}")
        self._reset_tracker()
        return False, None

    def _warmup_model(self):
        try:
            dummy = np.zeros((112, 112, 3), dtype=np.uint8)
            DeepFace.represent(
                img_path          = dummy,
                model_name        = DEEPFACE_MODEL,
                detector_backend  = DEEPFACE_DETECTOR,
                enforce_detection = False,
            )
            logger.info(f"[Face] DeepFace model '{DEEPFACE_MODEL}' ({DEEPFACE_DETECTOR}) đã nạp thành công.")
        except Exception as e:
            logger.debug(f"[Face] Lỗi quá trình Warm-up: {e}")

    def register_user(self, user_id: str, display_name: str, num_samples: int = 15) -> bool:
        if not DEEPFACE_AVAILABLE:
            logger.error("[Face] DeepFace chưa được cài đặt!")
            return False

        user_dir = os.path.join(config.FACES_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)

        is_bg_running = False
        with self._lock:
            if self._running and self.cap and self.cap.isOpened():
                is_bg_running = True
                
        cap = None
        if not is_bg_running:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            if not cap.isOpened():
                logger.error("[Face] Không thể giao tiếp với Camera phần cứng! (Camera đang bị lỗi hoặc chiếm dụng)")
                return False

        embeddings = []
        collected  = 0

        max_attempts = num_samples * 10
        attempts = 0
        start_time = time.time()
        timeout_sec = 120

        logger.info(f"[Face] Đang chụp {num_samples} mẫu cho '{display_name}'...")
        print(f"\n>>> Nhìn thẳng vào camera. Hệ thống đang lấy {num_samples} góc mặt (Timeout: {timeout_sec}s)...")

        try:
            while collected < num_samples:
                attempts += 1
                
                if attempts > max_attempts or (time.time() - start_time) > timeout_sec:
                    logger.warning(f"[Face] Hủy đăng ký do vượt quá thời gian chờ {timeout_sec}s hoặc {max_attempts} lần thử.")
                    print("  [!] Đăng ký thất bại do quá thời gian chờ. Vui lòng thử lại ở nơi có đủ ánh sáng.")
                    break

                if is_bg_running:
                    with self._lock:
                        frame = self._clean_frame.copy() if self._clean_frame is not None else None
                    ret = frame is not None
                else:
                    ret, frame = cap.read()
                    
                if not ret:
                    time.sleep(0.1)
                    continue

                display_frame = frame.copy()
                cv2.putText(display_frame, f"Dang chup: {collected}/{num_samples}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not self.headless:
                    cv2.imshow(f"Dang ky: {display_name}", display_frame)

                embeds_and_areas = _frame_to_embeddings_with_areas(frame)
                embeds = [e for e, _ in embeds_and_areas]

                if len(embeds) == 1:      
                    embeddings.append(embeds[0])
                    img_path = os.path.join(user_dir, f"sample_{collected}.jpg")
                    cv2.imwrite(img_path, frame)
                    collected += 1
                    logger.info(f"[Face] Trích xuất thành công mẫu {collected}/{num_samples}")
                    
                    if self.headless and collected % 3 == 0:
                        print(f"  [Headless] Đang chụp: {collected}/{num_samples}...")
                    
                    time.sleep(0.5) 
                elif len(embeds) > 1:
                    if attempts % 5 == 0:
                        print("  [!] Khung hình có nhiều hơn 1 khuôn mặt. Vui lòng đứng một mình.")

                if not self.headless:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("  [!] Đã hủy đăng ký bởi người dùng.")
                        break
        finally:
            if cap is not None:
                cap.release()
            if not self.headless:
                try:
                    cv2.destroyWindow(f"Dang ky: {display_name}")
                    cv2.waitKey(1) 
                except Exception:
                    pass

        if len(embeddings) < 8:
            logger.warning(f"[Face] Thu thập không đủ dữ liệu ({len(embeddings)}/{num_samples}). Hủy đăng ký.")
            return False

        with self._lock:
            self.known_embeddings[user_id] = embeddings
            self._save_encodings()

        self.db.upsert_face_profile(user_id, display_name)
        logger.info(f"[Face] Đăng ký thành công: {display_name} ({len(embeddings)} mẫu 512-D)")
        return True

    def identify_face(self, frame) -> tuple[str, float, list, list, str]:
        """
        Quét toàn bộ khuôn mặt trong khung hình và trả về:
        - dominant_user: user ưu tiên cao nhất
        - best_conf: độ tin cậy của user được chọn
        - areas: danh sách vùng mặt phát hiện được
        - identified_users: danh sách user hợp lệ được nhận diện trong frame
        - decision_mode: single_user hoặc multi_user_priority
        """
        if not DEEPFACE_AVAILABLE:
            return "unknown", 0.0, [], [], "single_user"

        query_embeds_and_areas = _frame_to_embeddings_with_areas(frame)
        if not query_embeds_and_areas:
            return "unknown", 0.0, [], [], "single_user"

        query_embeds = [e for e, _ in query_embeds_and_areas]
        areas = [a for _, a in query_embeds_and_areas]

        with self._lock:
            if not self.known_embeddings:
                return "guest", 0.0, areas, [], "single_user"

            known_pairs = [
                (uid, emb)
                for uid, embs in self.known_embeddings.items()
                for emb in embs
            ]

        if not known_pairs:
            return "guest", 0.0, areas, [], "single_user"

        known_arr = np.array([p[1] for p in known_pairs], dtype=np.float32)
        distances_matrix = cdist(query_embeds, known_arr, metric='cosine')

        identified_users = []
        per_face_matches = []
        for i, q_emb in enumerate(query_embeds):
            distances = distances_matrix[i]
            min_idx = int(np.argmin(distances))
            min_dist = float(distances[min_idx])
            conf = max(0.0, 1.0 - min_dist)

            if min_dist <= COSINE_THRESHOLD:
                uid = known_pairs[min_idx][0]
                identified_users.append((uid, conf))
                per_face_matches.append(uid)
            else:
                per_face_matches.append("guest")

        if identified_users:
            best_match, best_conf = min(
                identified_users,
                key=lambda x: (USER_PRIORITY.get(x[0], 99), -x[1])
            )
            decision_mode = "multi_user_priority" if len(set([u for u in per_face_matches if u not in (None, "", "unknown", "guest")])) > 1 else "single_user"
        else:
            best_match = "guest"
            best_conf = 0.0
            decision_mode = "single_user"

        return best_match, best_conf, areas, per_face_matches, decision_mode

    def start(self):
        if not DEEPFACE_AVAILABLE:
            logger.warning("[Face] DeepFace không khả dụng, hệ thống Camera bị vô hiệu hóa.")
            return
            
        with self._lock:
            if self._running:
                return
            self._running = True
            
        self._thread  = threading.Thread(
            target=self._recognition_loop,
            name="FaceRecognitionThread",
            daemon=True
        )
        self._thread.start()
        logger.info("[Face] Tiến trình nhận diện sinh trắc học đã khởi động.")

    def stop(self):
        with self._lock:
            self._running = False
            
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0) 
            
        with self._lock:
            if self.cap and self.cap.isOpened():
                logger.info("[Face] Đang bắt buộc đóng Camera phần cứng...")
                self.cap.release()
                self.cap = None
                
        if not self.headless:
            cv2.destroyAllWindows()
        logger.info("[Face] Tiến trình nhận diện sinh trắc học đã dừng an toàn.")

    def _recognition_loop(self):
        with self._lock:
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            is_opened = self.cap.isOpened()
            
        if not is_opened:
            logger.error("[Face] Lỗi truy xuất luồng Video từ Camera!")
            return

        frame_count   = 0
        last_conf     = 0.0 
        face_areas    = [] 
        last_seen_user = "unknown"
        last_re_detect = 0.0
        frame_skip_sec = max(0.5, float(getattr(config, "FACE_REDETECT_INTERVAL_SEC", 1.8)))

        try:
            while True:
                with self._lock:
                    if not self._running:
                        break
                    cap_ref = self.cap
                
                if cap_ref is None or not cap_ref.isOpened():
                    break
                    
                ret, frame = cap_ref.read()
                    
                if not ret:
                    time.sleep(0.1)
                    continue

                with self._lock:
                    self._clean_frame = frame.copy()

                frame_count += 1
                now = time.time()
                should_redetect = (now - last_re_detect) >= frame_skip_sec

                if not should_redetect:
                    tracking_ok, bbox = self._update_tracker(frame)
                    if tracking_ok and bbox is not None:
                        x, y, w, h = map(int, bbox)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(frame, f"Track: {self._tracker_user_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        with self._lock:
                            self.current_user = self._tracker_user_id if self._tracker_user_id else self.current_user
                            self._current_frame = frame.copy()
                        if self.on_face_update:
                            try:
                                self.on_face_update(self.current_user, last_conf, 1, self.current_user if self.current_user != "unknown" else "")
                            except Exception:
                                pass
                        if not self.headless:
                            cv2.imshow("Smart Home Edge - Camera AI", frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        continue
                    self._reset_tracker()
                    should_redetect = True

                # ─── TÍCH HỢP YOLO LOGIC LÀM NGƯỜI GÁC CỔNG ──────────
                person_detected = False
                person_bbox = None
                
                if self.yolo_model is not None:
                    try:
                        results = self.yolo_model(
                            frame,
                            classes=[0],
                            conf=getattr(config, 'YOLO_CONF_THRESHOLD', 0.5),
                            verbose=False,
                            imgsz=480,
                            half=False
                        )
                        for r in results:
                            boxes = r.boxes
                            if len(boxes) > 0:
                                person_detected = True
                                best_box = boxes[0]
                                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                                person_bbox = (x1, y1, x2 - x1, y2 - y1)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    except Exception as e:
                        logger.error(f"[Face] YOLO detect lỗi: {e}", exc_info=True)
                        person_detected = False
                else:
                    person_detected = False
                # ─────────────────────────────────────────────────────

                if should_redetect:
                    last_re_detect = now
                    if person_detected:
                        user_id, last_conf, face_areas, face_candidates, decision_mode = self.identify_face(frame)

                        with self._lock:
                            if user_id != "unknown":
                                self._user_buffer[user_id] = time.time()
                                last_seen_user = user_id
                                if user_id not in ("guest", "unknown"):
                                    self._last_host_seen_ts = time.time()
                                if person_bbox is not None:
                                    self._start_tracker(frame, person_bbox, user_id)
                            elif self.current_user not in ("unknown", "guest"):
                                self._user_buffer[self.current_user] = time.time()
                                self._last_host_seen_ts = time.time()
                                last_seen_user = self.current_user
                    else:
                        self._reset_tracker()
                        user_id, last_conf, face_areas, face_candidates, decision_mode = "unknown", 0.0, [], [], "single_user"

                    user_changed = False
                    resolved_user = last_seen_user if last_seen_user != "unknown" else self.get_current_user()
                    face_count_value = len(face_areas)
                    self._face_count_history.append(face_count_value)
                    stable_face_count = int(round(sum(self._face_count_history) / max(len(self._face_count_history), 1)))
                    active_faces = [u for u in sorted(set(face_candidates + [resolved_user, self._stable_user, self.current_user])) if u not in (None, "", "unknown")]
                    dominant_user, decision_reason = self._resolve_priority_user(active_faces)
                    if len(active_faces) > 1:
                        decision_mode = "multi_user_priority"

                    with self._lock:
                        if resolved_user != "unknown":
                            if resolved_user != self._last_detection.get("_last_announced"):
                                user_changed = True
                                self._last_detection["_last_announced"] = resolved_user
                                logger.debug(f"[Face] Chuyển đổi định danh: {resolved_user}")

                        disp_user = resolved_user
                        current_host = self.current_user
                        host_recently_seen = (time.time() - self._last_host_seen_ts) <= self._guest_grace_window

                    if disp_user == "guest" and current_host not in ("unknown", "guest") and host_recently_seen:
                        disp_user = current_host
                        last_seen_user = current_host
                        user_changed = False
                        logger.debug(f"[Face] Giữ user chủ nhà hiện tại '{current_host}' thay vì guest tạm thời.")

                    disp_user = self._update_stable_user(disp_user)
                    if decision_mode == "multi_user_priority" and dominant_user not in (None, "", "unknown"):
                        disp_user = dominant_user
                    if disp_user == "unknown" and self._stable_user not in ("unknown", "guest"):
                        disp_user = self._stable_user

                    if disp_user != "unknown":
                        now       = time.time()
                        last_time = self._last_detection.get(disp_user, 0)
                        should_emit = user_changed or (now - last_time > self._debounce_window)
                        
                        if should_emit:
                            self._last_detection[disp_user] = now
                            
                            if disp_user != "guest":
                                self._last_host_seen_ts = now
                                self.db.update_face_seen(disp_user)
                            
                            type_str = "CHỦ NHÀ" if disp_user != "guest" else "KHÁCH"
                            conf_str = f"{last_conf:.1%}" if disp_user != "guest" else "N/A"
                            logger.info(
                                f"[Face] Xác thực {type_str}: {disp_user.upper()} "
                                f"(Độ tin cậy={conf_str}) | mode={decision_mode} | dominant={dominant_user}"
                            )
                            if self.on_user_detected:
                                try:
                                    self.on_user_detected(disp_user)
                                except Exception as e:
                                    logger.error(f"[Face] Lỗi Callback on_user_detected: {e}")
                            if self.on_face_event:
                                try:
                                    active_users = [u for u in [disp_user, self._stable_user, self.current_user, dominant_user] if u not in (None, "", "unknown")]
                                    payload = self._build_face_event_payload(
                                        disp_user=disp_user,
                                        raw_user_id=user_id,
                                        stable_user_id=self._stable_user,
                                        dominant_user_id=dominant_user,
                                        active_users=sorted(set(active_users)),
                                        decision_reason=decision_reason,
                                        decision_mode=decision_mode,
                                        presence_type="multi_user" if len(set(active_users)) > 1 else type_str.lower(),
                                        face_count=stable_face_count,
                                        face_count_raw=face_count_value,
                                        event_type="face_seen",
                                        actor_type="face",
                                        source="face_recognition",
                                        trigger_source="camera",
                                        scenario_type="profile_personalization" if decision_mode == "multi_user_priority" else "sequence_behavior",
                                        confidence=float(last_conf),
                                        raw_confidence=float(last_conf),
                                    )
                                    self.on_face_event(payload)
                                except Exception as e:
                                    logger.debug(f"[Face] Lỗi callback on_face_event: {e}")

                # Vẽ khung khuôn mặt (DeepFace)
                for area in face_areas:
                    x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 128, 0), 2)

                with self._lock:
                    disp_user = self.current_user

                active_users = [u for u in {disp_user, self._stable_user, self.current_user} if u not in ("unknown", "guest", None, "")]
                dominant_user, reason = self._resolve_priority_user(active_users)
                if self.on_face_event and disp_user not in ("unknown", "guest"):
                    try:
                        payload = self._build_face_event_payload(
                            disp_user=disp_user,
                            raw_user_id=disp_user,
                            stable_user_id=self._stable_user,
                            dominant_user_id=dominant_user,
                            active_users=sorted(active_users),
                            decision_reason=reason,
                            decision_mode="multi_user_priority" if len(active_users) > 1 else "single_user",
                            presence_type=("multi_user" if len(active_users) > 1 else "single_known"),
                            face_count=len(active_users),
                            face_count_raw=len(active_users),
                            event_type="face_seen",
                            actor_type="face",
                            source="face_recognition",
                            trigger_source="camera",
                            scenario_type="profile_personalization" if len(active_users) > 1 else "sequence_behavior",
                            confidence=1.0,
                            raw_confidence=1.0,
                        )
                        self.on_face_event(payload)
                    except Exception:
                        pass

                if disp_user not in ("unknown", "guest"):
                    label = f"{disp_user} ({last_conf:.0%})"
                    color = (0, 255, 0)
                elif disp_user == "guest":
                    label = "Guest"
                    color = (0, 255, 255) 
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                with self._lock:
                    self.current_user = disp_user

                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, DEEPFACE_MODEL, (20, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                
                if self.on_face_update:
                    try:
                        self.on_face_update(disp_user, last_conf, len(face_areas), disp_user if disp_user != "unknown" else "")
                    except Exception as e:
                        pass
                
                with self._lock:
                    self._current_frame = frame.copy()
                
                self.get_current_user()
                
                if not self.headless:
                    cv2.imshow("Smart Home Edge - Camera AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
        finally:
            with self._lock:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                    self.cap = None
            if not self.headless:
                cv2.destroyAllWindows()

    def get_annotated_frame_bytes(self) -> bytes:
        with self._lock:
            if self._current_frame is None:
                return self._make_placeholder_jpeg("Dang khoi tao Camera...")
            frame_copy = self._current_frame.copy()
            
        ret, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            return buffer.tobytes()
        return self._make_placeholder_jpeg("Loi Encode Frame")

    def _make_placeholder_jpeg(self, text="Camera chua bat"):
        img = np.zeros((480,640,3), dtype=np.uint8)
        img[:] = (26,32,55)
        cv2.putText(img, text, (150,225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (96,165,250), 2)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def get_current_user(self) -> str:
        trigger_left_for = None
        current_to_return = "unknown"
        
        with self._lock:
            now = time.time()
            
            active_users = {
                uid: last_seen for uid, last_seen in self._user_buffer.items()
                if now - last_seen <= self._debounce_window
            }
            
            self._user_buffer = active_users
            
            stale_threshold = self._debounce_window * 10
            stale_users = [uid for uid, ts in self._last_detection.items() if uid != "_last_announced" and (now - ts) > stale_threshold]
            for uid in stale_users:
                self._last_detection.pop(uid, None)
            
            if not active_users:
                if self.current_user != "unknown":
                    logger.info(f"[Face] 👤 {self.current_user} đã rời đi quá {self._debounce_window}s. Reset ngữ cảnh.")
                    trigger_left_for = self.current_user
                    self.current_user = "unknown"
                    self._stable_user = "unknown"
                    self._stable_user_hits = 0
                    self._pending_user = "unknown"
                    self._pending_user_hits = 0
                    self._last_host_seen_ts = 0.0
                    self._user_buffer.clear()
            else:
                real_hosts = [u for u in active_users if u not in ("guest", "unknown")]
                if real_hosts:
                    best_user = min(real_hosts, key=lambda u: USER_PRIORITY.get(u, 99))
                else:
                    best_user = "guest"
                    
                if self.current_user != best_user:
                    self.current_user = best_user
                    if best_user not in ("guest", "unknown"):
                        self._stable_user = best_user
            
            current_to_return = self.current_user
            
        if trigger_left_for and self.on_user_left:
            try:
                self.on_user_left(trigger_left_for)
            except Exception as e:
                logger.error(f"[Face] Lỗi Callback on_user_left: {e}")
                
        return current_to_return

    def _save_encodings(self):
        safe_data = {uid: [emb.tolist() for emb in embs] for uid, embs in self.known_embeddings.items()}
        with open(ENCODINGS_FILE, "w") as f:
            json.dump(safe_data, f)
        logger.info(f"[Face] Đã đồng bộ {len(self.known_embeddings)} Vector khuôn mặt vào đĩa cứng (Định dạng JSON an toàn).")

    def delete_user(self, user_id: str):
        with self._lock:
            self.known_embeddings.pop(user_id, None)
            self._save_encodings()

        user_dir = os.path.join(config.FACES_DIR, user_id)
        if os.path.isdir(user_dir):
            try:
                for name in os.listdir(user_dir):
                    try:
                        os.remove(os.path.join(user_dir, name))
                    except Exception:
                        pass
                os.rmdir(user_dir)
            except Exception as e:
                logger.debug(f"[Face] Không xóa được thư mục người dùng {user_id}: {e}")

    def _load_encodings(self):
        legacy_file = ENCODINGS_FILE.replace('.json', '.pkl')
        
        if not os.path.exists(ENCODINGS_FILE):
            if os.path.exists(legacy_file):
                try:
                    import pickle
                    with open(legacy_file, "rb") as f:
                        self.known_embeddings = pickle.load(f)
                    
                    self._save_encodings() 
                    os.rename(legacy_file, legacy_file + ".bak") 
                    logger.info("[Face] Đã tự động migrate dữ liệu sinh trắc học từ Pickle (.pkl) sang JSON an toàn.")
                    return
                except Exception as e:
                    logger.error(f"[Face] Lỗi migrate file pickle cũ: {e}")
                    self.known_embeddings = {}
                    return
            else:
                logger.info("[Face] Chưa có dữ liệu sinh trắc học. Vui lòng đăng ký người dùng mới.")
                return
                
        try:
            with open(ENCODINGS_FILE, "r") as f:
                data = json.load(f)

            self.known_embeddings = {
                uid: [np.array(emb, dtype=np.float32) for emb in embs] 
                for uid, embs in data.items()
            }

            sample = next(iter(self.known_embeddings.values()), [])
            if sample and len(sample[0]) == 128:
                logger.warning(
                    "[Face] CẢNH BÁO MÂU THUẪN MODEL: Phát hiện Vector 128-D cũ. "
                    "DeepFace Facenet512 yêu cầu Vector 512-D. "
                    "Vui lòng vào Menu chọn [5] Đăng ký khuôn mặt người dùng mới để hệ thống encode lại."
                )
                self.known_embeddings = {}
                return

            logger.info(f"[Face] Đã tải thành công {len(self.known_embeddings)} hồ sơ: {list(self.known_embeddings.keys())}")
        except Exception as e:
            logger.error(f"[Face] Lỗi phân tích file encodings (JSON): {e}")
            self.known_embeddings = {}