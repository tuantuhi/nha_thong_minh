
import sqlite3
import logging
import json
import time
from datetime import datetime, timedelta
from contextlib import contextmanager
import config

from modules.event_schema import canonicalize_event

logger = logging.getLogger(__name__)

# [FIX BUG-DB-02] Thống nhất danh sách các source được phép dùng để huấn luyện AI
VALID_DECISION_MODES = (
    'single_user', 'multi_user', 'manual_override', 'fallback'
)

DEFAULT_TRAINING_SOURCES = (
    'manual_override', 'real', 'simulated', 'shadow', 'periodic_heartbeat'
)

DEFAULT_LOG_ORDER = (
    'manual_override', 'periodic_heartbeat',
    'pir_only', 'pir_motion', 'pir_timeout', 'pir_timeout_total_off',
    'shadow_pir', 'shadow', 'ai_session_hold', 'auto'
)

class Database:
    def prune_old_logs(self, days_to_keep=60):
        """
        Tự động xóa các log môi trường và thiết bị cũ hơn số ngày chỉ định
        để tránh phình to Database SQLite.
        Mặc định chỉ giữ dữ liệu trong 60 ngày.
        """
        rows_to_delete = 0
        try:
            with self.get_connection() as conn:
                # Tính mốc thời gian cắt
                cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Bật PRAGMA foreign_keys để xóa bảng cha (environmental_logs) 
                # thì bảng con (device_states, predictions) tự động bị xóa theo (nếu có ON DELETE CASCADE)
                conn.execute("PRAGMA foreign_keys = ON;")
                
                # Đếm số dòng sẽ xóa (để ghi log)
                cursor = conn.execute("SELECT COUNT(*) FROM environmental_logs WHERE timestamp < ?", (cutoff_date,))
                rows_to_delete = cursor.fetchone()[0]
                
                if rows_to_delete > 0:
                    conn.execute("DELETE FROM environmental_logs WHERE timestamp < ?", (cutoff_date,))
                    
            # [FIX BUG-DB-01] Dọn dẹp dung lượng thừa trong SQLite phải được gọi bên ngoài transaction
            if rows_to_delete > 0:
                conn_vacuum = sqlite3.connect(self.db_path, timeout=10.0)
                conn_vacuum.execute("VACUUM;")
                conn_vacuum.close()
                logger.info(f"[Database] Đã dọn dẹp {rows_to_delete} log cũ hơn 30 ngày và chạy VACUUM tối ưu DB.")
                
            return rows_to_delete
        except Exception as e:
            logger.error(f"[Database] Lỗi khi dọn dẹp log cũ: {e}")
            return 0
        
    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Context manager – tự động đóng kết nối sau khi dùng xong."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row   
        
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        
        # Bắt buộc kích hoạt Foreign Key (Mặc định SQLite sẽ tắt)
        conn.execute("PRAGMA foreign_keys = ON")
        
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[DB] Lỗi transaction: {e}")
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Khởi tạo schema database và kích hoạt tiến trình Migration."""
        # Đặt PRAGMA WAL một lần duy nhất ngoài transaction
        conn_wal = sqlite3.connect(self.db_path, timeout=10.0)
        conn_wal.execute('PRAGMA journal_mode=WAL')
        conn_wal.close()
        
        with self.get_connection() as conn:
            # Bảng quản lý phiên bản schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            
            # Đọc version hiện hành
            row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
            current_version = row["v"] if row and row["v"] is not None else 0
            
            # Gọi cơ chế tự động nâng cấp
            self._migrate(conn, current_version)
            
            row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
            final_version = row["v"] if row else 0
            logger.info(f"[DB] Database sẵn sàng tại: {self.db_path} (Schema Version: v{final_version})")

    def _migrate(self, conn, current_version: int):
        """Cơ chế tự động áp dụng các bản vá nâng cấp schema theo tuần tự."""
        
        # ─── MIGRATION V1: Khởi tạo các bảng cấu trúc cơ sở ──────────────────────────
        if current_version < 1:
            logger.info("[DB-Migration] Áp dụng Schema v1: Tạo các bảng cơ sở...")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS environmental_logs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                    hour            INTEGER NOT NULL,
                    minute          INTEGER NOT NULL,
                    day_of_week     INTEGER NOT NULL,
                    temperature     REAL    NOT NULL,
                    humidity        REAL    NOT NULL,
                    light_level     REAL    NOT NULL,
                    motion_detected INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_env_timestamp ON environmental_logs(timestamp DESC);

                CREATE TABLE IF NOT EXISTS device_states (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    env_log_id      INTEGER REFERENCES environmental_logs(id) ON DELETE CASCADE,
                    user_id         TEXT DEFAULT 'unknown',
                    dominant_user_id TEXT DEFAULT 'unknown',
                    presence_type   TEXT DEFAULT 'unknown',
                    face_count      INTEGER DEFAULT 0,
                    light_state     INTEGER NOT NULL DEFAULT 0,
                    fan_state       INTEGER NOT NULL DEFAULT 0,
                    confidence      REAL DEFAULT 0.0,
                    source          TEXT DEFAULT 'auto',
                    scenario_type   TEXT DEFAULT 'background_noise',
                    sequence_phase  TEXT DEFAULT 'background_noise',
                    override_flag   INTEGER DEFAULT 0,
                    override_reason TEXT DEFAULT '',
                    session_id      TEXT DEFAULT NULL,
                    decision_mode   TEXT DEFAULT 'fallback',
                    data_source     TEXT DEFAULT 'real'
                    
                );
                CREATE INDEX IF NOT EXISTS idx_ds_env_log_id ON device_states(env_log_id);
                CREATE INDEX IF NOT EXISTS idx_ds_timestamp ON device_states(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_ds_source ON device_states(source);
                CREATE INDEX IF NOT EXISTS idx_ds_source_ts ON device_states(source, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_ds_data_source ON device_states(data_source);
                CREATE INDEX IF NOT EXISTS idx_ds_scenario_type ON device_states(scenario_type);

                CREATE TABLE IF NOT EXISTS face_profiles (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id      TEXT    NOT NULL UNIQUE,
                    display_name TEXT    NOT NULL,
                    created_at   TEXT    DEFAULT (datetime('now','localtime')),
                    last_seen    TEXT,
                    visit_count  INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS face_sessions (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id        TEXT NOT NULL UNIQUE,
                    user_id           TEXT NOT NULL,
                    dominant_user_id  TEXT DEFAULT 'unknown',
                    active_users      TEXT DEFAULT '[]',
                    presence_type     TEXT DEFAULT 'unknown',
                    decision_mode     TEXT DEFAULT 'fallback',
                    started_at        TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    ended_at          TEXT,
                    end_reason        TEXT DEFAULT '',
                    event_count       INTEGER DEFAULT 0,
                    last_event_at     TEXT,
                    is_active         INTEGER DEFAULT 1,
                    metadata_json     TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_face_sessions_active ON face_sessions(is_active, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_face_sessions_user ON face_sessions(user_id, started_at DESC);

                CREATE TABLE IF NOT EXISTS predictions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    env_log_id      INTEGER REFERENCES environmental_logs(id) ON DELETE CASCADE,
                    user_id         TEXT DEFAULT 'unknown',
                    pred_light      INTEGER,
                    pred_fan        INTEGER,
                    confidence      REAL,
                    executed        INTEGER DEFAULT 1,
                    actual_light    INTEGER DEFAULT NULL,
                    actual_fan      INTEGER DEFAULT NULL,
                    overridden      INTEGER DEFAULT 0,
                    override_reason TEXT DEFAULT '',
                    feedback_source TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp DESC);

                CREATE TABLE IF NOT EXISTS decision_evidence (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    prediction_id   INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
                    env_log_id      INTEGER REFERENCES environmental_logs(id) ON DELETE CASCADE,
                    user_id         TEXT DEFAULT 'unknown',
                    sensor_json     TEXT DEFAULT '{}',
                    evidence_json   TEXT DEFAULT '{}',
                    frame_url       TEXT DEFAULT '',
                    frame_path      TEXT DEFAULT '',
                    reason          TEXT DEFAULT '',
                    confidence      REAL DEFAULT 0.0,
                    override_reason TEXT DEFAULT '',
                    active_users    TEXT DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_timestamp ON decision_evidence(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_evidence_prediction_id ON decision_evidence(prediction_id);
                
                CREATE TABLE IF NOT EXISTS system_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS prediction_feedback_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    env_log_id INTEGER REFERENCES environmental_logs(id) ON DELETE CASCADE,
                    user_id TEXT DEFAULT 'unknown',
                    source TEXT DEFAULT 'manual_override',
                    predicted_light INTEGER DEFAULT NULL,
                    predicted_fan INTEGER DEFAULT NULL,
                    actual_light INTEGER DEFAULT NULL,
                    actual_fan INTEGER DEFAULT NULL,
                    override_reason TEXT DEFAULT '',
                    feedback_source TEXT DEFAULT '',
                    dominant_user_id TEXT DEFAULT NULL,
                    active_users TEXT DEFAULT '[]',
                    decision_mode TEXT DEFAULT 'fallback',
                    processed INTEGER DEFAULT 0,
                    processed_at TEXT DEFAULT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pfq_processed ON prediction_feedback_queue(processed, created_at DESC);

                INSERT OR IGNORE INTO system_meta (key, value) VALUES ('shadow_mode', 'true');
                INSERT OR IGNORE INTO system_meta (key, value) VALUES ('shadow_start_time', datetime('now', 'localtime'));
            """)
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            current_version = 1

        # ─── MIGRATION V2: đưa device_states về schema canonical rút gọn ───────────
        if current_version < 2:
            logger.info("[DB-Migration] Áp dụng Schema v2: Rút gọn device_states canonical fields...")
            cursor = conn.execute("PRAGMA table_info(device_states)")
            cols = {row["name"] for row in cursor.fetchall()}
            additions = {
                "dominant_user_id": "TEXT DEFAULT 'unknown'",
                "presence_type": "TEXT DEFAULT 'unknown'",
                "face_count": "INTEGER DEFAULT 0",
                "confidence": "REAL DEFAULT 0.0",
                "scenario_type": "TEXT DEFAULT 'background_noise'",
                "sequence_phase": "TEXT DEFAULT 'background_noise'",
                "override_flag": "INTEGER DEFAULT 0",
                "override_reason": "TEXT DEFAULT ''",
                "decision_mode": "TEXT DEFAULT 'fallback'",
            }
            for col, ddl in additions.items():
                if col not in cols:
                    conn.execute(f"ALTER TABLE device_states ADD COLUMN {col} {ddl}")
            conn.execute("INSERT INTO schema_version (version) VALUES (2)")
            current_version = 2

        # ─── MIGRATION V3: chuẩn hóa predicted history & feedback ──────────
        if current_version < 3:
            logger.info("[DB-Migration] Áp dụng Schema v3: Bổ sung các cột tối thiểu cho predictions...")
            cursor = conn.execute("PRAGMA table_info(predictions)")
            cols = {row["name"] for row in cursor.fetchall()}
            if "confidence" not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN confidence REAL DEFAULT 0.0")
            if "scenario_type" not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN scenario_type TEXT DEFAULT 'sequence_behavior'")
            conn.execute("INSERT INTO schema_version (version) VALUES (3)")
            current_version = 3

        if current_version < 4:
            logger.info("[DB-Migration] Áp dụng Schema v4: Chuẩn hóa feedback queue tối thiểu...")
            cursor = conn.execute("PRAGMA table_info(prediction_feedback_queue)")
            cols = {row["name"] for row in cursor.fetchall()}
            if not cols:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_feedback_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                        env_log_id INTEGER REFERENCES environmental_logs(id) ON DELETE CASCADE,
                        user_id TEXT DEFAULT 'unknown',
                        source TEXT DEFAULT 'manual_override',
                        predicted_light INTEGER DEFAULT NULL,
                        predicted_fan INTEGER DEFAULT NULL,
                        actual_light INTEGER DEFAULT NULL,
                        actual_fan INTEGER DEFAULT NULL,
                        override_reason TEXT DEFAULT '',
                        feedback_source TEXT DEFAULT '',
                        decision_mode TEXT DEFAULT 'fallback',
                        processed INTEGER DEFAULT 0,
                        processed_at TEXT DEFAULT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pfq_processed ON prediction_feedback_queue(processed, created_at DESC)")
            conn.execute("INSERT INTO schema_version (version) VALUES (4)")
            current_version = 4

        if current_version < 5:
            logger.info("[DB-Migration] Áp dụng Schema v5: Giữ bảng face_profiles/sessions nguyên trạng...")
            conn.execute("INSERT INTO schema_version (version) VALUES (5)")
            current_version = 5

        if current_version < 6:
            logger.info("[DB-Migration] Áp dụng Schema v6: Giữ bảng logs/history nguyên trạng...")
            conn.execute("INSERT INTO schema_version (version) VALUES (6)")
            current_version = 6

        if current_version < 7:
            logger.info("[DB-Migration] Áp dụng Schema v7: Bổ sung các cột tương thích ngược nếu thiếu...")
            cursor = conn.execute("PRAGMA table_info(device_states)")
            cols = {row["name"] for row in cursor.fetchall()}
            for col, ddl in {
                "session_id": "TEXT DEFAULT NULL",
                "data_source": "TEXT DEFAULT 'real'",
                "light_state": "INTEGER DEFAULT 0",
                "fan_state": "INTEGER DEFAULT 0",
                "source": "TEXT DEFAULT 'auto'",
            }.items():
                if col not in cols:
                    conn.execute(f"ALTER TABLE device_states ADD COLUMN {col} {ddl}")
            conn.execute("INSERT INTO schema_version (version) VALUES (7)")
            current_version = 7

        if current_version < 8:
            logger.info("[DB-Migration] Áp dụng Schema v8: Đánh dấu schema ổn định cuối cùng...")
            conn.execute("INSERT INTO schema_version (version) VALUES (8)")
            current_version = 8

        if current_version < 9:
            logger.info("[DB-Migration] Áp dụng Schema v9: Thêm decision evidence log...")
            cursor = conn.execute("PRAGMA table_info(decision_evidence)")
            cols = {row["name"] for row in cursor.fetchall()}
            if not cols:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS decision_evidence (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp       TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                        prediction_id   INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
                        env_log_id      INTEGER REFERENCES environmental_logs(id) ON DELETE CASCADE,
                        user_id         TEXT DEFAULT 'unknown',
                        sensor_json     TEXT DEFAULT '{}',
                        evidence_json   TEXT DEFAULT '{}',
                        frame_url       TEXT DEFAULT '',
                        frame_path      TEXT DEFAULT '',
                        reason          TEXT DEFAULT '',
                        confidence      REAL DEFAULT 0.0,
                        override_reason TEXT DEFAULT '',
                        active_users    TEXT DEFAULT '[]'
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_timestamp ON decision_evidence(timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_prediction_id ON decision_evidence(prediction_id)")
            conn.execute("INSERT INTO schema_version (version) VALUES (9)")
            current_version = 9

        if current_version < 10:
            logger.info("[DB-Migration] Áp dụng Schema v10: Thêm anomaly log...")
            cursor = conn.execute("PRAGMA table_info(anomaly_events)")
            cols = {row["name"] for row in cursor.fetchall()}
            if not cols:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS anomaly_events (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp     TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                        anomaly_type  TEXT NOT NULL,
                        severity      TEXT NOT NULL DEFAULT 'medium',
                        message       TEXT NOT NULL,
                        sensor_json   TEXT DEFAULT '{}',
                        system_json   TEXT DEFAULT '{}',
                        action_taken  TEXT DEFAULT 'none',
                        safe_mode     INTEGER DEFAULT 0,
                        user_id       TEXT DEFAULT 'unknown'
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_events(timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_type ON anomaly_events(anomaly_type)")
            conn.execute("INSERT INTO schema_version (version) VALUES (10)")
            current_version = 10
            
    # ─────────────────────────────────────────────────────
    # System Meta (Shadow Mode Persistence)
    # ─────────────────────────────────────────────────────
    def get_meta(self, key: str) -> str | None:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM system_meta WHERE key=?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO system_meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, value))

    def set_manual_override_cooldown(self, *, user_id: str, expires_at: str, reason: str, source: str = "manual_web") -> None:
        payload = {
            "user_id": user_id,
            "expires_at": expires_at,
            "reason": reason,
            "source": source,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.set_meta(f"manual_override:{user_id}", json.dumps(payload, ensure_ascii=False))

    def get_manual_override_cooldown(self, user_id: str) -> dict | None:
        raw = self.get_meta(f"manual_override:{user_id}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def clear_manual_override_cooldown(self, user_id: str) -> None:
        self.set_meta(f"manual_override:{user_id}", "")

    def cleanup_expired_sessions(self):
        """Dọn dẹp session token cũ trong system_meta tránh phình to DB"""
        cutoff = time.time()
        with self.get_connection() as conn:
            rows = conn.execute("SELECT key, value FROM system_meta WHERE key LIKE 'session:%'").fetchall()
            for row in rows:
                try:
                    data = json.loads(row["value"])
                    if data.get("expires", 0) < cutoff:
                        conn.execute("DELETE FROM system_meta WHERE key=?", (row["key"],))
                except Exception:
                    pass

    def create_face_session(self, *, session_id: str, user_id: str, dominant_user_id: str = "unknown", active_users: list[str] | None = None, presence_type: str = "unknown", decision_mode: str = "fallback", metadata: dict | None = None) -> None:
        active_users_json = json.dumps(active_users or [], ensure_ascii=False)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO face_sessions (session_id, user_id, dominant_user_id, active_users, presence_type, decision_mode, started_at, ended_at, end_reason, event_count, last_event_at, is_active, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now','localtime'), NULL, '', 0, NULL, 1, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    dominant_user_id=excluded.dominant_user_id,
                    active_users=excluded.active_users,
                    presence_type=excluded.presence_type,
                    decision_mode=excluded.decision_mode,
                    ended_at=NULL,
                    end_reason='',
                    is_active=1,
                    metadata_json=excluded.metadata_json
            """, (session_id, user_id, dominant_user_id, active_users_json, presence_type, decision_mode, metadata_json))

    def close_face_session(self, session_id: str, end_reason: str = "user_left") -> None:
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE face_sessions
                SET ended_at=datetime('now','localtime'), end_reason=?, is_active=0
                WHERE session_id=? AND is_active=1
            """, (end_reason, session_id))

    def touch_face_session(self, session_id: str, *, user_id: str | None = None, dominant_user_id: str | None = None, active_users: list[str] | None = None, presence_type: str | None = None, decision_mode: str | None = None, increment: int = 1) -> None:
        updates = ["event_count = event_count + ?", "last_event_at = datetime('now','localtime')"]
        params = [increment]
        if user_id is not None:
            updates.append("user_id = ?")
            params.append(user_id)
        if dominant_user_id is not None:
            updates.append("dominant_user_id = ?")
            params.append(dominant_user_id)
        if active_users is not None:
            updates.append("active_users = ?")
            params.append(json.dumps(active_users, ensure_ascii=False))
        if presence_type is not None:
            updates.append("presence_type = ?")
            params.append(presence_type)
        if decision_mode is not None:
            updates.append("decision_mode = ?")
            params.append(decision_mode)
        params.append(session_id)
        with self.get_connection() as conn:
            conn.execute(f"UPDATE face_sessions SET {', '.join(updates)} WHERE session_id=?", params)

    def get_active_face_session(self, user_id: str | None = None) -> dict | None:
        with self.get_connection() as conn:
            if user_id:
                row = conn.execute("SELECT * FROM face_sessions WHERE is_active=1 AND user_id=? ORDER BY started_at DESC LIMIT 1", (user_id,)).fetchone()
            else:
                row = conn.execute("SELECT * FROM face_sessions WHERE is_active=1 ORDER BY started_at DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    # ─────────────────────────────────────────────────────
    # CRUD: Environmental Logs & Training
    # ─────────────────────────────────────────────────────
    def insert_sensor_data(self, temp: float, hum: float, light: float, motion: bool = False) -> int:
        now = datetime.now()
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO environmental_logs
                    (hour, minute, day_of_week, temperature, humidity, light_level, motion_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now.hour, now.minute, now.weekday(), temp, hum, light, 1 if motion else 0))
            return cursor.lastrowid

    def normalize_event_payload(self, payload: dict) -> dict:
        """Chuẩn hóa mọi nguồn event về schema chung chính thức của dự án."""
        return canonicalize_event(payload)

    def insert_event_log(self, payload: dict) -> int:
        """Lưu payload MQTT sự kiện giàu ngữ cảnh vào database để phục vụ train."""
        event = self.normalize_event_payload(payload)
        sensor_id = self.insert_sensor_data(
            event["temperature"],
            event["humidity"],
            event["light_level"],
            bool(event["motion_detected"])
        )

        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO device_states
                    (env_log_id, user_id, dominant_user_id, decision_mode, light_state, fan_state, source, data_source,
                     scenario_type, sequence_phase, override_flag, override_reason, session_id,
                     presence_type, face_count, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sensor_id, event["user_id"], event["dominant_user_id"], event["decision_mode"], event["light_state"], event["fan_state"],
                event["source"], event["data_source"], event["scenario_type"], event["sequence_phase"],
                event["override_flag"], event["override_reason"], event["session_id"],
                event["presence_type"], event["face_count"], event["confidence"]
            ))
            return cursor.lastrowid

    def _enrich_training_rows(self, rows: list[dict]) -> list[dict]:
        """Bổ sung lag features trực tiếp từ dữ liệu DB khi schema gốc chưa lưu."""
        enriched: list[dict] = []
        prev_by_user: dict[str, dict] = {}

        for row in rows:
            user_id = (row.get("user_id") or "unknown").strip() or "unknown"
            prev = prev_by_user.get(user_id)

            temp = float(row.get("temperature", 0.0) or 0.0)
            hum = float(row.get("humidity", 0.0) or 0.0)
            light = float(row.get("light_level", 0.0) or 0.0)
            face_count = int(row.get("face_count", 0) or 0)

            if prev is None:
                last_temp = temp
                last_humidity = hum
                last_light_level = light
                last_light_state = int(row.get("light_state", 0) or 0)
                last_fan_state = int(row.get("fan_state", 0) or 0)
                face_count_raw = face_count
            else:
                last_temp = float(prev["temperature"])
                last_humidity = float(prev["humidity"])
                last_light_level = float(prev["light_level"])
                last_light_state = int(prev.get("light_state", row.get("light_state", 0)) or 0)
                last_fan_state = int(prev.get("fan_state", row.get("fan_state", 0)) or 0)
                face_count_raw = int(prev.get("face_count", face_count) or face_count)

            row["temp_delta"] = abs(temp - last_temp)
            row["humidity_delta"] = abs(hum - last_humidity)
            row["light_delta"] = abs(light - last_light_level)
            enriched.append(row)

            prev_by_user[user_id] = row

        return enriched

    def get_training_data(self, limit: int = 20000):
        # [FIX BUG-DB-02] Dùng danh sách source chuẩn thay vì hardcode thiếu
        sources = getattr(config, 'TRAINING_SOURCES', DEFAULT_TRAINING_SOURCES)
        placeholders = ', '.join(['?'] * len(sources))
        
        with self.get_connection() as conn:
            query = f"""
                SELECT * FROM (
                    SELECT
                        e.timestamp,
                        e.hour, e.minute, e.day_of_week,
                        e.temperature, e.humidity, e.light_level, e.motion_detected,
                        d.user_id,
                        d.dominant_user_id,
                        d.decision_mode,
                        d.light_state,
                        d.fan_state,
                        d.presence_type,
                        d.face_count,
                        d.source,
                        d.data_source,
                        d.scenario_type,
                        d.sequence_phase,
                        d.override_flag,
                        d.override_reason,
                        d.session_id,
                        d.confidence
                    FROM device_states d
                    JOIN environmental_logs e ON d.env_log_id = e.id
                    WHERE d.source IN ({placeholders})
                    ORDER BY e.timestamp DESC
                    LIMIT ?
                ) ORDER BY timestamp ASC
            """
            params = list(sources) + [limit]
            rows = conn.execute(query, params).fetchall()
            return self._enrich_training_rows([dict(r) for r in rows])

    def get_training_sources(self) -> tuple:
        return getattr(config, 'TRAINING_SOURCES', DEFAULT_TRAINING_SOURCES)

    def get_training_data_by_source(self, sources: tuple | list[str], limit: int = 20000):
        if not sources:
            return []
        placeholders = ', '.join(['?'] * len(sources))
        with self.get_connection() as conn:
            query = f"""
                SELECT * FROM (
                    SELECT
                        e.timestamp,
                        e.hour, e.minute, e.day_of_week,
                        e.temperature, e.humidity, e.light_level, e.motion_detected,
                        d.user_id,
                        d.dominant_user_id,
                        d.decision_mode,
                        d.light_state,
                        d.fan_state,
                        d.presence_type,
                        d.face_count,
                        d.source,
                        d.data_source,
                        d.scenario_type,
                        d.sequence_phase,
                        d.override_flag,
                        d.override_reason,
                        d.session_id,
                        d.confidence
                    FROM device_states d
                    JOIN environmental_logs e ON d.env_log_id = e.id
                    WHERE d.source IN ({placeholders})
                    ORDER BY e.timestamp DESC
                    LIMIT ?
                ) ORDER BY timestamp ASC
            """
            params = list(sources) + [limit]
            rows = conn.execute(query, params).fetchall()
            return self._enrich_training_rows([dict(r) for r in rows])


    def count_training_data(self) -> int:
        # [FIX BUG-DB-02] Dùng danh sách source chuẩn
        sources = getattr(config, 'TRAINING_SOURCES', DEFAULT_TRAINING_SOURCES)
        placeholders = ', '.join(['?'] * len(sources))
        
        with self.get_connection() as conn:
            query = f"""
                SELECT COUNT(*) FROM device_states 
                WHERE source IN ({placeholders})
            """
            return conn.execute(query, list(sources)).fetchone()[0]

    def count_training_data_by_source(self) -> dict:
        sources = getattr(config, 'TRAINING_SOURCES', DEFAULT_TRAINING_SOURCES)
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) AS cnt FROM device_states GROUP BY source"
            ).fetchall()
        data = {s: 0 for s in sources}
        for r in rows:
            if r["source"] in data:
                data[r["source"]] = int(r["cnt"])
        return data

    def log_device_state(self, env_log_id: int, user_id: str,
                         light: int, fan: int, source: str = "manual_override",
                         dominant_user_id: str = "unknown", decision_mode: str = "fallback",
                         data_source: str = "real", scenario_type: str = "background_noise",
                         override_flag: int = 0, override_reason: str = "",
                         session_id: str = None, presence_type: str = 'unknown', face_count: int = 0,
                         confidence: float = 0.0):
        if env_log_id is None:
            raise ValueError("env_log_id is required for device state logging")
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO device_states
                    (env_log_id, user_id, dominant_user_id, decision_mode, light_state, fan_state, source, data_source,
                     scenario_type, sequence_phase, override_flag, override_reason, session_id,
                     presence_type, face_count, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (env_log_id, user_id, dominant_user_id, decision_mode, light, fan, source, data_source, scenario_type,
                   scenario_type, override_flag, override_reason, session_id, presence_type, face_count, confidence))


    # ─────────────────────────────────────────────────────
    # Face Profiles & Users
    # ─────────────────────────────────────────────────────
    def upsert_face_profile(self, user_id: str, display_name: str):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO face_profiles (user_id, display_name)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET display_name=excluded.display_name
            """, (user_id, display_name))

    def update_face_seen(self, user_id: str):
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE face_profiles
                SET last_seen=datetime('now','localtime'), visit_count=visit_count+1
                WHERE user_id=?
            """, (user_id,))

    def get_all_users(self):
        with self.get_connection() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM face_profiles ORDER BY CASE WHEN last_seen IS NULL THEN 1 ELSE 0 END, last_seen DESC"
            ).fetchall()]

    def get_user_info(self, user_id: str) -> dict:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM face_profiles WHERE user_id=?", (user_id,)
            ).fetchone()
            return dict(row) if row else {}
            
    def delete_face_profile(self, user_id: str):
        """[FIX BUG-WEB-02] Xóa triệt để người dùng khỏi Database"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM face_profiles WHERE user_id=?", (user_id,))

    def delete_device_state(self, row_id: int) -> int:
        """Xóa một dòng nhật ký thiết bị theo ID và dọn cả log môi trường liên quan nếu không còn tham chiếu."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT env_log_id FROM device_states WHERE id=?", (row_id,)).fetchone()
            if not row:
                return 0

            env_log_id = row["env_log_id"]
            cur = conn.execute("DELETE FROM device_states WHERE id=?", (row_id,))

            if env_log_id is not None:
                remaining = conn.execute("SELECT COUNT(*) AS cnt FROM device_states WHERE env_log_id=?", (env_log_id,)).fetchone()["cnt"]
                if remaining == 0:
                    conn.execute("DELETE FROM environmental_logs WHERE id=?", (env_log_id,))

            return cur.rowcount or 0

    # ─────────────────────────────────────────────────────
    # Predictions & Logs
    # ─────────────────────────────────────────────────────
    def log_prediction(self, env_log_id: int, user_id: str,
                       pred_light: int, pred_fan: int, confidence: float,
                       source: str = "ai", actual_light: int | None = None,
                       actual_fan: int | None = None, override_flag: int = 0,
                       scenario_type: str = "sequence_behavior", override_reason: str = "",
                       feedback_source: str = ""):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO predictions
                    (env_log_id, user_id, pred_light, pred_fan, confidence, executed,
                     actual_light, actual_fan, overridden, override_reason, feedback_source, scenario_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                env_log_id,
                user_id,
                pred_light,
                pred_fan,
                confidence,
                1 if source == "ai" else 0,
                actual_light,
                actual_fan,
                int(override_flag),
                override_reason,
                feedback_source or source,
                scenario_type,
            ))
            if actual_light is not None or actual_fan is not None or override_flag:
                conn.execute("""
                    INSERT INTO prediction_feedback_queue
                        (env_log_id, user_id, source, predicted_light, predicted_fan,
                         actual_light, actual_fan, override_reason, feedback_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    env_log_id,
                    user_id,
                    source,
                    pred_light,
                    pred_fan,
                    actual_light,
                    actual_fan,
                    override_reason,
                    feedback_source or source,
                ))

    def log_prediction_with_feedback(self, env_log_id: int, user_id: str,
                                     pred_light: int, pred_fan: int, confidence: float,
                                     source: str = "ai", actual_light: int | None = None,
                                     actual_fan: int | None = None, override_flag: int = 0,
                                     scenario_type: str = "sequence_behavior", override_reason: str = "",
                                     feedback_source: str = ""):
        return self.log_prediction(
            env_log_id=env_log_id,
            user_id=user_id,
            pred_light=pred_light,
            pred_fan=pred_fan,
            confidence=confidence,
            source=source,
            actual_light=actual_light,
            actual_fan=actual_fan,
            override_flag=override_flag,
            scenario_type=scenario_type,
            override_reason=override_reason,
            feedback_source=feedback_source,
        )

    def log_event_bundle(self, *, env_log_id: int, user_id: str, light: int, fan: int,
                         source: str = "ai", data_source: str = "real",
                         scenario_type: str = "background_noise", override_flag: int = 0,
                         override_reason: str = "", confidence: float = 0.0,
                         session_id: str = None, presence_type: str = "unknown", face_count: int = 0,
                         feedback_source: str = "", decision_evidence: dict | None = None,
                         prediction_id: int | None = None, actual_light: int | None = None,
                         actual_fan: int | None = None, predicted_light: int | None = None,
                         predicted_fan: int | None = None, event_type: str | None = None,
                         actor_type: str | None = None, trigger_source: str | None = None,
                         hysteresis_meta: str | None = None, **_ignored):
        self.log_prediction(
            env_log_id=env_log_id,
            user_id=user_id,
            pred_light=predicted_light if predicted_light is not None else light,
            pred_fan=predicted_fan if predicted_fan is not None else fan,
            confidence=confidence,
            source=source,
            actual_light=actual_light,
            actual_fan=actual_fan,
            override_flag=override_flag,
            scenario_type=scenario_type,
            override_reason=override_reason,
            feedback_source=feedback_source or source,
        )
        self.log_device_state(
            env_log_id=env_log_id,
            user_id=user_id,
            light=light,
            fan=fan,
            source=source,
            data_source=data_source,
            scenario_type=scenario_type,
            override_flag=override_flag,
            override_reason=override_reason,
            session_id=session_id,
            presence_type=presence_type,
            face_count=face_count,
            confidence=confidence,
        )
        if decision_evidence:
            self.insert_decision_evidence(
                prediction_id=prediction_id,
                env_log_id=env_log_id,
                user_id=user_id,
                sensor_json=decision_evidence.get("sensor_json", {}),
                evidence_json=decision_evidence.get("evidence_json", {}),
                frame_url=decision_evidence.get("frame_url", ""),
                frame_path=decision_evidence.get("frame_path", ""),
                reason=decision_evidence.get("reason", ""),
                confidence=confidence,
                override_reason=override_reason,
                active_users=decision_evidence.get("active_users", []),
            )

    def insert_anomaly_event(self, anomaly_type: str, message: str, sensor_json=None, system_json=None,
                             severity: str = 'medium', action_taken: str = 'none', safe_mode: int = 0,
                             user_id: str = 'unknown'):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO anomaly_events (anomaly_type, severity, message, sensor_json, system_json, action_taken, safe_mode, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                anomaly_type,
                severity,
                message,
                json.dumps(sensor_json or {}, ensure_ascii=False),
                json.dumps(system_json or {}, ensure_ascii=False),
                action_taken,
                int(bool(safe_mode)),
                user_id,
            ))

    def get_anomaly_events(self, limit: int = 50):
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM anomaly_events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def insert_decision_evidence(self, *, prediction_id: int | None = None,
                                 env_log_id: int | None = None, user_id: str = 'unknown',
                                 sensor_json=None, evidence_json=None, frame_url: str = '',
                                 frame_path: str = '', reason: str = '', confidence: float = 0.0,
                                 override_reason: str = '', active_users=None) -> int:
        sensor_payload = json.dumps(sensor_json or {}, ensure_ascii=False)
        evidence_payload = json.dumps(evidence_json or {}, ensure_ascii=False)
        active_users_payload = json.dumps(active_users or [], ensure_ascii=False)

        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO decision_evidence
                    (prediction_id, env_log_id, user_id, sensor_json, evidence_json,
                     frame_url, frame_path, reason, confidence, override_reason, active_users)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                env_log_id,
                user_id,
                sensor_payload,
                evidence_payload,
                frame_url,
                frame_path,
                reason,
                confidence,
                override_reason,
                active_users_payload,
            ))
            return cursor.lastrowid

    def get_decision_evidence(self, limit: int = 50):
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM decision_evidence
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_recent_logs(self, n: int = 20):
        with self.get_connection() as conn:
            return [dict(r) for r in conn.execute(f"""
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY env_log_id 
                        ORDER BY CASE source 
                            WHEN 'manual_override' THEN 1 
                            WHEN 'manual'          THEN 2
                            WHEN 'periodic_heartbeat' THEN 3
                            WHEN 'pir_only'        THEN 4
                            WHEN 'pir_motion'      THEN 5
                            WHEN 'pir_timeout'     THEN 6
                            WHEN 'pir_timeout_total_off' THEN 7
                            WHEN 'shadow_pir'      THEN 8
                            WHEN 'shadow'          THEN 9
                            WHEN 'ai_session_hold' THEN 10
                            ELSE 99 
                        END, id DESC
                    ) as rn 
                    FROM device_states
                )
                SELECT 
                    e.timestamp, e.temperature, e.humidity, e.light_level, e.motion_detected,
                    COALESCE(r.user_id, 'unknown') AS user_id,
                    COALESCE(r.light_state, 0) AS light_state,
                    COALESCE(r.fan_state, 0) AS fan_state,
                    COALESCE(r.source, 'no_action') AS source,
                    COALESCE(r.presence_type, 'unknown') AS presence_type,
                    COALESCE(r.face_count, 0) AS face_count
                FROM environmental_logs e
                LEFT JOIN ranked r ON r.env_log_id = e.id AND r.rn = 1
                ORDER BY e.timestamp DESC
                LIMIT ?
            """, (n,)).fetchall()]