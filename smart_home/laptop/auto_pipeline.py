import csv
import random
import time
import os
import argparse
from datetime import datetime, timedelta

# Import config và Database từ hệ thống của bạn
import config
from database import Database
from modules.priority_policy import (
    DECISION_MODE_FALLBACK,
    DECISION_MODE_MANUAL_OVERRIDE,
    DECISION_MODE_MULTI_USER,
    DECISION_MODE_SINGLE_USER,
    map_scenario_type,
)

VALID_DECISION_MODES = {
    DECISION_MODE_SINGLE_USER,
    DECISION_MODE_MULTI_USER,
    DECISION_MODE_MANUAL_OVERRIDE,
    DECISION_MODE_FALLBACK,
}


def normalize_row_schema(row):
    normalized = dict(row)
    normalized['user_id'] = (normalized.get('user_id') or 'unknown').strip().lower() or 'unknown'
    normalized['dominant_user_id'] = (normalized.get('dominant_user_id') or normalized['user_id'] or 'unknown').strip().lower() or 'unknown'
    normalized['presence_type'] = (normalized.get('presence_type') or 'unknown').strip().lower() or 'unknown'
    normalized['source'] = (normalized.get('source') or 'real').strip().lower() or 'real'
    normalized['data_source'] = (normalized.get('data_source') or normalized['source'] or 'real').strip().lower() or 'real'
    normalized['sequence_phase'] = (normalized.get('sequence_phase') or 'sequence_idle').strip().lower() or 'sequence_idle'
    normalized['override_reason'] = (normalized.get('override_reason') or '').strip().lower()
    normalized['session_id'] = (normalized.get('session_id') or '').strip()
    decision_mode = (normalized.get('decision_mode') or DECISION_MODE_SINGLE_USER).strip().lower()
    normalized['decision_mode'] = decision_mode if decision_mode in VALID_DECISION_MODES else DECISION_MODE_FALLBACK
    normalized['scenario_type'] = map_scenario_type(
        decision_mode=normalized['decision_mode'],
        source=normalized['source'],
        target_user=normalized['dominant_user_id'],
        override_flag=int(normalized.get('override_flag', 0) or 0),
        review_reason=normalized['override_reason'],
        event_type=normalized['sequence_phase'],
        presence_type=normalized['presence_type'],
    )
    return normalized

def interpolate_in_memory(raw_data, interval_minutes=2):
    """Bước 1: Lấp khoảng trống thời gian (Nội suy)"""
    interpolated_data = []

    for i in range(len(raw_data) - 1):
        current_row = raw_data[i]
        next_row = raw_data[i + 1]

        curr_time = datetime.strptime(current_row['timestamp'], "%Y-%m-%d %H:%M:%S")
        next_time = datetime.strptime(next_row['timestamp'], "%Y-%m-%d %H:%M:%S")

        interpolated_data.append(current_row)

        step_time = curr_time + timedelta(minutes=interval_minutes)
        while step_time < next_time:
            heartbeat_row = current_row.copy()
            heartbeat_row['timestamp'] = step_time.strftime("%Y-%m-%d %H:%M:%S")
            heartbeat_row['hour'] = step_time.hour
            heartbeat_row['minute'] = step_time.minute
            heartbeat_row['day_of_week'] = step_time.weekday()
            heartbeat_row['override_flag'] = 0
            heartbeat_row['source'] = 'real'
            heartbeat_row['data_source'] = 'real'
            heartbeat_row['decision_mode'] = DECISION_MODE_FALLBACK
            heartbeat_row['scenario_type'] = map_scenario_type(
                decision_mode=heartbeat_row.get('decision_mode', DECISION_MODE_FALLBACK),
                source=heartbeat_row.get('source', 'real'),
                target_user=heartbeat_row.get('dominant_user_id', heartbeat_row.get('user_id', 'unknown')),
                override_flag=0,
                review_reason=heartbeat_row.get('override_reason', ''),
                event_type=heartbeat_row.get('sequence_phase', 'sequence_idle'),
                presence_type=heartbeat_row.get('presence_type', 'unknown'),
            )
            heartbeat_row['sequence_phase'] = heartbeat_row.get('sequence_phase', 'sequence_idle')
            interpolated_data.append(heartbeat_row)
            step_time += timedelta(minutes=interval_minutes)

    if raw_data:
        interpolated_data.append(raw_data[-1])

    print(f"✔️ Bước 1: Đã nội suy từ {len(raw_data)} mốc lên {len(interpolated_data)} mẫu.")
    return interpolated_data

def augment_data_in_memory(base_data, num_days=30):
    """Bước 2: Nhân bản và thêm nhiễu môi trường/thời gian"""
    expanded_data = []
    print(f"⏳ Bước 2: Bắt đầu nhân bản {num_days} ngày...")

    def clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def gaussian_noise(std_dev):
        return random.gauss(0.0, std_dev)

    scenario_bias = {
        'sequence_behavior': 0.90,
        'context_exception': 1.08,
        'manual_override': 1.18,
        'profile_personalization': 1.00,
        'background_noise': 1.08,
    }

    allowed_decision_modes = VALID_DECISION_MODES

    for day_offset in range(num_days):
        daily_time_shift = int(round(random.gauss(0, 2)))
        daily_temp_bias = random.gauss(0, 0.45)
        daily_hum_bias = random.gauss(0, 1.6)
        daily_light_scale = clamp(random.gauss(1.0, 0.04), 0.90, 1.10)
        daily_light_bias = random.gauss(0, 1.4)
        fan_warmup_bias = random.gauss(-0.08, 0.10)

        daily_rows = []
        prev_time = None
        for idx, row in enumerate(base_data):
            new_row = row.copy()

            orig_time = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
            new_time = orig_time + timedelta(days=day_offset, minutes=daily_time_shift)

            row_time_jitter = int(round(random.gauss(0, 0.4)))
            if prev_time is not None:
                min_gap = max(1, int((orig_time - prev_time).total_seconds() // 60))
                row_time_jitter = clamp(row_time_jitter, -min(1, min_gap // 2), min(1, min_gap // 2))
            new_time += timedelta(minutes=row_time_jitter)
            prev_time = orig_time

            new_row['timestamp'] = new_time.strftime("%Y-%m-%d %H:%M:%S")
            new_row['hour'] = new_time.hour
            new_row['minute'] = new_time.minute
            new_row['day_of_week'] = new_time.weekday()

            base_temp = float(row['temperature'])
            base_hum = float(row['humidity'])
            base_light = float(row['light_level'])

            temp_time_factor = 0.22 * gaussian_noise(0.40)
            hum_time_factor = 0.55 * gaussian_noise(0.75)
            light_time_factor = clamp(random.gauss(1.0, 0.02), 0.94, 1.06)

            new_temp = base_temp + daily_temp_bias + temp_time_factor + gaussian_noise(0.10)
            new_hum = base_hum + daily_hum_bias + hum_time_factor + gaussian_noise(0.40)
            new_light = (base_light * daily_light_scale * light_time_factor) + daily_light_bias + gaussian_noise(1.5)

            new_row['temperature'] = round(clamp(new_temp, 15.0, 45.0), 1)
            new_row['humidity'] = round(clamp(new_hum, 20.0, 95.0), 1)
            new_row['light_level'] = round(clamp(new_light, 0.0, 1000.0), 1)

            motion_base = int(row.get('motion_detected', 0))
            if random.random() < 0.02:
                new_row['motion_detected'] = 1 - motion_base
            else:
                new_row['motion_detected'] = motion_base

            light_state = int(row.get('light_state', 0))
            fan_state = int(row.get('fan_state', 0))
            override_flag = int(row.get('override_flag', 0))
            scenario_type = row.get('scenario_type', 'background_noise')
            scenario_boost = scenario_bias.get(scenario_type, 1.0)

            if scenario_type == 'background_noise':
                if new_row['light_level'] > 150 or new_row['hour'] >= 21:
                    new_row['light_state'] = 0
                else:
                    new_row['light_state'] = light_state if random.random() < 0.35 else 0
                if new_row['temperature'] < 28.2:
                    new_row['fan_state'] = 0
                else:
                    new_row['fan_state'] = fan_state if random.random() < 0.25 else 0
            elif scenario_type == 'sequence_behavior':
                if new_row['temperature'] >= 30.0 or new_row['hour'] in (17, 18, 19):
                    new_row['fan_state'] = 1 if random.random() < (0.70 * scenario_boost) else fan_state
                elif new_row['temperature'] <= 27.8 and new_row['hour'] >= 21:
                    new_row['fan_state'] = 0
                else:
                    new_row['fan_state'] = fan_state if random.random() < 0.62 else 0
                if new_row['light_level'] < 170 or new_row['motion_detected'] == 1:
                    new_row['light_state'] = 1 if random.random() < (0.88 * scenario_boost) else light_state
                else:
                    new_row['light_state'] = 0 if random.random() < 0.52 else light_state
            elif scenario_type == 'context_exception':
                if new_row['temperature'] >= 29.8 and new_row['humidity'] <= 72.5:
                    new_row['fan_state'] = 1 if random.random() < 0.92 else fan_state
                elif new_row['temperature'] <= 27.2 and new_row['hour'] >= 22:
                    new_row['fan_state'] = 0
                else:
                    new_row['fan_state'] = fan_state if random.random() < 0.52 else 0
                if new_row['light_level'] < 100:
                    new_row['light_state'] = 1 if random.random() < 0.92 else light_state
                elif new_row['hour'] >= 22:
                    new_row['light_state'] = 0
            elif scenario_type == 'profile_personalization':
                if 'user_b' in str(new_row.get('user_id', '')):
                    new_row['fan_state'] = 0 if new_row['temperature'] < 29.0 else 1
                elif 'user_a' in str(new_row.get('user_id', '')):
                    new_row['fan_state'] = 1 if new_row['temperature'] >= 29.8 else 0
                if 'user_a' in str(new_row.get('user_id', '')):
                    new_row['light_state'] = 1 if new_row['light_level'] < 180 else light_state
                else:
                    new_row['light_state'] = 0 if new_row['light_level'] > 240 else light_state
            elif scenario_type == 'manual_override':
                if new_row['override_flag'] == 1:
                    new_row['fan_state'] = fan_state
                    new_row['light_state'] = light_state
                else:
                    if new_row['temperature'] >= 29.2:
                        new_row['fan_state'] = 1
                    elif new_row['temperature'] <= 27.4 and new_row['hour'] >= 21:
                        new_row['fan_state'] = 0
                    else:
                        new_row['fan_state'] = fan_state if random.random() < 0.45 else int(new_row['temperature'] >= 29.0)
                    if new_row['light_level'] < 150:
                        new_row['light_state'] = 1
                    elif new_row['light_level'] > 260:
                        new_row['light_state'] = 0

            if override_flag == 1 and random.random() < 0.08:
                new_row['override_reason'] = random.choice([
                    row.get('override_reason', 'manual_override'),
                    'manual_adjustment',
                    'temporary_exception',
                    'comfort_preference'
                ])

            if random.random() < 0.06:
                new_row['presence_type'] = random.choice([
                    row.get('presence_type', 'unknown'),
                    'unknown',
                    'guest',
                    'single_known',
                    'empty'
                ])

            if new_row['session_id']:
                new_row['session_id'] = f"{new_row['session_id']}_day{day_offset}"

            new_row['source'] = row.get('source', 'real')
            new_row['data_source'] = row.get('data_source', 'real')
            new_row['decision_mode'] = row.get('decision_mode', DECISION_MODE_SINGLE_USER)
            if new_row['decision_mode'] not in allowed_decision_modes:
                new_row['decision_mode'] = DECISION_MODE_FALLBACK
            new_row['scenario_type'] = map_scenario_type(
                decision_mode=new_row['decision_mode'],
                source=new_row['source'],
                target_user=new_row.get('dominant_user_id', new_row.get('user_id', 'unknown')),
                override_flag=override_flag,
                review_reason=new_row.get('override_reason', ''),
                event_type=row.get('sequence_phase', 'sequence_idle'),
                presence_type=new_row.get('presence_type', 'unknown'),
            )
            new_row['sequence_phase'] = row.get('sequence_phase', 'sequence_idle')

            # cân bằng nhẹ fan giữa ngày bằng một chút bias theo ngày, nhưng giữ real-like.
            if random.random() < 0.02:
                fan_state_value = int(new_row.get('fan_state', 0))
                if fan_warmup_bias > 0:
                    fan_state_value = 1
                elif fan_warmup_bias < -0.05:
                    fan_state_value = 0
                new_row['fan_state'] = int(clamp(fan_state_value, 0, 1))

            daily_rows.append(new_row)

        daily_rows.sort(key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"))
        expanded_data.extend(daily_rows)

    print(f"✔️ Bước 2: Hoàn tất nhân bản thành {len(expanded_data)} mẫu.")
    return expanded_data

def bulk_insert_to_sqlite(data_list, db_instance):
    """Bước 3: Bơm thẳng dữ liệu từ RAM vào SQLite"""
    total_rows = len(data_list)
    if total_rows == 0:
        return

    print(f"🚀 Bước 3: Đang nạp {total_rows} dòng vào Database...")
    start_time = time.time()

    with db_instance.get_connection() as conn:
        try:
            conn.execute("BEGIN TRANSACTION;")

            for row in data_list:
                cursor = conn.execute("""
                    INSERT INTO environmental_logs
                    (timestamp, hour, minute, day_of_week, temperature, humidity, light_level, motion_detected)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['timestamp'], int(row['hour']), int(row['minute']), int(row['day_of_week']),
                    float(row['temperature']), float(row['humidity']), float(row['light_level']), int(row['motion_detected'])
                ))
                
                env_log_id = cursor.lastrowid

                presence = row.get('presence_type', 'unknown')
                face_count = int(row.get('face_count', 0))
                seq_phase = row.get('sequence_phase', 'sequence_idle')
                override = int(row.get('override_flag', 0))
                conf = float(row.get('confidence', 0.0))
                source = row.get('source', 'real')
                data_source = row.get('data_source', 'real')
                if source not in ('real', 'manual_override', 'manual'):
                    source = 'real'
                if data_source not in ('real', 'manual_override', 'manual'):
                    data_source = 'real'
                decision_mode = row.get('decision_mode', DECISION_MODE_SINGLE_USER)
                if decision_mode not in VALID_DECISION_MODES:
                    decision_mode = DECISION_MODE_FALLBACK
                scenario_type = map_scenario_type(
                    decision_mode=decision_mode,
                    source=source,
                    target_user=row.get('dominant_user_id', row['user_id']),
                    override_flag=override,
                    review_reason=row.get('override_reason', ''),
                    event_type=seq_phase,
                    presence_type=presence,
                )

                conn.execute("""
                    INSERT INTO device_states
                    (timestamp, env_log_id, user_id, light_state, fan_state, source, data_source,
                     scenario_type, sequence_phase, override_flag, override_reason, session_id,
                     presence_type, face_count, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['timestamp'], env_log_id, row['user_id'], int(row['light_state']), int(row['fan_state']),
                    source, data_source, scenario_type, seq_phase,
                    override, row.get('override_reason', ''), row.get('session_id', ''),
                    presence, face_count, conf
                ))

 
            conn.commit()
            elapsed_time = time.time() - start_time
            print(f"✅ XONG! Đã nạp thành công {total_rows} mẫu vào '{config.DATABASE_PATH}'")

        except Exception as e:
            conn.rollback()
            print(f"❌ LỖI DATABASE: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Toàn Diện: Đọc CSV -> Nội suy -> Thêm Nhiễu -> Nạp DB")
    parser.add_argument("--input", type=str, default="custom_scenario.csv", help="File CSV kịch bản mốc gốc")
    parser.add_argument("--days", type=int, default=30, help="Số ngày muốn nhân bản ra")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ LỖI: Không tìm thấy file gốc tại '{args.input}'")
        exit(1)
        
    print(f"--- KHỞI ĐỘNG PIPELINE DỮ LIỆU ---")
    
    # Khởi tạo DB để nó tự động tạo bảng nếu thiếu
    db_instance = Database()

    # 1. Đọc file CSV mốc gốc
    with open(args.input, 'r', encoding='utf-8') as fin:
        raw_data = list(csv.DictReader(fin))
        
    if not raw_data:
        print("❌ LỖI: File CSV đầu vào trống!")
        exit(1)

    # 2. Chạy lần lượt các bước xử lý trên RAM
    interpolated_data = interpolate_in_memory(raw_data, interval_minutes=2)
    augmented_data = augment_data_in_memory(interpolated_data, num_days=args.days)
    
    # 3. Đổ thẳng vào DB
    bulk_insert_to_sqlite(augmented_data, db_instance)