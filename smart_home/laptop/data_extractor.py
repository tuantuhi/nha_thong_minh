
import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

try:
    from database import Database
except ImportError:
    pass # Nếu chạy thông qua main.py hoặc web_server.py đã có sẵn module

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DataExtractor")

# Nguồn dùng để nhận diện feedback loop trong dữ liệu thực tế
AI_SOURCES = ['auto', 'ai']
MANUAL_SOURCES = ['manual_override', 'manual']

class SmartHomeDataExtractor:
    def __init__(self, db_path=config.DATABASE_PATH, output_csv="data/training_data.csv"):
        self.db_path    = db_path
        self.output_csv = output_csv
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

    def extract_and_process(self, lookahead_minutes=5, correction_weight=3.0) -> pd.DataFrame:
        """
        Khai thác dữ liệu, gán trọng số và loại bỏ các dự đoán sai của AI.
        """
        logger.info("[Extractor] Bắt đầu trích xuất dữ liệu từ Database...")
        
        try:
            db = Database(self.db_path)
            # Lấy toàn bộ dữ liệu huấn luyện đã được chuẩn hóa bằng danh sách TRAINING_SOURCES của Database
            raw_data = db.get_training_data(limit=50000) 
        except Exception as e:
            logger.error(f"[Extractor] Lỗi khi kết nối lấy dữ liệu từ DB: {e}")
            return None

        if not raw_data:
            logger.warning("[Extractor] Không có dữ liệu huấn luyện trong DB.")
            return None

        df = pd.DataFrame(raw_data)
        
        # Ép kiểu dữ liệu thời gian an toàn
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Khởi tạo cột trọng số và cờ xóa
        df['sample_weight'] = 1.0
        df['drop_flag'] = False
        df['override_flag'] = df.get('override_flag', 0).fillna(0).astype(int)
        df['scenario_type'] = df.get('scenario_type', 'background_noise').fillna('background_noise')
        df['face_count'] = df.get('face_count', 0).fillna(0).astype(int)
        df['presence_type'] = df.get('presence_type', 'unknown').fillna('unknown')
        df['session_id'] = df.get('session_id', '').fillna('')
        df['confidence'] = df.get('confidence', 0.0).fillna(0.0).astype(float)
        df['dominant_user_id'] = df.get('dominant_user_id', df.get('user_id', 'unknown')).fillna('unknown')

        correction_count = 0
        ai_mistake_count = 0
        manual_override_count = 0
        sequence_behavior_count = 0
        context_exception_count = 0

        logger.info(f"[Extractor] Đang phân tích Feedback Loop (Nhìn trước {lookahead_minutes} phút)...")
        
        total_rows = len(df)
        
        # Ưu tiên các lệnh/ghi đè thật thay vì lệnh AI để tránh làm bẩn dataset train.
        df['source'] = df['source'].fillna('').astype(str).str.strip().str.lower()
        df['user_id'] = df['user_id'].fillna('unknown').astype(str).str.strip().str.lower()
        df['scenario_type'] = df['scenario_type'].fillna('background_noise').astype(str).str.strip().str.lower()
        df['presence_type'] = df['presence_type'].fillna('unknown').astype(str).str.strip().str.lower()
        
        # Thuật toán tìm Feedback Loop
        for i in range(total_rows):
            current_source = df.loc[i, 'source']
            current_scenario = str(df.loc[i, 'scenario_type']).strip().lower()
            
            if current_scenario == 'manual_override':
                manual_override_count += 1
            elif current_scenario == 'sequence_behavior':
                sequence_behavior_count += 1
            elif current_scenario == 'context_exception':
                context_exception_count += 1
            
            # Nếu phát hiện AI vừa ra lệnh
            if current_source in AI_SOURCES:
                ai_time  = df.loc[i, 'timestamp']
                ai_user  = df.loc[i, 'user_id']
                ai_light = df.loc[i, 'light_state']
                ai_fan   = df.loc[i, 'fan_state']

                # Look-ahead (Nhìn về tương lai)
                for j in range(i + 1, total_rows):
                    future_time = df.loc[j, 'timestamp']
                    
                    # Nếu vượt quá khung thời gian (5 phút), dừng tìm kiếm cho lệnh AI này
                    if (future_time - ai_time).total_seconds() > lookahead_minutes * 60:
                        break

                    future_source = df.loc[j, 'source']
                    future_user   = df.loc[j, 'user_id']

                    # Nếu người dùng can thiệp tay trong thời gian cho phép
                    if future_source in MANUAL_SOURCES and future_user == ai_user:
                        future_light = df.loc[j, 'light_state']
                        future_fan   = df.loc[j, 'fan_state']

                        # Chỉ coi là feedback thật khi người dùng đảo ngược ít nhất một thiết bị.
                        if future_light != ai_light or future_fan != ai_fan:
                            # 1. Xóa bản ghi sai của AI
                            df.loc[i, 'drop_flag'] = True
                            # 2. Tăng trọng số cho hành động sửa sai của người dùng
                            df.loc[j, 'sample_weight'] = float(correction_weight)
                            df.loc[j, 'override_flag'] = 1
                            df.loc[j, 'scenario_type'] = 'manual_override'
                            df.loc[j, 'source'] = 'manual_override'
                            
                            correction_count += 1
                            ai_mistake_count += 1
                            
                            logger.debug(
                                f"Feedback Loop: AI sai lúc {ai_time.strftime('%H:%M:%S')}. "
                                f"User sửa lúc {future_time.strftime('%H:%M:%S')}."
                            )
                            break # Đã tìm thấy điểm sửa sai, chuyển sang xét bản ghi AI tiếp theo

        # Làm sạch dữ liệu (Drop các row bị đánh cờ)
        clean_df = df[df['drop_flag'] == False].copy()
        clean_df.drop(columns=['drop_flag'], inplace=True)
        clean_df.reset_index(drop=True, inplace=True)

        # Tăng tính nhất quán dữ liệu cho training.
        clean_df['dominant_user_id'] = clean_df.get('dominant_user_id', clean_df['user_id']).fillna(clean_df['user_id'])
        clean_df['light_state'] = clean_df['light_state'].astype(int)
        clean_df['fan_state'] = clean_df['fan_state'].astype(int)
        clean_df['override_flag'] = clean_df['override_flag'].astype(int)
        clean_df['face_count'] = clean_df['face_count'].astype(int)
        clean_df['confidence'] = clean_df['confidence'].astype(float)
        clean_df['temp_delta'] = (clean_df['temperature'] - clean_df.groupby('user_id')['temperature'].shift(1).fillna(clean_df['temperature'])).abs()
        clean_df['humidity_delta'] = (clean_df['humidity'] - clean_df.groupby('user_id')['humidity'].shift(1).fillna(clean_df['humidity'])).abs()
        clean_df['light_delta'] = (clean_df['light_level'] - clean_df.groupby('user_id')['light_level'].shift(1).fillna(clean_df['light_level'])).abs()

        # Lưu ra CSV dự phòng/phân tích
        try:
            clean_df.to_csv(self.output_csv, index=False)
        except Exception as e:
            logger.warning(f"[Extractor] Lỗi khi lưu file CSV: {e}")
        
        logger.info("=" * 50)
        logger.info(" BÁO CÁO TRÍCH XUẤT VÀ LÀM SẠCH DỮ LIỆU:")
        logger.info(f" - Tổng số bản ghi thô: {total_rows}")
        logger.info(f" - Bắt được thao tác sửa sai: {correction_count} lần (Gắn trọng số x{correction_weight})")
        logger.info(f" - Đã loại bỏ số lần AI quyết định sai: {ai_mistake_count} bản ghi")
        logger.info(f" - Manual override: {manual_override_count} | Sequence: {sequence_behavior_count} | Context exception: {context_exception_count}")
        logger.info(f" - Dữ liệu Vàng (Golden Data) xuất ra: {len(clean_df)} bản ghi")
        logger.info("=" * 50)

        return clean_df

    def build_sequence_windows(self, df: pd.DataFrame, time_steps: int = None, max_gap_seconds: int = None):
        """Tạo cửa sổ chuỗi LSTM trực tiếp từ dữ liệu đã làm sạch."""
        if df is None or df.empty:
            return []

        time_steps = time_steps or int(getattr(config, "LSTM_TIME_STEPS", 6))
        max_gap_seconds = max_gap_seconds or int(getattr(config, "MAX_SEQUENCE_GAP_SEC", 180))
        seq_cols = [
            "hour", "minute", "day_of_week", "temperature", "humidity",
            "light_level", "motion_detected", "temp_delta", "humidity_delta",
            "light_delta", "motion_streak", "time_bucket", "is_weekend",
            "face_count", "override_flag", "confidence",
            "dominant_user_id"
        ]

        work = df.copy()
        work['timestamp'] = pd.to_datetime(work['timestamp'])
        work = work.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        windows = []
        for user_id, group in work.groupby('user_id'):
            group = group.reset_index(drop=True)
            if len(group) <= time_steps:
                continue

            for i in range(len(group) - time_steps):
                chunk = group.iloc[i:i + time_steps + 1]
                gaps = chunk['timestamp'].diff().dt.total_seconds().dropna().tolist()
                if gaps and max(gaps) > max_gap_seconds:
                    continue

                windows.append({
                    "user_id": user_id,
                    "X": chunk.iloc[:time_steps][seq_cols].to_dict('records'),
                    "y_light": int(chunk.iloc[time_steps]['light_state']),
                    "y_fan": int(chunk.iloc[time_steps]['fan_state']),
                    "scenario_type": str(chunk.iloc[time_steps].get('scenario_type', 'background_noise')),
                    "override_flag": int(chunk.iloc[time_steps].get('override_flag', 0)),
                    "confidence": float(chunk.iloc[time_steps].get('confidence', 0.0)),
                    "timestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in chunk['timestamp'].tolist()],
                })
        return windows

if __name__ == "__main__":
    extractor = SmartHomeDataExtractor()
    df = extractor.extract_and_process()
    if df is not None:
        print(df.head())