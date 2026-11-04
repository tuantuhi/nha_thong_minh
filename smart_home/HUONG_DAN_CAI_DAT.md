# 🏠 SMART HOME AI
## HƯỚNG DẪN CÀI ĐẶT CHI TIẾT (PHIÊN BẢN LOCAL MQTT)
**Linh kiện • Hệ thống MQTT • Python AI**

### 1. Kiến Trúc Hệ Thống

┌────────────────────────┐                                        ┌──────────────────────┐
│     PHẦN CỨNG (ESP32)  │                                        │   MÁY CHỦ (LAPTOP)   │
│                        │                                        │                      │
│   ┌─────────────┐      │                                        │   ┌───────────────┐  │
│   │   Cảm biến  │      │                                        │   │   Camera/Face │  │
│   │ (DHT, PIR..)├──┐   │                                        │   │   Recognition │  │
│   └─────────────┘  │   │                                        │   └───────┬───────┘  │
│                    ▼   │        WiFi Local Network              │           ▼          │
│             ┌────────┐ │       (Topic: home/room1/...)          │   ┌───────────────┐  │
│             │  ESP32 ├─┼─────────────── MQTT ────────────────────┼─►│     Python    │  │
│             └────┬───┘ │                                        │   │   (main.py)   │  │
│                  │     │                                        │   └────┬─────┬───┘  │
│   ┌─────────────┐│     │          ┌──────────────────┐          │        │     │      │
│   │   Thiết bị  │◄─┘   │          │ Mosquitto Broker │          │        │     │      │
│   │ (Relay, LED)│      │          │     (Port 1883)  │          │  ┌─────▼─┐ ┌─▼─────┐│
│   └─────────────┘      │          └──────────────────┘          │  │ SQLite│ │ AI    ││
└────────────────────────┘                                        │  │   DB  │ │ Model ││
                                                                  │  └───────┘ └───────┘│
                                                                  └──────────────────────┘

**Luồng hoạt động cốt lõi:**
1. **Cảm biến:** PIR quét chống nhiễu liên tục (Debounce), DHT22/BH1750 cập nhật định kỳ, gửi luồng Time-series từ ESP32 lên Mosquitto Broker.
2. **Giao vận:** `mqtt_handler.py` nhận tín hiệu, đẩy vào bộ đệm, xử lý qua Command Queue và ghi nhận vào Database (SQLite).
3. **Thị giác máy tính:** `face_recognition.py` liên tục quét camera để định danh người dùng (User ID) duy trì ngữ cảnh, lưu trữ an toàn bằng định dạng JSON.
4. **Trí tuệ nhân tạo:** `ai_model.py` sử dụng mạng Stacked LSTM trích xuất đặc trưng chuỗi thời gian kết hợp XGBoost để ra quyết định đóng cắt thiết bị. Tích hợp thuật toán Hysteresis và Threshold độc lập.
5. **Học tập liên tục:** Nếu người dùng can thiệp bằng lệnh thủ công (CLI), hệ thống sẽ lưu cấu hình vào Database và tự động Re-train XGBoost theo chu kỳ để "nhập tâm" thói quen mới.

---

### 🛠️ BƯỚC 1: Cài đặt Mosquitto MQTT Broker (Trên Laptop)
Vì hệ thống vận hành hoàn toàn Local (Edge AI), Laptop của bạn sẽ đóng vai trò là "Trái tim" điều phối mạng (Broker).
1. Tải Eclipse Mosquitto tại: https://mosquitto.org/download/
2. Cài đặt bình thường (Next -> Next).
3. **Mở Port tường lửa (Rất quan trọng):**
   * Vào `Windows Defender Firewall` -> `Advanced Settings`.
   * `Inbound Rules` -> `New Rule` -> `Port` -> `TCP` -> `Specific local ports: 1883`.
   * `Allow the connection` -> Name: `MQTT 1883`.
4. **Lấy địa chỉ IP của Laptop:**
   * Mở Command Prompt (cmd) gõ `ipconfig`.
   * Ghi lại dòng IPv4 Address (Ví dụ: `192.168.1.15`). Đây sẽ là `mqtt_server`.

---

### 🔌 BƯỚC 2: Cài đặt và Nạp Code cho ESP32
**2.1 Sơ đồ đấu nối phần cứng**
Giữ nguyên sơ đồ vật lý, đảm bảo module PIR và LED RGB được nối đúng chân GPIO.

| ESP32 GPIO | Thiết bị | Ghi chú |
| :--- | :--- | :--- |
| **GPIO 4** | DHT22 DATA | Kèm điện trở pullup 10kΩ lên 3.3V |
| **GPIO 21** | BH1750 SDA | Bus I2C Data |
| **GPIO 22** | BH1750 SCL | Bus I2C Clock |
| **GPIO 32** | PIR (OUT) | Cảm biến chuyển động |
| **GPIO 26** | IN1 Relay (Đèn) | ACTIVE LOW |
| **GPIO 27** | IN2 Relay (Quạt) | ACTIVE LOW |
| **GPIO 13** | LED RGB (R) | Chân Đỏ, qua trở 220Ω |
| **GPIO 25** | LED RGB (G) | Chân Xanh lá, qua trở 220Ω |
| **GPIO 14** | LED RGB (B) | Chân Xanh dương, qua trở 220Ω |
| **3.3V** | VCC DHT22, BH1750| Nguồn Logic |
| **5V / Vin**| VCC Relay, PIR | Nguồn Công suất |
| **GND** | GND | Nối chung toàn bộ hệ thống |

**2.2 Cài đặt thư viện Arduino IDE**
Cài đặt các thư viện sau qua Library Manager:
* `PubSubClient` (by Nick O'Leary)
* `ArduinoJson` (by Benoit Blanchon - Hỗ trợ cả bản v6 & v7)
* `DHT sensor library` (by Adafruit)
* `BH1750` (by Christopher Laws)

**2.3 Cấu hình Firmware**
Mở file `main.cpp`, cấu hình mạng WiFi nội bộ:
```cpp
const char* ssid        = "Ten_WiFi_Cua_Ban"; 
const char* password    = "Mat_Khau_WiFi";   
const char* mqtt_server = "192.168.1.15"; // ĐIỀN ĐỊA CHỈ IP LAPTOP (Bước 1)

Nạp code xuống ESP32. Nếu mở Serial Monitor (115200 baud) thấy báo [MQTT] Dang ket noi toi Broker... THANH CONG! là đã hoàn tất.

💻 BƯỚC 3: Cài đặt Môi trường Python (AI & Điều phối)
3.1 Cài đặt thư viện
Mở Terminal/CMD tại thư mục laptop/, tạo môi trường ảo và cài đặt:

Bash
# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate          # Trên Windows
source venv/bin/activate       # Trên Linux/macOS

# Cài đặt toàn bộ thư viện cần thiết
pip install paho-mqtt>=2.0.0 xgboost>=2.0.3 tensorflow>=2.16.0 scikit-learn>=1.4.0 numpy>=1.26.0 joblib>=1.3.0 opencv-python>=4.9.0 deepface>=0.0.93 scipy>=1.11.0 python-dotenv
3.2 Cấu hình Môi trường Bảo mật (.env)
Để bảo mật thông tin IP và kết nối, hệ thống sử dụng file .env. Hãy tạo một file tên là .env (lưu ý có dấu chấm ở đầu) ngang hàng với file main.py và nhập:

Đoạn mã
MQTT_BROKER=192.168.1.15   # Điền IP LAN của laptop (Khớp với C++)
MQTT_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=
🧠 BƯỚC 4: Huấn luyện AI & Vận hành Hệ thống
4.1 Tạo dữ liệu giả lập (Lần đầu tiên)
Mạng LSTM-XGBoost cần dữ liệu lịch sử để hiểu thói quen. Chạy script sinh dữ liệu chuẩn hóa theo môi trường mùa hè nóng bức (đã bao gồm nhãn Session ID chống nhiễu chuỗi thời gian):

Bash
python modules/training_data_generator.py --n 5000
Lệnh này sẽ tạo ra file smart_home.db chứa 5000 mẫu dữ liệu Time-series.

4.2 Khởi động hệ thống chính
Chạy file điều phối trung tâm:

Bash
python main.py
Quy trình vận hành khuyên dùng:

   Khi giao diện CLI hiện lên, chọn [5] Đăng ký khuôn mặt để hệ thống định danh bạn (Ví dụ: user_a).

   Chọn [4] Train AI ngay bây giờ để hệ thống huấn luyện bộ trọng số (Weights) từ 5000 mẫu vừa sinh.

   Chọn [8] Bật/Tắt Shadow Mode (Chuyển sang AI MODE để AI giành quyền điều khiển thiết bị thật).

   Bạn có thể chọn [2] Điều khiển thủ công. Bất cứ lúc nào bạn can thiệp tay, hệ thống sẽ lưu thao tác này dưới dạng "Nhãn" (Label) mới để AI tiến hành Retrain và thích nghi ở chu kỳ tiếp theo.

📁 CẤU TRÚC THƯ MỤC HỆ THỐNG

smart_home/
├── esp32/
│   └── main.cpp                    ← Firmware ESP32 (FreeRTOS, MQTT)
├── laptop/
│   ├── .env                        ← File bảo mật cấu hình mạng (NÊN ẨN KHI LÊN GITHUB)
│   ├── main.py                     ← Điều phối hệ thống (CLI, Dual Mode, Command Queue)
│   ├── config.py                   ← Định nghĩa hằng số hệ thống (Thresholds)
│   ├── database.py                 ← Quản lý SQLite DB (System Meta, Logs)
│   ├── ai_model.py                 ← Mô hình Hybrid (Stacked LSTM + XGBoost)
│   ├── modules/
│   │   ├── mqtt_handler.py         ← Giao tiếp Mosquitto Broker (Tương thích Paho v2.0+)
│   │   ├── face_recognition_module.py ← Xử lý OpenCV & DeepFace (Vectorized Distance)
│   │   └── training_data_generator.py ← Bộ tạo dữ liệu chuỗi thời gian mô phỏng
│   ├── data/                       
│   │   └── smart_home.db           ← Cơ sở dữ liệu SQLite
│   └── faces/                      
│       └── encodings.json          ← Vector sinh trắc học (Định dạng JSON an toàn)
└── requirements.txt