
 #include <Arduino.h>
 #include <WiFi.h>
 #include <ESPmDNS.h>
 #include <WiFiUdp.h>
 #include <ArduinoOTA.h>
 #include <PubSubClient.h>
 #include <ArduinoJson.h>
 #include <DHT.h>
 #include <Wire.h>
 #include <BH1750.h>
 #include <time.h>
 
 // Nhúng file chứa các biến bảo mật
 #include "secrets.h" 
 
 
 
 // ─── CẤU HÌNH WIFI & MQTT TỪ SECRETS ─────────────────────────
 const char* ssid        = WIFI_SSID;       
 const char* password    = WIFI_PASSWORD;   
 const char* mqtt_server = MQTT_SERVER;     
 const int   mqtt_port   = MQTT_PORT;
 const char* mqtt_user   = MQTT_USERNAME; 
 const char* mqtt_pass   = MQTT_PASSWORD;
 const char* ota_pass    = OTA_PASSWORD;
 
 // Các Topic MQTT
 const char* topic_sensor    = "home/room1/sensors";
 const char* topic_status    = "home/room1/status";
 const char* topic_light_cmd  = "home/room1/light/command";
 const char* topic_fan_cmd    = "home/room1/fan/command";
 const char* topic_light_state = "home/room1/light/state";
 const char* topic_fan_state   = "home/room1/fan/state";
 const char* topic_event_log   = "home/room1/events";
 
 // ─── CẤU HÌNH GPIO ───────────────────────────────────────────
 #define DHT_PIN         4
 #define DHT_TYPE        DHT22
 #define PIR_PIN         32    
 #define RELAY_LIGHT     26    // ACTIVE LOW: LOW=BẬT
 #define RELAY_FAN       27    // ACTIVE LOW: LOW=BẬT
 #define SDA_PIN         21
 #define SCL_PIN         22
 
 // ─── THAM SỐ THỜI GIAN (Tối ưu cho chuỗi LSTM) ───────────────
 #define DHT_READ_MS     20000   // Đọc DHT/BH1750 mỗi 20 giây
 #define PIR_READ_MS     100    // Quét cảm biến mỗi 100ms
 #define PIR_COOLDOWN_MS 2000   // Chống nhiễu PIR (Bỏ qua các dao động dưới 2 giây)
 #define HEARTBEAT_MS    60000 // Thời gian lấy mẫu (Gửi MQTT dù không có thay đổi)
 #define SERIAL_BAUD     115200
 
 // ─── ĐỐI TƯỢNG CẢM BIẾN & KẾT NỐI ────────────────────────────
 DHT           dht(DHT_PIN, DHT_TYPE);
 BH1750        lightMeter;
 WiFiClient    espClient;
 PubSubClient  client(espClient);
 
 // ─── TRẠNG THÁI HỆ THỐNG ────────────────────────────────────
 bool systemReady = false;
 
 // ─── MUTEX + STRUCT DỮ LIỆU CẢM BIẾN DÙNG CHUNG ─────────────
 SemaphoreHandle_t xSensorMutex;
 
 struct SensorReading {
   float temp;
   float hum;
   float lux;
   bool  motion;
   bool  newData;
   String timestampIso;
   unsigned long hour;
   unsigned long minute;
   unsigned long dayOfWeek;
   String payloadVersion;
   String source;
   String dataSource;
   String scenarioType;
   String sequencePhase;
   String userId;
   String dominantUserId;
   String decisionMode;
   bool  overrideFlag;
   String overrideReason;
   bool  lightState;
   bool  fanState;
   String sessionId;
   String presenceType;
   int   faceCount;
 } latestReading = {-999.0, 60.0, 0.0, false, false, "", 0, 0, 0, "v2", "esp32", "real", "background_noise", "background_noise", "unknown", "unknown", "single_user", false, "", false, false, "", "unknown", 0};
 
 bool bh1750_ready = false; 
 
 // ─────────────────────────────────────────────────────────────
 //  HÀM XỬ LÝ KHI NHẬN LỆNH ĐIỀU KHIỂN TỪ PYTHON
 // ─────────────────────────────────────────────────────────────
 void publishActuatorState(const char* topic, const char* payload, bool retain = true) {
   client.publish(topic, payload, retain);
 }
 
 void publishEventLog(const char* payload) {
   client.publish(topic_event_log, payload, false);
 }
 
 String makeIsoTimestamp() {
   struct tm timeinfo;
   if (getLocalTime(&timeinfo)) {
     char buffer[25];
     strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
     return String(buffer);
   }
   return String("");
 }
 
 bool getTimeFields(unsigned long &hour, unsigned long &minute, unsigned long &dayOfWeek, String &timestampIso) {
   struct tm timeinfo;
   if (!getLocalTime(&timeinfo)) {
     hour = 0;
     minute = 0;
     dayOfWeek = 0;
     timestampIso = "";
     return false;
   }
   hour = timeinfo.tm_hour;
   minute = timeinfo.tm_min;
   dayOfWeek = timeinfo.tm_wday;
   char buffer[25];
   strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
   timestampIso = String(buffer);
   return true;
 }
 
 void mqttCallback(char* topic, byte* payload, unsigned int length) {
   if (!systemReady) {
     Serial.printf("\n[MQTT-IN] Bo qua lenh truoc khi san sang: %s\n", topic);
     return;
   }
 
   Serial.printf("\n[MQTT-IN] Nhan tin tu Topic: %s\n", topic);
   Serial.print("[MQTT-IN] Noi dung: ");
   for (unsigned int i = 0; i < length; i++) {
     Serial.print((char)payload[i]);
   }
   Serial.println("\n");
 
   #if ARDUINOJSON_VERSION_MAJOR == 6
     StaticJsonDocument<256> doc;
   #else
     JsonDocument doc;
   #endif
   
   DeserializationError error = deserializeJson(doc, payload, length);
   
   if (error) {
     Serial.printf("[MQTT] Loi parse JSON: %s\n", error.c_str());
     return;
   }
 
   if (strcmp(topic, topic_light_cmd) == 0) {
     const char* state = doc["state"];
     if (state) {
         bool isOn = (strcmp(state, "ON") == 0);
         digitalWrite(RELAY_LIGHT, isOn ? LOW : HIGH);
         publishActuatorState(topic_light_state, isOn ? "{\"state\":\"ON\"}" : "{\"state\":\"OFF\"}");
 
         SensorReading snapshot = latestReading;
         if (xSemaphoreTake(xSensorMutex, portMAX_DELAY) == pdTRUE) {
           latestReading.source = doc["source"] | "manual";
           latestReading.dataSource = doc["data_source"] | "real";
           latestReading.scenarioType = doc["scenario_type"] | "manual_override";
           latestReading.sequencePhase = doc["sequence_phase"] | "override_response";
           latestReading.userId = doc["user_id"] | "unknown";
           latestReading.dominantUserId = doc["dominant_user_id"] | latestReading.userId;
           latestReading.decisionMode = doc["decision_mode"] | "single_user";
           latestReading.overrideFlag = (int)(doc["override_flag"] | 1) > 0;
           latestReading.overrideReason = doc["override_reason"] | "manual_override";
           latestReading.lightState = isOn;
           latestReading.fanState = (digitalRead(RELAY_FAN) == LOW);
           latestReading.sessionId = doc["session_id"] | "";
           latestReading.presenceType = doc["presence_type"] | "human";
           latestReading.faceCount = doc["face_count"] | 0;
           snapshot = latestReading;
           xSemaphoreGive(xSensorMutex);
         }
 
         char eventPayload[768];
 
         snprintf(eventPayload, sizeof(eventPayload),
                  "{\"timestamp\":\"%s\",\"hour\":%lu,\"minute\":%lu,\"day_of_week\":%lu,\"temperature\":%.2f,\"humidity\":%.2f,\"light_level\":%.2f,\"motion_detected\":%d,\"user_id\":\"%s\",\"dominant_user_id\":\"%s\",\"presence_type\":\"%s\",\"face_count\":%d,\"light_state\":%d,\"fan_state\":%d,\"confidence\":%.2f,\"source\":\"%s\",\"scenario_type\":\"%s\",\"sequence_phase\":\"%s\",\"override_flag\":%d,\"override_reason\":\"%s\",\"session_id\":\"%s\",\"decision_mode\":\"%s\",\"data_source\":\"%s\"}",
                  snapshot.timestampIso.c_str(), (unsigned long)0, (unsigned long)0, (unsigned long)0, snapshot.temp, snapshot.hum, snapshot.lux, snapshot.motion ? 1 : 0,
                  snapshot.userId.c_str(), snapshot.dominantUserId.c_str(), snapshot.presenceType.c_str(), snapshot.faceCount,
                  isOn ? 1 : 0, snapshot.fanState ? 1 : 0, 0.0, snapshot.source.c_str(), snapshot.scenarioType.c_str(), snapshot.sequencePhase.c_str(),
                  snapshot.overrideFlag ? 1 : 0, snapshot.overrideReason.c_str(), snapshot.sessionId.c_str(), snapshot.decisionMode.c_str(), snapshot.dataSource.c_str(), snapshot.payloadVersion.c_str());
         publishEventLog(eventPayload);
         Serial.printf("[Light] Den -> %s\n", isOn ? "BAT" : "TAT");
     }
   }
   
   else if (strcmp(topic, topic_fan_cmd) == 0) {
     const char* state = doc["state"];
     if (state) {
         bool isOn = (strcmp(state, "ON") == 0);
         digitalWrite(RELAY_FAN, isOn ? LOW : HIGH);
         publishActuatorState(topic_fan_state, isOn ? "{\"state\":\"ON\"}" : "{\"state\":\"OFF\"}");
 
         if (xSemaphoreTake(xSensorMutex, portMAX_DELAY) == pdTRUE) {
         latestReading.source = doc["source"] | "manual";
         latestReading.scenarioType = doc["scenario_type"] | "manual_override";
         latestReading.userId = doc["user_id"] | "unknown";
         latestReading.dominantUserId = doc["dominant_user_id"] | latestReading.userId;
         latestReading.decisionMode = doc["decision_mode"] | "single_user";
         latestReading.overrideFlag = (int)(doc["override_flag"] | 1) > 0;
         latestReading.overrideReason = doc["override_reason"] | "manual_override";
         latestReading.lightState = (digitalRead(RELAY_LIGHT) == LOW);
         latestReading.fanState = isOn;
         latestReading.sessionId = doc["session_id"] | "";
         latestReading.presenceType = doc["presence_type"] | "unknown";
         latestReading.faceCount = doc["face_count"] | 0;
         xSemaphoreGive(xSensorMutex);
         }
 
         char eventPayload[768];
         snprintf(eventPayload, sizeof(eventPayload),
                  "{\"timestamp\":\"%s\",\"hour\":%lu,\"minute\":%lu,\"day_of_week\":%lu,\"temperature\":%.2f,\"humidity\":%.2f,\"light_level\":%.2f,\"motion_detected\":%d,\"user_id\":\"%s\",\"dominant_user_id\":\"%s\",\"presence_type\":\"%s\",\"face_count\":%d,\"light_state\":%d,\"fan_state\":%d,\"confidence\":%.2f,\"source\":\"%s\",\"scenario_type\":\"%s\",\"sequence_phase\":\"%s\",\"override_flag\":%d,\"override_reason\":\"%s\",\"session_id\":\"%s\",\"decision_mode\":\"%s\",\"data_source\":\"%s\"}",
                  latestReading.timestampIso.c_str(), 0UL, 0UL, 0UL, latestReading.temp, latestReading.hum, latestReading.lux, latestReading.motion ? 1 : 0,
                  latestReading.userId.c_str(), latestReading.dominantUserId.c_str(), latestReading.presenceType.c_str(), latestReading.faceCount,
                  latestReading.lightState ? 1 : 0, isOn ? 1 : 0, 0.0, latestReading.source.c_str(), latestReading.scenarioType.c_str(), latestReading.sequencePhase.c_str(),
                  latestReading.overrideFlag ? 1 : 0, latestReading.overrideReason.c_str(), latestReading.sessionId.c_str(), latestReading.decisionMode.c_str(), latestReading.dataSource.c_str(), latestReading.payloadVersion.c_str());
         publishEventLog(eventPayload);
         Serial.printf("[Fan] Quat -> %s\n", isOn ? "BAT" : "TAT");
     }
   }
 
 }
 
 // ─────────────────────────────────────────────────────────────
 //  [FIX BUG-ESP-02] HÀM KẾT NỐI LẠI WIFI & MQTT (NON-BLOCKING)
 // ─────────────────────────────────────────────────────────────
 unsigned long lastReconnectAttempt = 0;
 
 void reconnect() {
   if (client.connected()) return;
 
   unsigned long now = millis();
   // Rate limiting: Thử kết nối lại mỗi 5 giây để không chiếm dụng CPU
   if (now - lastReconnectAttempt < 5000 && lastReconnectAttempt != 0) {
     return; 
   }
   lastReconnectAttempt = now;
 
   if (WiFi.status() != WL_CONNECTED) {
     Serial.println("\n[WiFi] Mat ket noi mang! Dang thu ket noi lai...");
     WiFi.disconnect();
     WiFi.reconnect(); 
     return; // Thoát hàm để không block Task, chờ chu kỳ sau kiểm tra lại
   }
 
   Serial.print("[MQTT] Dang ket noi toi Broker...");
   
   String clientId = "SmartHome_ESP32_";
   String macAddr = WiFi.macAddress();
   macAddr.replace(":", "");
   clientId += macAddr;
   
   if (client.connect(clientId.c_str(), mqtt_user, mqtt_pass, topic_status, 0, true, "{\"status\": \"offline\"}")) {
     Serial.println(" THANH CONG!");
     client.publish(topic_status, "{\"status\": \"online\"}", true);
     
     bool isLightOn = (digitalRead(RELAY_LIGHT) == LOW); 
     bool isFanOn   = (digitalRead(RELAY_FAN) == LOW);
     
     // Gửi bản tin Retain (Lưu giữ) lên topic state để Web cập nhật ngay lập tức
     publishActuatorState(topic_light_state, isLightOn ? "{\"state\":\"ON\"}" : "{\"state\":\"OFF\"}", true);
     publishActuatorState(topic_fan_state, isFanOn ? "{\"state\":\"ON\"}" : "{\"state\":\"OFF\"}", true);
 
     if (xSemaphoreTake(xSensorMutex, portMAX_DELAY) == pdTRUE) {
       latestReading.source = "real";
       latestReading.scenarioType = "background_noise";
       latestReading.userId = "unknown";
       latestReading.dominantUserId = "unknown";
       latestReading.decisionMode = "single_user";
       latestReading.overrideFlag = false;
       latestReading.overrideReason = "";
       latestReading.lightState = isLightOn;
       latestReading.fanState = isFanOn;
       latestReading.presenceType = "unknown";
       latestReading.faceCount = 0;
       xSemaphoreGive(xSensorMutex);
     }
 
     client.subscribe(topic_light_cmd, 1);
     client.subscribe(topic_fan_cmd, 1);
     
     lastReconnectAttempt = 0; // Reset bộ đếm sau khi thành công
   } else {
     Serial.printf(" THAT BAI (Ma loi: %d). Thu lai sau 5 giay...\n", client.state());
   }
 }
 
 // ─────────────────────────────────────────────────────────────
 //  TASK 1: ĐỌC CẢM BIẾN (Core 1)
 // ─────────────────────────────────────────────────────────────
 void sensorTask(void* pvParameters) {
   Serial.println("[Sensor] Task bat dau tren Core 1");
 
   unsigned long lastDhtTime = 0;
   unsigned long lastPirChangeTime = 0; 
   unsigned long lastHeartbeatTime = 0; // Thêm bộ đếm Heartbeat
   
   float lastTemp = -999.0, lastHum = -999.0, lastLux = -999.0;
   bool lastMotion = false;
   
   unsigned long localLightTurnOnTime = 0;
   bool isLocalFallbackActive = false;
 
   for (;;) {
     bool forcePublish = false;
 
     // 1. ĐỌC CẢM BIẾN CHUYỂN ĐỘNG (Event-Driven Tức thời)
     bool currentMotion = (digitalRead(PIR_PIN) == HIGH);
     
     if (currentMotion != lastMotion) {
       if (millis() - lastPirChangeTime > PIR_COOLDOWN_MS) {
         lastMotion = currentMotion;
         lastPirChangeTime = millis();
         forcePublish = true; // Luôn báo cáo ngay khi PIR thay đổi trạng thái
         latestReading.source = "real";
         latestReading.scenarioType = "sequence_behavior";
       }
     } 
     
     // TÍNH NĂNG TỰ TRỊ CỤC BỘ (LOCAL FALLBACK MODE)
     // Chỉ cho phép tự bật đèn khi MQTT online để tránh trạng thái bật ngoài ý muốn lúc khởi động.
     if (!client.connected()) {
         if (lastMotion && !isLocalFallbackActive) {
             Serial.println("[Fallback] MQTT Offline -> Phát hiện chuyển động nhưng KHÔNG tự bật đèn.");
             isLocalFallbackActive = true;
         }
         if (!lastMotion) {
             isLocalFallbackActive = false;
         }
     } else {
         isLocalFallbackActive = false;
     }
 
     // 2. ĐỌC NHIỆT ẨM ÁNH SÁNG (Kiểm tra mỗi 10 giây, nhưng chỉ gửi khi có Delta/Heartbeat)
     if (millis() - lastDhtTime >= DHT_READ_MS) {
       lastDhtTime = millis();
       float t = dht.readTemperature();
       float h = dht.readHumidity();
       
       float l = -1.0;
       if (bh1750_ready) {
           l = lightMeter.readLightLevel();
       }
 
       // [FIX BUG-E2] LỚP BẢO VỆ CHUỖI & ĐẾM LỖI
       static int dht_fail_count = 0;
       float validTemp = lastTemp;
       float validHum = lastHum;
       
       if (!isnan(t) && !isnan(h) && t > 0.0 && h > 0.0) {
         if (lastTemp == -999.0 || abs(t - lastTemp) < 5.0) {
             validTemp = t;
             validHum = h;
             dht_fail_count = 0;  // Reset counter khi đọc thành công
         } else {
             Serial.printf("[Sensor] GLITCH FILTERED: Nhay vot bat thuong (%.1f -> %.1f).\n", lastTemp, t);
         }
       } else {
         dht_fail_count++;
         Serial.printf("[Sensor] WARN: Loi doc DHT22 (0.0 hoac NaN) #%d\n", dht_fail_count);
       }
       
       float validLux = (l >= 0) ? l : lastLux;
 
       // -------------------------------------------------------------
       // [TỐI ƯU HÓA] BỘ LỌC DELTA VÀ HEARTBEAT (Thay vì forcePublish = true mù quáng)
       // -------------------------------------------------------------
       bool envChanged = false;
       
       // Bỏ qua kiểm tra Delta ở lần lấy mẫu đầu tiên (khi lastTemp đang là giả định 25.0)
       if (lastTemp != -999.0) { 
           if (abs(validTemp - lastTemp) >= 0.5 || abs(validHum - lastHum) >= 3.0 || abs(validLux - lastLux) >= 20.0) {
               envChanged = true;
           }
       } else {
           envChanged = true; // Bắt buộc gửi lần đầu để đồng bộ hệ thống
       }
 
       bool isHeartbeat = (millis() - lastHeartbeatTime >= HEARTBEAT_MS);
 
       if (envChanged || isHeartbeat) {
           lastTemp = validTemp;
           lastHum = validHum;
           lastLux = validLux;
           lastHeartbeatTime = millis(); // Reset Heartbeat
           forcePublish = true; // Kích hoạt cờ cho phép MQTT đẩy gói tin
           
           if (isHeartbeat && !envChanged) {
               Serial.println("[Sensor] Gửi Heartbeat duy trì kết nối (2 phút).");
           } else {
               Serial.println("[Sensor] Gửi MQTT do có thay đổi thông số Môi trường (Delta).");
           }
       }
 
       // [FIX BUG-E2] Sau 3 lần lỗi liên tiếp, force gửi để server biết (tránh kẹt Delta filter)
       if (dht_fail_count >= 3) {
           forcePublish = true;
           Serial.println("[Sensor] Force publish do DHT loi lien tiep.");
           dht_fail_count = 0; // Reset để không spam liên tục (sẽ gửi lại sau 3 fail nữa)
       }
     }
 
     // 3. GỬI GÓI TIN ĐÃ QUA LỌC XUỐNG CORE 0
     if (forcePublish) {
       unsigned long hour = 0, minute = 0, dayOfWeek = 0;
       String timestampIso;
       getTimeFields(hour, minute, dayOfWeek, timestampIso);
       if (xSemaphoreTake(xSensorMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
         latestReading.temp   = lastTemp;
         latestReading.hum    = lastHum;
         latestReading.lux    = lastLux;
         latestReading.motion = lastMotion;
         latestReading.newData = true;
         latestReading.timestampIso = timestampIso;
         latestReading.hour = hour;
         latestReading.minute = minute;
         latestReading.dayOfWeek = dayOfWeek;
         latestReading.payloadVersion = "v2";
         latestReading.sessionId = latestReading.sessionId.length() ? latestReading.sessionId : String((uint32_t)esp_random(), HEX);
         latestReading.userId = latestReading.userId.length() ? latestReading.userId : "unknown";
         latestReading.dominantUserId = latestReading.dominantUserId.length() ? latestReading.dominantUserId : latestReading.userId;
         latestReading.presenceType = latestReading.presenceType.length() ? latestReading.presenceType : (lastMotion ? "human" : "unknown");
         latestReading.faceCount = latestReading.faceCount;
         xSemaphoreGive(xSensorMutex);
       }
       Serial.printf("[Sensor] Data Ready -> %.1f C | %.0f%% | %.0f lux | PIR:%s\n",
                     lastTemp, lastHum, lastLux, lastMotion ? "CO" : "KHONG");
     }
 
     vTaskDelay(pdMS_TO_TICKS(PIR_READ_MS));
   }
 }
 
 // ─────────────────────────────────────────────────────────────
 //  TASK 2: XỬ LÝ MQTT & GỬI DỮ LIỆU (Core 0)
 // ─────────────────────────────────────────────────────────────
 void mqttTask(void* pvParameters) {
   Serial.println("[MQTT] Task bat dau tren Core 0");
 
   for (;;) {
     if (!client.connected()) {
       reconnect();
     } else {
       client.loop(); 
     }
 
     bool shouldPublish = false;
     SensorReading tempReading;
 
     if (xSemaphoreTake(xSensorMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
       if (latestReading.newData) {
         tempReading = latestReading;    
         latestReading.newData = false;  
         shouldPublish = true;
       }
       xSemaphoreGive(xSensorMutex);
     }
 
     if (shouldPublish && client.connected()) {
       #if ARDUINOJSON_VERSION_MAJOR == 6
         StaticJsonDocument<256> doc;
       #else
         JsonDocument doc;
       #endif
       
       doc["timestamp"] = tempReading.timestampIso;
       doc["hour"] = tempReading.hour;
       doc["minute"] = tempReading.minute;
       doc["day_of_week"] = tempReading.dayOfWeek;
       doc["temperature"] = tempReading.temp;
       doc["humidity"] = tempReading.hum;
       doc["light_level"] = tempReading.lux;
       doc["motion_detected"] = tempReading.motion;
       doc["user_id"] = tempReading.userId;
       doc["dominant_user_id"] = tempReading.dominantUserId;
       doc["presence_type"] = tempReading.presenceType;
       doc["face_count"] = tempReading.faceCount;
       doc["light_state"] = tempReading.lightState;
       doc["fan_state"] = tempReading.fanState;
       doc["confidence"] = 0.0;
       doc["source"] = tempReading.source;
       doc["scenario_type"] = tempReading.scenarioType;
       doc["sequence_phase"] = tempReading.sequencePhase;
       doc["override_flag"] = tempReading.overrideFlag;
       doc["override_reason"] = tempReading.overrideReason;
       doc["session_id"] = tempReading.sessionId;
       doc["decision_mode"] = tempReading.decisionMode;
       doc["data_source"] = tempReading.dataSource;
       
 
       char jsonBuffer[768];
       serializeJson(doc, jsonBuffer, sizeof(jsonBuffer));
       
       client.publish(topic_sensor, jsonBuffer);
     }
 
     vTaskDelay(pdMS_TO_TICKS(50)); 
   }
 }
 
 // ─────────────────────────────────────────────────────────────
 //  SETUP & VÒNG LẶP CHÍNH
 // ─────────────────────────────────────────────────────────────
 void setup() {
   randomSeed(esp_random());
 
   // Đưa relay về trạng thái an toàn càng sớm càng tốt
   pinMode(RELAY_LIGHT, OUTPUT);
   digitalWrite(RELAY_LIGHT, HIGH);
 
   pinMode(RELAY_FAN, OUTPUT);
   digitalWrite(RELAY_FAN, HIGH);
   delay(100);
   
   Serial.begin(SERIAL_BAUD);
   delay(1500);
   Serial.println("\n[Setup] Smart Home ESP32 - LSTM Edge Node");
 
   pinMode(PIR_PIN, INPUT_PULLDOWN);
 
   dht.begin();
   Wire.begin(SDA_PIN, SCL_PIN);
   
   if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE_2)) {
       bh1750_ready = true;
   } else {
       Serial.println("[Setup] WARN: BH1750 khong tim thay");
   }
 
   Serial.printf("[WiFi] Dang ket noi vao %s ", ssid);
   WiFi.begin(ssid, password);
   while (WiFi.status() != WL_CONNECTED) {
     delay(500);
     Serial.print(".");
   }
   Serial.println("\n[WiFi] KET NOI THANH CONG!");
   Serial.print("[WiFi] IP Address: ");
   Serial.println(WiFi.localIP());
 
   ArduinoOTA.setHostname("SmartHome-ESP32-Node");
   ArduinoOTA.setPassword(ota_pass);
   
   ArduinoOTA.onStart([]() {
     Serial.println("\n[OTA] Bat dau cap nhat Firmware...");
   });
   ArduinoOTA.onEnd([]() {
     Serial.println("\n[OTA] Cap nhat hoan tat. Dang khoi dong lai...");
   });
   ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
     Serial.printf("[OTA] Tien do: %u%%\r", (progress / (total / 100)));
   });
   ArduinoOTA.onError([](ota_error_t error) {
     Serial.printf("\n[OTA] Loi cap nhat [%u]: ", error);
     if (error == OTA_AUTH_ERROR) Serial.println("Loi xac thuc");
     else if (error == OTA_BEGIN_ERROR) Serial.println("Loi khoi tao");
     else if (error == OTA_CONNECT_ERROR) Serial.println("Loi ket noi");
     else if (error == OTA_RECEIVE_ERROR) Serial.println("Loi nhan du lieu");
     else if (error == OTA_END_ERROR) Serial.println("Loi ket thuc");
   });
   
   configTime(7 * 3600, 0, "pool.ntp.org", "time.nist.gov");
   Serial.println("[NTP] Dang dong bo thoi gian...");
   ArduinoOTA.begin();
   Serial.println("[OTA] He thong san sang nhan ban cap nhat qua WiFi");
 
   client.setBufferSize(1024); 
   client.setServer(mqtt_server, mqtt_port);
   client.setCallback(mqttCallback);
 
   xSensorMutex = xSemaphoreCreateMutex();
 
   // [FIX BUG-ESP-01] Tăng kích thước bộ nhớ Stack cho sensorTask từ 4096 lên 8192
   xTaskCreatePinnedToCore(sensorTask, "SensorTask", 8192, NULL, 1, NULL, 1);
   xTaskCreatePinnedToCore(mqttTask,   "MqttTask",   16384, NULL, 2, NULL, 0);
 
   systemReady = true;
   Serial.println("[Setup] OK - FreeRTOS tasks dang chay");
 }
 
 void loop() {
   ArduinoOTA.handle();
   vTaskDelay(pdMS_TO_TICKS(10)); 
 }