✨ 1. Tính Năng Nổi Bật 
- AI Cá Nhân Hóa (Hybrid LSTM-XGBoost): Thay vì dùng các tập luật (if-else) cứng nhắc, hệ thống kết hợp mạng nơ-ron hồi quy LSTM để trích xuất đặc trưng chuỗi thời gian (sự thay đổi nhiệt, độ ẩm, ánh sáng trong quá khứ) và cây quyết định XGBoost để phân loại và ra lệnh Bật/Tắt chính xác theo thói quen của từng cá nhân.   

- Nhận Diện Người Dùng và Phân Quyền Khách/Chủ: Tích hợp thuật toán YOLOv8 (phát hiện người realtime nhẹ và mượt trên laptop) và DeepFace (nhận diện khuôn mặt). Hệ thống có khả năng nhận biết phòng trống (chuyển về mode tự động tắt), nhận biết khách hoặc ưu tiên quyết định theo người dùng chính khi có nhiều người trong phòng.   

- Active Learning và Vòng Lặp Phản Hồi: Thuật toán tự động quét lịch sử để tìm các sai lầm của AI. Nếu AI tự động tắt quạt nhưng người dùng bật lại trong vòng 5 phút, hệ thống sẽ xóa bỏ quyết định sai của AI khỏi tập dữ liệu và nhân trọng số cao cho hành động sửa sai của người dùng để học lại.   

- Edge Computing với ESP32 và FreeRTOS: Vi điều khiển ESP32 xử lý song song nhờ hệ điều hành FreeRTOS. Core 1 chuyên quét cảm biến và chống nhiễu, Core 0 chuyên xử lý kết nối MQTT. Đặc biệt, node ESP32 có tính năng "Local Fallback" - tự trị cục bộ, tiếp tục duy trì điều khiển an toàn (ví dụ: tắt đèn khi không có người) ngay cả khi mất kết nối WiFi/Server.   

- Digital Twin: Môi trường mô phỏng ảo cho phép giả lập các trạng thái nhiệt độ, ánh sáng và người dùng ra/vào phòng để kiểm thử các kịch bản của AI mà không cần phải tương tác vật lý với phần cứng thực tế.   

- Cơ Chế Hysteresis (Chống Nhấp Nháy): AI được tích hợp bộ điều khiển trễ (Hysteresis) để tránh tình trạng các thiết bị như Quạt/Điều hòa bị bật tắt liên tục khi nhiệt độ dao động quanh ngưỡng ranh giới.   

🏗️ 2. Kiến Trúc Hệ Thống 
Hệ thống được chia thành hai phân hệ chính giao tiếp với nhau qua giao thức MQTT:

- Phần Cứng (Edge Node - C++/Arduino):

  Vi điều khiển: ESP32 chạy FreeRTOS.   

  Cảm biến: DHT22 (Nhiệt/Ẩm), BH1750 (Cảm biến ánh sáng chuyên dụng giao tiếp I2C), PIR (Phát hiện chuyển động tức thời).   

  Thiết bị chấp hành: Module Relay 2 kênh điều khiển luồng điện 220V cho Đèn và Quạt.   

  Cơ chế cập nhật: Hỗ trợ OTA cho phép cập nhật Firmware không dây qua WiFi.   

- Phần Mềm và Trí Tuệ Nhân Tạo:

  Web Dashboard: Xây dựng bằng Flask, sử dụng Server-Sent Events đẩy dữ liệu theo thời gian thực lên giao diện Web.   

  Giao thức truyền thông: Mosquitto MQTT Broker làm trung gian truyền bản tin JSON giữa ESP32 và Server Python.   

  Cơ sở dữ liệu: SQLite được tối ưu hóa với chế độ PRAGMA WAL, tự động dọn dẹp các log cũ và chạy VACUUM để chống phân mảnh, đảm bảo tốc độ đọc/ghi cao cho dữ liệu IoT liên tục.   

  AI Models: TensorFlow/Keras cho mạng LSTM, XGBoost Classifier, Scikit-learn cho các bộ Scaler/Encoder.   

  Xử lý luồng đa nhiệm: Server Python sử dụng threading và queue để chạy độc lập các Worker xử lý MQTT, Camera Vision và AI Inference mà không gây nghẽn Web Server.   

🧠 3. Luồng Xử Lý Của Trí Tuệ Nhân Tạo
Quá trình AI học hỏi và ra quyết định diễn ra theo một chu trình khép kín:

- Thu Thập và Nội Suy Dữ Liệu: Dữ liệu môi trường liên tục được lưu trữ. Ban đầu, hệ thống sẽ chạy ở "Shadow Mode" (chỉ quan sát, không can thiệp). Để làm phong phú dữ liệu, kịch bản gốc của người dùng sẽ được đưa qua auto_pipeline.py để nội suy khoảng trống thời gian và thêm nhiễu Gauss (nhiệt độ, ánh sáng ngẫu nhiên) để sinh ra hàng ngàn mẫu huấn luyện bao quát các ngày/đêm khác nhau.   

- Khảo Sát Phản Hồi: Script data_extractor.py quét lịch sử. Thuật toán tìm kiếm các thao tác ghi đè từ người dùng. Các bản ghi mà AI dự đoán sai bị đánh cờ xóa, và các bản ghi do người dùng tự thao tác tay được hệ thống tự động gán nhãn ưu tiên để AI điều chỉnh trọng số.   

- Cấu Trúc Cửa Sổ Chuỗi: Dữ liệu không đưa vào AI từng dòng rời rạc mà được đóng gói thành các chuỗi thời gian. Điều này giúp AI hiểu được sự biến đổi: ví dụ "phòng đang mát dần lên" thay vì chỉ biết "nhiệt độ hiện tại là 28 độ".   

- Huấn Luyện Multi-task:

  Đầu tiên, các chuỗi đặc trưng môi trường đi qua mạng LSTM với cơ chế Attention để trích xuất ra một vector đặc trưng không gian-thời gian.   

  Vector này sau đó được truyền vào hai mô hình XGBoost độc lập: một để quyết định Đèn, một để quyết định Quạt.   

  Suy Luận và Áp Dụng Thực Tế: Khi hoạt động, chuỗi cảm biến 1 phút gần nhất được tổng hợp. Kết quả dự đoán (Đèn/Quạt) cùng với Độ tin cậy sẽ được đánh giá qua một Bộ phân giải. Nếu độ tin cậy quá thấp, AI sẽ nhường quyền cho các tập luật an toàn để tránh bật/tắt thiết bị một cách bất hợp lý.   
