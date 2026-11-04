
# Sử dụng Relative Import (import tương đối) để lấy các Class chính ra ngoài
from .face_recognition_module import FaceRecognitionSystem
from .mqtt_handler import MQTTHandler
from .control_pipeline import ControlPipeline
from .hysteresis_controller import HysteresisController

# Định nghĩa __all__ để kiểm soát chính xác những gì được export ra ngoài
# khi file khác gọi: `from modules import *`
__all__ = [
    "FaceRecognitionSystem",
    "MQTTHandler",
    "ControlPipeline",
    "HysteresisController",
]

# (Tùy chọn) Metadata của package
__version__ = "1.0.0"
__author__ = "Nguyen Tuan Tu"
