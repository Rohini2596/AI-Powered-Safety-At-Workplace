import cv2
from ultralytics import YOLO

class PPEDetector:
    def __init__(self, model_path=r"C:\WE\Infosys\models\yolov8_ppe.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = [
            "Hardhat",
            "Mask",
            "NO-Hardhat",
            "NO-Mask",
            "NO-Safety Vest",
            "Person",
            "Safety Cone",
            "Safety Vest",
            "Machinery",
            "Vehicle"
        ]

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        annotated = results[0].plot()

        detections = []
        boxes = results[0].boxes

        for box in boxes:
            cls_idx = int(box.cls[0])
            class_name = self.model.names[cls_idx]

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 2),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

        return annotated, detections