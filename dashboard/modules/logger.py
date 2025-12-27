import csv
import os
from datetime import datetime

class ViolationLogger:
    def __init__(self, log_path="data/violations.csv"):
        self.log_path = log_path

        # Create directory if missing
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Create file with header if missing
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "worker_id", "violation_type", "confidence", "status"])

    def write(self, worker_id, violation_type, confidence, status):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                worker_id,
                violation_type,
                round(confidence, 3),
                status
            ])
