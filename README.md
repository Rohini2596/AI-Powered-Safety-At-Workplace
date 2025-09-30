# SafetyEye – AI-Powered Workplace Occupancy & Safety Monitor

## Project Overview
**SafetyEye** is an AI-based system designed to monitor workplace occupancy levels and ensure safety compliance using video surveillance feeds. The platform detects personal protective equipment (PPE) violations such as missing helmets or vests, generates real-time alerts, and provides a live dashboard for administrators to monitor safety compliance.

This tool helps office and industrial space managers improve space utilization while ensuring employees follow safety protocols.

---

## Project Outcomes
- Detect safety compliance violations (e.g., missing helmet, vest, mask).
- Generate visualizations and alerts to improve workplace safety.
- Present real-time results via a dashboard with analytics for administrators.

---

## Dataset
We use the **Construction Site Safety Image Dataset** (Roboflow) available at [Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow).  

Dataset is provided in **YOLO format** for training object detection models (images + labels).

---

## Project Modules
1. **Data Preparation**
   - Load and process YOLO-formatted images.
   - Split dataset into training, validation, and test sets.
   - Preprocess images and labels for YOLOv8 training.

2. **Model Training**
   - Train YOLOv8 model for PPE detection.
   - Perform validation, hyperparameter tuning, and augmentation.
   - Evaluate using metrics: mAP, precision, recall.

3. **Detection**
   - Real-time video processing using the trained model.
   - Overlay bounding boxes and labels for detected PPE.
   - Identify missing PPE as violations.

4. **Alerts**
   - Trigger notifications for detected violations.
   - Alerts can be via console logs, email, or GUI warning.


## Project Architecture

