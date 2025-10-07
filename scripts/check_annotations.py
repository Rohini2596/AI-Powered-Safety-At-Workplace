import os
import cv2
from tqdm import tqdm
def load_classes(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]
def check_annotations(images_dir, labels_dir, invalid_dir, class_file):
    classes = load_classes(class_file)
    os.makedirs(invalid_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    label_files = set(os.listdir(labels_dir))
    invalid_count = 0
    missing_count = 0
    print("Checking annotations...\n")
    for img_file in tqdm(image_files):
        img_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        if not os.path.exists(label_path):
            print(f"[MISSING LABEL] → {img_file}")
            missing_count += 1
            continue
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[CORRUPTED IMAGE] → {img_file}")
            continue
        h, w = img.shape[:2]
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) == 0:
            print(f"[EMPTY LABEL] → {label_path}")
            os.rename(label_path, os.path.join(invalid_dir, os.path.basename(label_path)))
            invalid_count += 1
            continue
        seen_boxes = set()
        valid = True
        reason = ""
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                valid = False
                reason = "INVALID FORMAT"
                break
            try:
                cls_id = int(parts[0])
                x, y, bw, bh = map(float, parts[1:])
            except ValueError:
                valid = False
                reason = "NON-NUMERIC VALUE"
                break
            if cls_id < 0 or cls_id >= len(classes):
                valid = False
                reason = f"INVALID CLASS ID ({cls_id})"
                break
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                valid = False
                reason = f"OUT OF RANGE BBOX ({x}, {y}, {bw}, {bh})"
                break
            if line in seen_boxes:
                valid = False
                reason = "DUPLICATE BOX"
                break
            seen_boxes.add(line)
        if not valid:
            print(f"{label_path} → {reason}")
            os.rename(label_path, os.path.join(invalid_dir, os.path.basename(label_path)))
            invalid_count += 1
    print("\nAnnotation check complete!")
    print(f"Invalid labels moved to: {invalid_dir}")
    print(f"Missing labels: {missing_count}")
    print(f"Invalid labels: {invalid_count}")
if __name__ == "__main__":
    images_dir = "/content/drive/MyDrive/Infosys/css-data/train/images"
    labels_dir = "/content/drive/MyDrive/Infosys/css-data/train/labels"
    invalid_dir = "/content/drive/MyDrive/Infosys/css-data/labels_invalid"
    class_file = "/content/drive/MyDrive/Infosys/css-data/class_names.txt"
    check_annotations(images_dir, labels_dir, invalid_dir, class_file)