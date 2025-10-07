import cv2
import os
import random
import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
def load_classes(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]
def draw_boxes(image_path, label_path, class_file, save_dir=None):
    classes = load_classes(class_file)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return None 
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            print(f"Invalid label format in {label_path}")
            continue
        try:
            cls_id, x, y, bw, bh = map(float, parts)
            cls_id = int(cls_id)
        except ValueError:
            print(f"Non-numeric label in {label_path}")
            continue
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = classes[cls_id] if cls_id < len(classes) else f"ID:{cls_id}"
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, img)
    return img
def visualize_dataset(img_dir, label_dir, class_file, save_dir, sample_count=10):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    image_paths = sorted(image_paths)
    if not image_paths:
        print("No images found.")
        return
    print(f"Found {len(image_paths)} images. Generating visualizations...")
    for img_path in tqdm(image_paths, desc="Drawing boxes"):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        draw_boxes(img_path, label_path, class_file, save_dir)
    print(f"\nAll annotated images saved to: {save_dir}")
    sample_images = random.sample(image_paths, min(sample_count, len(image_paths)))
    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(sample_images):
        vis_path = os.path.join(save_dir, os.path.basename(img_path))
        img = cv2.imread(vis_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, sample_count, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(os.path.basename(img_path))
    plt.show()
if __name__ == "__main__":
    img_dir = "/content/drive/MyDrive/Infosys/css-data/train/images"
    label_dir = "/content/drive/MyDrive/Infosys/css-data/train/labels"
    class_file = "/content/drive/MyDrive/Infosys/css-data/class_names.txt"
    save_dir = "/content/drive/MyDrive/Infosys/css-data/visualization/samples"
    visualize_dataset(img_dir, label_dir, class_file, save_dir, sample_count=3)