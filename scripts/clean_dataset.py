import os
import shutil
import cv2
from tqdm import tqdm
def clean_dataset(images_path, labels_path, out_path, removed_path, invalid_path):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(removed_path, exist_ok=True)
    os.makedirs(invalid_path, exist_ok=True)
    for lbl in os.listdir(labels_path):
        lbl_path = os.path.join(labels_path, lbl)
        if lbl.endswith(".txt") and "INVALID" in lbl.upper():
            shutil.move(lbl_path, os.path.join(invalid_path, lbl))
    for img_file in tqdm(os.listdir(images_path), desc="Cleaning images"):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(images_path, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[CORRUPTED IMAGE] {img_file}")
            shutil.move(img_path, os.path.join(removed_path, img_file))
            continue
        if not os.path.exists(label_path):
            print(f"[NO LABEL] {img_file}")
            shutil.move(img_path, os.path.join(removed_path, img_file))
            continue
        if os.path.getsize(label_path) == 0:
            print(f"[EMPTY LABEL] {label_file}")
            shutil.move(img_path, os.path.join(removed_path, img_file))
            shutil.move(label_path, os.path.join(invalid_path, label_file))
            continue
        shutil.copy(img_path, os.path.join(out_path, img_file))
        shutil.copy(label_path, os.path.join(out_path, label_file))
    for lbl_file in os.listdir(labels_path):
        lbl_path = os.path.join(labels_path, lbl_file)
        if not lbl_file.lower().endswith(".txt"):
            continue
        corresponding_img = os.path.splitext(lbl_file)[0]
        found_image = False
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            if os.path.exists(os.path.join(images_path, corresponding_img + ext)):
                found_image = True
                break
        if not found_image:
            print(f"[LABEL WITHOUT IMAGE] {lbl_file}")
            shutil.move(lbl_path, os.path.join(invalid_path, lbl_file))
    print("\nDataset cleaned successfully!")
    print(f"Cleaned files → {out_path}")
    print(f"Removed images → {removed_path}")
    print(f"Invalid labels → {invalid_path}")
if __name__ == "__main__":
    images_path = "/content/drive/MyDrive/Infosys/css-data/train/images"
    labels_path = "/content/drive/MyDrive/Infosys/css-data/train/labels"
    out_path = "/content/drive/MyDrive/Infosys/css-data/annotations_cleaned/train"
    removed_path = "/content/drive/MyDrive/Infosys/css-data/images_removed"
    invalid_path = "/content/drive/MyDrive/Infosys/css-data/labels_invalid"
    clean_dataset(images_path, labels_path, out_path, removed_path, invalid_path)