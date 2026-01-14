import numpy as np
import pandas as pd
import os
import zipfile
import random
import shutil
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gdown

# 1. Download necessary files using gdown
'''def download_files():
    # Model weights and datasets
    os.system("gdown --id 1K3ONWL95Ol7Y47yXJbI2jza8iwPjY4uJ")
    os.system("gdown --id 15NBOHLitJblqsrVWDC1RO5FH8_va4yct")
    os.system("gdown --id 1wF-j4VaHh5scAu1n7hMzyvuPEb70mM_a")
    os.system("gdown --id 1kojPlfQqfP7JZqehnnLiuDyICIhOxcbU")'''

def download_files():
    # Format: "File_ID": "Desired_Filename"
    files = {
        "1K3ONWL95Ol7Y47yXJbI2jza8iwPjY4uJ": "file1.pt",
        "15NBOHLitJblqsrVWDC1RO5FH8_va4yct": "file2.pt",
        "1wF-j4VaHh5scAu1n7hMzyvuPEb70mM_a": "file3.zip",
        "1kojPlfQqfP7JZqehnnLiuDyICIhOxcbU": "file4.zip"
    }

    for file_id, filename in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            # Using gdown's python API is cleaner than os.system
            gdown.download(id=file_id, output=filename, quiet=False)
        else:
            print(f"Skipping {filename}, already exists.")

# 2. Unzip utility
def silent_unzip(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# 3. Data Preparation (Train/Val split)
def prepare_dataset():
    random.seed(42)
    img_dir = "./yolo/yolo-647/new-yolo/images"
    lbl_dir = "./yolo/yolo-647/new-yolo/labels"
    
    train_img_dir = "./dataset/images/train"
    val_img_dir = "./dataset/images/val"
    train_lbl_dir = "./dataset/labels/train"
    val_lbl_dir = "./dataset/labels/val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    split_ratio = 0.9  
    split_index = int(len(images) * split_ratio)

    train_imgs = images[:split_index]
    val_imgs = images[split_index:]

    for img_list, target_img_dir, target_lbl_dir in [(train_imgs, train_img_dir, train_lbl_dir), (val_imgs, val_img_dir, val_lbl_dir)]:
        for img_file in img_list:
            base = os.path.splitext(img_file)[0]
            label_file = base + ".txt"
            shutil.copy(os.path.join(img_dir, img_file), os.path.join(target_img_dir, img_file))
            if os.path.exists(os.path.join(lbl_dir, label_file)):
                shutil.copy(os.path.join(lbl_dir, label_file), os.path.join(target_lbl_dir, label_file))

# 4. Helper for Visualization
def draw_boxes(image, boxes, color=(0, 255, 0), label=''):
    for box in boxes:
        x_center, y_center, width, height = box
        h, w, _ = image.shape
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

if __name__ == "__main__":
    # Download and Unzip
    download_files()
    silent_unzip('yolo-647.zip', 'yolo')
    
    # Prepare Data
    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    prepare_dataset()
    
    # Create data.yaml
    with open("dataset/data.yaml", "w") as f:
        f.write("train: images/train\nval: images/val\nnc: 1\nnames: ['h_pylori']")

    # Load Model
    model = YOLO('last(dabest)220(2-4).pt') #base model used is yolo11l.pt
    
    # Optional: Training
    # torch.cuda.empty_cache()
    # model.train(data='dataset/data.yaml', epochs=100, imgsz=1280, batch=4, name='small_dataset_yolo', project='runs/train')

    # Inference Example
    image_path = './yolo/22.12.25/0001.jpg'
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model.predict(
            source=image_path, imgsz=1280, conf=0.1, iou=0.45,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        pred_boxes = results[0].boxes.xywhn.cpu().numpy()
        img_pred = draw_boxes(image.copy(), pred_boxes, color=(0, 255, 0), label='Pred')

        plt.figure(figsize=(12, 6))
        plt.imshow(img_pred)
        plt.axis('off')
        plt.title("YOLO Prediction")
        plt.show()