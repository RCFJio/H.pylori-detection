import os
import math
import zipfile
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
from ultralytics import YOLO
from super_image import EdsrModel, ImageLoader

# --- Configuration & Paths ---
# Adjust these paths if running locally instead of in a Kaggle environment
IN_20X       = "1b.jpg"                        # Input 20× image
YOLO_WEIGHTS = "last(dabest)220(2-4).pt"      # 40×-trained YOLO weights
TILE_20X     = 640                             # Tile size at 20×
OVERLAP      = 0.25                            # 25% overlap between tiles
IMG_SZ_40X   = 1280                            # YOLO inference size
CONF, IOU    = 0.10, 0.45                      # Detection thresholds

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utility Functions ---
def download_resources():
    """Downloads dataset and model weights using gdown."""
    # Dataset (40x.zip)
    os.system("gdown --fuzzy 'https://drive.google.com/file/d/19uzLovYrJ_Q0squYfPuKGRo_8m8H6rhl/view?usp=sharing'")
    # Example Images
    os.system("gdown --fuzzy 'https://drive.google.com/file/d/1eTqu23L0eJ4VNZvB95lpivYBqYT13oF4/view?usp=sharing'")
    os.system("gdown --fuzzy 'https://drive.google.com/file/d/1RqW4k9r4bfChNyG08hQXVotkIuEOEaxU/view?usp=sharing'")
    # YOLO Model Weights
    os.system("gdown --fuzzy 'https://drive.google.com/file/d/1kojPlfQqfP7JZqehnnLiuDyICIhOxcbU/view?usp=sharing'")

def silent_unzip(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def sliding_windows_fullcover(W: int, H: int, tile: int, overlap: float):
    """Yield (x0, y0, x1, y1) windows that fully cover the image."""
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if not xs or xs[-1] != W - tile: xs.append(W - tile)
    if not ys or ys[-1] != H - tile: ys.append(H - tile)
    for y0 in ys:
        for x0 in xs:
            yield (x0, y0, min(x0 + tile, W), min(y0 + tile, H))

def run_patchwise_sr_yolo(in_path, edsr_model, yolo_model, tile_20x=512, overlap=0.25):
    """Runs Super-Resolution on patches, detects with YOLO, and maps back to original coords."""
    base_img = Image.open(in_path).convert("RGB")
    W, H = base_img.size
    detections_20x = []
    tile_coords = []
    
    for (x0, y0, x1, y1) in sliding_windows_fullcover(W, H, tile_20x, overlap):
        tile_coords.append((x0, y0, x1, y1))
        crop_20x = base_img.crop((x0, y0, x1, y1))

        # Super-Resolution (x2 upscale)
        with torch.no_grad():
            inputs = ImageLoader.load_image(crop_20x).to(device)
            sr_tensor = edsr_model(inputs).detach().cpu()
        
        sr = sr_tensor.squeeze(0).clamp(0, 1)
        sr_np = (sr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

        # YOLO Detection on the SR tile
        res = yolo_model.predict(source=sr_bgr, imgsz=IMG_SZ_40X, conf=CONF, iou=IOU, verbose=False)[0]

        if res.boxes is not None:
            for box, confv, clsid in zip(res.boxes.xyxy.cpu().numpy(),
                                         res.boxes.conf.cpu().numpy(),
                                         res.boxes.cls.cpu().numpy()):
                # Map SR (40x) coords back to Original (20x) coords
                gx1, gy1 = (box[0] / 2) + x0, (box[1] / 2) + y0
                gx2, gy2 = (box[2] / 2) + x0, (box[3] / 2) + y0
                detections_20x.append([gx1, gy1, gx2, gy2, float(confv), int(clsid)])

    return np.array(detections_20x, dtype=np.float32), tile_coords

def visualize_results(img_path, detections, tile_coords):
    """Draws detections and tiling grid for visualization."""
    # Draw detections
    im_det = cv2.imread(img_path)
    for (x1, y1, x2, y2, conf, clsid) in detections:
        cv2.rectangle(im_det, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Draw tile boundaries
    im_tiles = cv2.imread(img_path)
    for (x0, y0, x1, y1) in tile_coords:
        cv2.rectangle(im_tiles, (x0, y0), (x1, y1), (0, 0, 255), 2)

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(im_det, cv2.COLOR_BGR2RGB)); plt.title("Detections")
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(im_tiles, cv2.COLOR_BGR2RGB)); plt.title("Tiles")
    plt.show()

if __name__ == "__main__":
    # 1. Setup environment
    # download_resources() # Uncomment if you need to download files
    if os.path.exists('40x.zip'):
        silent_unzip('40x.zip', 'yolo')
    
    # 2. Initialize Models
    print("Loading models...")
    edsr = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2).to(device).eval()
    yolo_model = YOLO(YOLO_WEIGHTS)

    # 3. Run Pipeline
    test_img = "yolo/21.11.2025/0001.jpg" # Example path from notebook
    if os.path.exists(test_img):
        print(f"Processing {test_img}...")
        results, tiles = run_patchwise_sr_yolo(test_img, edsr, yolo_model, TILE_20X, OVERLAP)
        print(f"Detected {len(results)} objects across {len(tiles)} tiles.")
        
        # 4. Visualize
        visualize_results(test_img, results, tiles)
    else:
        print(f"Test image not found at {test_img}")