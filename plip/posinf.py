import os
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import glob

import model

def extract_coordinates_nms(heatmaps, downsample_factor=8, conf_threshold=0.3, kernel_size=3):
    if heatmaps.max() > 1.0 or heatmaps.min() < 0.0:
        heatmaps = torch.sigmoid(heatmaps)
        
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmaps, kernel_size=kernel_size, stride=1, padding=pad)
    peaks = (heatmaps == hmax).float() * heatmaps
    
    batch_coords = []
    for b in range(heatmaps.shape[0]):
        peak_map = peaks[b, 0] 
        y_coords, x_coords = torch.where(peak_map > conf_threshold)
        scores = peak_map[y_coords, x_coords]
        
        coords = [{"x": int(x.item() * downsample_factor), 
                   "y": int(y.item() * downsample_factor), 
                   "confidence": score.item()} 
                  for y, x, score in zip(y_coords, x_coords, scores)]
        batch_coords.append(coords)
    return batch_coords

def visualize_positive_results(model,model_path, image_dir, label_dir, output_dir, num_samples=5, device='cuda', conf_thresh=0.35):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading weights from: {model_path}")
    model.load_state_dict(load_file(model_path))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    all_images = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    if not all_images:
        print("No images found in the specified directory!")
        return
        
    # --- NEW: THE POSITIVE SLICE FILTER ---
    positive_images = []
    print("Filtering dataset for slices containing H. pylori...")
    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Check if label file exists and has at least one line of data
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                if len(f.readlines()) > 0:
                    positive_images.append(img_path)
                    
    print(f"Found {len(positive_images)} positive slices out of {len(all_images)} total.")
    
    if len(positive_images) == 0:
        print("No positive slices found to visualize!")
        return
        
    # Sample ONLY from the guaranteed positive images
    sample_images = random.sample(positive_images, min(num_samples, len(positive_images)))
    
    print(f"Generating visualizations for {len(sample_images)} positive slices...")
    
    for idx, img_path in enumerate(sample_images):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        raw_img = Image.open(img_path).convert("RGB")
        raw_img = raw_img.resize((224, 224))
        
        input_tensor = transform(raw_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_heatmaps, _ = model(input_tensor)
            
        pred_coords = extract_coordinates_nms(pred_heatmaps, downsample_factor=8, conf_threshold=conf_thresh)[0]
        
        # 4. Load Ground Truth Labels (Custom Slice Format)
        gt_coords = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Our custom slicing script saves exactly 2 values: absolute X and absolute Y
                if len(parts) >= 2: 
                    # No need to multiply by 224, they are already absolute pixel coordinates!
                    x_c, y_c = float(parts[0]), float(parts[1])
                    gt_coords.append((int(x_c), int(y_c)))
                        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Image: {filename}", fontsize=16)
        
        # Ground Truth
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Ground Truth ({len(gt_coords)} Bacteria)")
        axes[0].axis('off')
        for (gx, gy) in gt_coords:
            circle = plt.Circle((gx, gy), radius=5, color='lime', fill=False, linewidth=2)
            axes[0].add_patch(circle)
            axes[0].plot(gx, gy, 'g+', markersize=8)
            
        # Prediction
        axes[1].imshow(raw_img)
        axes[1].set_title(f"Model Prediction ({len(pred_coords)} Detected | Thresh: {conf_thresh})")
        axes[1].axis('off')
        for pred in pred_coords:
            px, py, conf = pred['x'], pred['y'], pred['confidence']
            circle = plt.Circle((px, py), radius=5, color='cyan', fill=False, linewidth=2)
            axes[1].add_patch(circle)
            axes[1].plot(px, py, 'c+', markersize=8)
            axes[1].text(px + 7, py - 7, f"{conf:.2f}", color='cyan', fontsize=9, weight='bold')
            
        save_path = os.path.join(output_dir, f"compare_{base_name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        
    print(f"Done! Check the '{output_dir}' folder.")

# Execute
model=model.SRIntegratedPLIP(num_classes=1)
visualize_positive_results(
     model,
     model_path="./saved_models/hpylori_best_model.safetensors",
     image_dir="./balanced_slices/images", 
     label_dir="./balanced_slices/labels", 
     output_dir="./visual_results_pos",        
     num_samples=100,                       
     conf_thresh=0.1                      
)