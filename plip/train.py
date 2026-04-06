import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
from safetensors.torch import save_file

# Import your custom modules
import model2
import loss
import augment

# ==========================================
#        mAP EVALUATION FUNCTIONS
# ==========================================
def extract_coordinates_nms(heatmaps, downsample_factor=8, conf_threshold=0.1, kernel_size=3):
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

def calculate_distance_ap(ground_truths, predictions, distance_threshold=10.0):
    if len(ground_truths) == 0 and len(predictions) == 0: return 1.0
    if len(predictions) == 0: return 0.0

    preds_sorted = sorted(predictions, key=lambda k: k['confidence'], reverse=True)
    matched_gt = set()
    TP, FP = np.zeros(len(preds_sorted)), np.zeros(len(preds_sorted))
    
    for i, pred in enumerate(preds_sorted):
        best_dist, best_gt_idx = float('inf'), -1
        for j, gt in enumerate(ground_truths):
            if j in matched_gt: continue
            dist = math.hypot(pred['x'] - gt['x'], pred['y'] - gt['y'])
            if dist < best_dist:
                best_dist, best_gt_idx = dist, j
                
        if best_dist <= distance_threshold:
            TP[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            FP[i] = 1
            
    cum_TP, cum_FP = np.cumsum(TP), np.cumsum(FP)
    total_gt = len(ground_truths)
    recalls = cum_TP / total_gt if total_gt > 0 else np.zeros_like(cum_TP)
    precisions = cum_TP / (cum_TP + cum_FP)
    
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    return np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])


# ==========================================
#          MAIN TRAINING LOOP
# ==========================================
def train_sr_integrated_model(model, full_dataset, epochs=100, batch_size=32, device='cuda', save_dir='./saved_models'):
    
    # 1. Split the dataset (90% Train, 10% Validation)
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    # 2. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    l1_loss_fn = nn.L1Loss()
    lambda_sr = 0.1 
    
    # --- NEW: Checkpointing setup for mAP ---
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "hpylori_best_model.safetensors")
    best_val_map = 0.0  # Start at 0% mAP
    
    for epoch in range(epochs):
        # ==========================================
        #               TRAINING PHASE
        # ==========================================
        model.train()
        train_loss, train_det_loss, train_sr_loss = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        for inputs, target_heatmaps, target_sr_imgs in pbar:
            inputs = inputs.to(device)
            target_heatmaps = target_heatmaps.to(device)
            target_sr_imgs = target_sr_imgs.to(device)
            
            optimizer.zero_grad()
            pred_heatmaps, pred_sr_imgs = model(inputs)
            
            pred_heatmaps_sig = torch.sigmoid(pred_heatmaps)
            l_det = loss.centernet_focal_loss(pred_heatmaps_sig, target_heatmaps) # Update namespace if needed
            
            target_sr_imgs_resized = F.interpolate(target_sr_imgs, size=pred_sr_imgs.shape[2:], mode='bilinear', align_corners=False)
            l_sr = l1_loss_fn(pred_sr_imgs, target_sr_imgs_resized)
            
            total_loss = l_det + (lambda_sr * l_sr)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_det_loss += l_det.item()
            train_sr_loss += l_sr.item()
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
            
        # ==========================================
        #              VALIDATION PHASE
        # ==========================================
        model.eval()
        val_loss = 0
        epoch_aps = [] # --- NEW: List to track APs for this epoch ---
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]")
            for inputs, target_heatmaps, target_sr_imgs in pbar_val:
                inputs = inputs.to(device)
                target_heatmaps = target_heatmaps.to(device)
                target_sr_imgs = target_sr_imgs.to(device)
                
                pred_heatmaps, pred_sr_imgs = model(inputs)
                
                # Still calculate loss for logging purposes
                pred_heatmaps_sig = torch.sigmoid(pred_heatmaps)
                l_det = loss.centernet_focal_loss(pred_heatmaps_sig, target_heatmaps)
                target_sr_imgs_resized = F.interpolate(target_sr_imgs, size=pred_sr_imgs.shape[2:], mode='bilinear', align_corners=False)
                l_sr = l1_loss_fn(pred_sr_imgs, target_sr_imgs_resized)
                total_loss = l_det + (lambda_sr * l_sr)
                val_loss += total_loss.item()
                
                # --- NEW: Extract coordinates and calculate AP ---
                batch_pred_coords = extract_coordinates_nms(pred_heatmaps, downsample_factor=8, conf_threshold=0.3)
                batch_gt_coords = extract_coordinates_nms(target_heatmaps, downsample_factor=8, conf_threshold=0.1)
                
                for i in range(len(inputs)):
                    ap = calculate_distance_ap(batch_gt_coords[i], batch_pred_coords[i], distance_threshold=5.0)
                    epoch_aps.append(ap)
                
                pbar_val.set_postfix({'Val_Loss': f"{total_loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        current_val_map = np.mean(epoch_aps) # --- NEW: Final mAP score for the epoch ---
        
        # Summary Print
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train -> Loss: {avg_train_loss:.4f}")
        print(f"  Val   -> Loss: {avg_val_loss:.4f} | mAP@50px: {current_val_map:.4f} ({(current_val_map*100):.2f}%)")
        
        # ==========================================
        #              CHECKPOINTING
        # ==========================================
        # --- NEW: Save based on mAP instead of loss ---
        if current_val_map > best_val_map:
            print(f"🌟 New best mAP! ({best_val_map:.4f} --> {current_val_map:.4f}). Saving model...")
            best_val_map = current_val_map
            
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            save_file(state_dict, best_model_path)
            
    print(f"\nTraining Complete! The best model was saved with a mAP of {best_val_map:.4f}")
    

train_dataset = augment.AugmentedHistopathologyDataset(slice_dir='./balanced_slices', downsample_factor=8)
model = model2.SRIntegratedPLIP(num_classes=1)
train_sr_integrated_model(model, train_dataset, batch_size=8)