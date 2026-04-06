import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
from safetensors.torch import save_file
import math
import loss
import augment
import model # Assuming your file is model.py

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
def train_sr_integrated_model(model, full_train_dataset, full_val_dataset, epochs=100, batch_size=32, device='cuda', save_dir='./saved_models_unfreeze'):
    
    # 1. Secure Dataset Splitting (The Subset Strategy)
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    
    # Lock the seed so the train/val split is identical every time you run this
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(np.floor(0.9 * dataset_size))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    # Assign indices to their respective instances
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    
    print(f"Dataset securely split: {len(train_dataset)} Train (Augmented) | {len(val_dataset)} Val (Clean)")

    # 2. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # --- INITIAL FREEZING STRATEGY ---
    for param in model.module.backbone.parameters() if isinstance(model, nn.DataParallel) else model.backbone.parameters():
        param.requires_grad = False
        
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    
    # --- OPTIMIZER UPDATE ---
    # weight_decay bumped to 1e-2 to aggressively combat ViT overfitting
    optimizer = AdamW(head_params, lr=1e-4, weight_decay=1e-2)
    
    # --- NEW: PROGRESSIVE UNFREEZING SCHEDULE ---
    # { Epoch_To_Trigger : [List of layers to unfreeze] }
    unfreeze_schedule = {
        10: ['encoder.layers.10', 'encoder.layers.11', 'post_layernorm'],
        25: ['encoder.layers.8', 'encoder.layers.9'],
        40: ['encoder.layers.6', 'encoder.layers.7']
    }
    
    l1_loss_fn = nn.L1Loss()
    lambda_sr = 0.1 
    
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "hpylori_best_model.safetensors")
    best_val_map = 0.0  
    
    for epoch in range(epochs):
        
        # --- NEW: DYNAMIC PROGRESSIVE UNFREEZE TRIGGER ---
        if epoch in unfreeze_schedule:
            target_layers = unfreeze_schedule[epoch]
            print(f"\n🔓 Epoch {epoch+1}: PROGRESSIVE UNFREEZE TRIGGERED!")
            print(f"Surgically unlocking layers: {target_layers}")
            
            unfrozen_params = []
            backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
            
            for name, param in backbone.named_parameters():
                if any(target in name for target in target_layers):
                    param.requires_grad = True
                    unfrozen_params.append(param)
            
            if unfrozen_params:
                # Inject the newly unfrozen layers into the optimizer with a tiny learning rate
                optimizer.add_param_group({
                    'params': unfrozen_params,
                    'lr': 1e-5  
                })
                print(f"Successfully added {len(unfrozen_params)} parameter tensors to the optimizer for deep fine-tuning.")

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
            l_det = loss.centernet_focal_loss(pred_heatmaps_sig, target_heatmaps)
            
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
        epoch_aps = [] 
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]")
            for inputs, target_heatmaps, target_sr_imgs in pbar_val:
                inputs = inputs.to(device)
                target_heatmaps = target_heatmaps.to(device)
                target_sr_imgs = target_sr_imgs.to(device)
                
                pred_heatmaps, pred_sr_imgs = model(inputs)
                
                pred_heatmaps_sig = torch.sigmoid(pred_heatmaps)
                l_det = loss.centernet_focal_loss(pred_heatmaps_sig, target_heatmaps)
                target_sr_imgs_resized = F.interpolate(target_sr_imgs, size=pred_sr_imgs.shape[2:], mode='bilinear', align_corners=False)
                l_sr = l1_loss_fn(pred_sr_imgs, target_sr_imgs_resized)
                total_loss = l_det + (lambda_sr * l_sr)
                val_loss += total_loss.item()
                
                batch_pred_coords = extract_coordinates_nms(pred_heatmaps, downsample_factor=4, conf_threshold=0.3)
                batch_gt_coords = extract_coordinates_nms(target_heatmaps, downsample_factor=4, conf_threshold=0.1)
                
                for i in range(len(inputs)):
                    ap = calculate_distance_ap(batch_gt_coords[i], batch_pred_coords[i], distance_threshold=10.0)
                    epoch_aps.append(ap)
                
                pbar_val.set_postfix({'Val_Loss': f"{total_loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        current_val_map = np.mean(epoch_aps) 
        
        # Summary Print
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train -> Loss: {avg_train_loss:.4f}")
        print(f"  Val   -> Loss: {avg_val_loss:.4f} | mAP@10px: {current_val_map:.4f} ({(current_val_map*100):.2f}%)")
        
        # ==========================================
        #              CHECKPOINTING
        # ==========================================
        if current_val_map > best_val_map:
            print(f"🌟 New best mAP! ({best_val_map:.4f} --> {current_val_map:.4f}). Saving model...")
            best_val_map = current_val_map
            
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            save_file(state_dict, best_model_path)
            
    print(f"\nTraining Complete! The best model was saved with a mAP of {best_val_map:.4f}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Init the Augmented version for Training
    full_train = augment.AugmentedHistopathologyDataset(
        slice_dir='./balanced_slices', 
        downsample_factor=4, 
        is_training=True
    )

    # 2. Init the Clean version for Validation
    full_val = augment.AugmentedHistopathologyDataset(
        slice_dir='./balanced_slices', 
        downsample_factor=4, 
        is_training=False
    )

    # 3. Load Model
    model_instance = model.SRIntegratedPLIP(num_classes=1)

    # 4. Train!
    train_sr_integrated_model(
        model=model_instance, 
        full_train_dataset=full_train, 
        full_val_dataset=full_val, 
        batch_size=8  # Drop this if your DGX VRAM complains!
    )