import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
from safetensors.torch import save_file
import augment
import loss
import model
# Make sure extract_coordinates_nms and calculate_distance_ap are defined above this!

# ==========================================
#          MAIN TRAINING LOOP
# ==========================================
# Notice the signature changed: it now expects BOTH dataset instances
def train_sr_integrated_model(model, full_train_dataset, full_val_dataset, epochs=100, batch_size=32, device='cuda', save_dir='./saved_models'):
    
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
    
    # --- UNFREEZE UPDATE ---
    # Pushed to Epoch 15 to let the heads stabilize before fine-tuning the ViT
    unfreeze_epoch = 15 
    layers_to_unfreeze = ['encoder.layers.10', 'encoder.layers.11', 'post_layernorm'] 
    
    l1_loss_fn = nn.L1Loss()
    lambda_sr = 0.1 
    
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "hpylori_best_model.safetensors")
    best_val_map = 0.0  
    
    for epoch in range(epochs):
        
        # --- SELECTIVE UNFREEZING TRIGGER ---
        if epoch == unfreeze_epoch:
            print(f"\n🔓 Epoch {epoch+1}: Surgically unfreezing the final layers of the PLIP backbone!")
            
            unfrozen_params = []
            backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
            
            for name, param in backbone.named_parameters():
                if any(target_layer in name for target_layer in layers_to_unfreeze):
                    param.requires_grad = True
                    unfrozen_params.append(param)
            
            if unfrozen_params:
                optimizer.add_param_group({
                    'params': unfrozen_params,
                    'lr': 1e-5  
                })
                print(f"Successfully unlocked {len(unfrozen_params)} parameter tensors for deep fine-tuning.")

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
model = model.SRIntegratedPLIP(num_classes=1)

# 4. Train!
train_sr_integrated_model(
    model=model, 
    full_train_dataset=full_train, 
    full_val_dataset=full_val, 
    batch_size=32  # Drop this if your DGX VRAM complains!
)