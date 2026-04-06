import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from safetensors.torch import load_file
from scipy.spatial import cKDTree
import torch.nn.functional as F

# Import your custom modules
# Make sure extract_coordinates_nms is available in your utils/loss file

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

def stitch_to_whole_slide(slice_predictions, distance_threshold=10.0):
    """Merges overlapping predictions using Global NMS."""
    global_points, global_scores = [], []
    
    for slice_data in slice_predictions:
        off_x, off_y = slice_data['offset_x'], slice_data['offset_y']
        for pt, score in zip(slice_data['points'], slice_data['scores']):
            global_points.append([pt[0] + off_x, pt[1] + off_y])
            global_scores.append(score)
            
    if not global_points:
        return []

    global_points = np.array(global_points)
    global_scores = np.array(global_scores)
    
    # Sort by highest confidence first
    sorted_indices = np.argsort(global_scores)[::-1]
    global_points = global_points[sorted_indices]
    global_scores = global_scores[sorted_indices]
    
    final_points = []
    active = np.ones(len(global_points), dtype=bool)
    tree = cKDTree(global_points)
    
    for i in range(len(global_points)):
        if not active[i]: continue
        final_points.append(global_points[i].tolist())
        # Suppress neighbors within the 10px threshold
        neighbors = tree.query_ball_point(global_points[i], distance_threshold)
        for n in neighbors:
            active[n] = False
            
    return final_points

def run_inference_pipeline(image_path, model_weights_path, output_path, patches_dir=None, device='cuda'):
    print(f"--- Starting H. pylori Inference Pipeline ---")
    
    # 1. Load the heavily trained Model
    print("Loading Model...")
    import model2
    model = model2.SRIntegratedPLIP(num_classes=1)
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval() # CRITICAL: Turn off dropout and batchnorm updates!

    # 2. Setup the exact normalization the model saw during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # 3. Load the massive original image
    print(f"Loading raw image: {image_path}")
    full_image = cv2.imread(image_path)
    if full_image is None:
        raise ValueError("Could not read the image. Check the path!")
    
    # Keep a pristine copy for drawing later
    draw_image = full_image.copy() 
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = full_image.shape
    
    # 4. Sliding Window Configuration
    patch_size = 224
    stride = 200 # 24 pixels of overlap to catch bacteria on the cut-lines
    
    slice_predictions = []
    batch_tensors = []
    batch_offsets = []
    batch_size = 64 # Adjust based on your DGX VRAM
    
    print(f"Slicing image ({img_w}x{img_h}) and running inference...")
    
    # Loop over the image grid
    with torch.no_grad(): # No gradients needed for inference!
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                
                # Cut the patch
                patch = full_image[y:y+patch_size, x:x+patch_size]
                patch_tensor = transform(patch)
                
                batch_tensors.append(patch_tensor)
                batch_offsets.append({'offset_x': x, 'offset_y': y})
                
                # If the batch is full, push it through the GPU
                if len(batch_tensors) == batch_size:
                    inputs = torch.stack(batch_tensors).to(device)
                    
                    # Forward pass (Ignore the SR output entirely!)
                    pred_heatmaps, _ = model(inputs)
                    pred_heatmaps_sig = torch.sigmoid(pred_heatmaps)
                    
                    # Extract local coordinates using downsample_factor=8
                    batch_results = extract_coordinates_nms(pred_heatmaps_sig, downsample_factor=8, conf_threshold=0.3)
                    
                    # Package them with their global offsets
                    for i, result in enumerate(batch_results):
                        # Assuming extract_coordinates_nms returns a list of dictionaries: [{'x': 10, 'y': 20, 'conf': 0.9}, ...]
                        pts = [[p['x'], p['y']] for p in result]
                        scores = [p['conf'] for p in result]
                        
                        slice_predictions.append({
                            'offset_x': batch_offsets[i]['offset_x'],
                            'offset_y': batch_offsets[i]['offset_y'],
                            'points': pts,
                            'scores': scores
                        })
                    
                    # Clear the batch
                    batch_tensors, batch_offsets = [], []

        # Process any leftover patches in the final batch
        if batch_tensors:
            inputs = torch.stack(batch_tensors).to(device)
            pred_heatmaps, _ = model(inputs)
            pred_heatmaps_sig = torch.sigmoid(pred_heatmaps)
            batch_results = extract_coordinates_nms(pred_heatmaps_sig, downsample_factor=8, conf_threshold=0.35)
            for i, result in enumerate(batch_results):
                pts = [[p['x'], p['y']] for p in result]
                scores = [p['conf'] for p in result]
                slice_predictions.append({
                    'offset_x': batch_offsets[i]['offset_x'],
                    'offset_y': batch_offsets[i]['offset_y'],
                    'points': pts,
                    'scores': scores
                })

    # 5. Global Stitching & Deduplication
    print("Stitching predictions back to the whole slide...")
    final_global_coordinates = stitch_to_whole_slide(slice_predictions, distance_threshold=10.0)
    
    saved_patches = []

    # 6. Visualization
    print("Drawing cyan prediction targets...")
    for idx, pt in enumerate(final_global_coordinates):
        gx, gy = int(pt[0]), int(pt[1])
        # Draw a beautiful Cyan circle with a 15px radius and 3px thickness
        cv2.circle(draw_image, (gx, gy), 15, (255, 255, 0), 3) 
        # Draw a tiny dot exactly in the center
        cv2.circle(draw_image, (gx, gy), 2, (0, 255, 0), -1)

    if patches_dir is not None:
        print("Extracting individual patches...")
        for idx, pt in enumerate(final_global_coordinates):
            gx, gy = int(pt[0]), int(pt[1])
            half_size = 112
            y1 = max(0, gy - half_size)
            y2 = min(img_h, gy + half_size)
            x1 = max(0, gx - half_size)
            x2 = min(img_w, gx + half_size)
            
            patch_crop = draw_image[y1:y2, x1:x2]
            patch_filename = f"patch_{idx}.jpg"
            patch_filepath = os.path.join(patches_dir, patch_filename)
            cv2.imwrite(patch_filepath, patch_crop)
            saved_patches.append(patch_filename)

    cv2.imwrite(output_path, draw_image)
    print(f"✅ Success! Found {len(final_global_coordinates)} bacteria.")
    print(f"✅ Annotated image saved to: {output_path}")

    return saved_patches

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Change these paths to match your DGX folders
    RAW_IMAGE = "./test_images/unseen_patient_slide.jpg"
    MODEL_WEIGHTS = "./saved_models/hpylori_best_model.safetensors"
    OUTPUT_IMAGE = "./test_images/annotated_result.jpg"
    
    run_inference_pipeline(RAW_IMAGE, MODEL_WEIGHTS, OUTPUT_IMAGE)