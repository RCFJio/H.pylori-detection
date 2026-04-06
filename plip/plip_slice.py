import os
import cv2
import glob
import random

def process_yolo_to_slices(image_dir, label_dir, output_dir, slice_size=224):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    pos_slices = []
    neg_slices = []
    
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) # Adjust extension if needed
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w, _ = img.shape
        
        # Parse YOLO labels into absolute center coordinates
        centers = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height
                        x_c, y_c = float(parts[1]), float(parts[2])
                        abs_x, abs_y = int(x_c * img_w), int(y_c * img_h)
                        centers.append((abs_x, abs_y))
        
        # Slide window across the image
        for y in range(0, img_h - slice_size + 1, slice_size):
            for x in range(0, img_w - slice_size + 1, slice_size):
                slice_img = img[y:y+slice_size, x:x+slice_size]
                
                # Find centers that fall within this slice
                slice_centers = []
                for (cx, cy) in centers:
                    if x <= cx < x + slice_size and y <= cy < y + slice_size:
                        # Adjust coordinates to be relative to the slice
                        slice_centers.append((cx - x, cy - y))
                
                slice_name = f"{base_name}_x{x}_y{y}"
                
                if len(slice_centers) > 0:
                    pos_slices.append((slice_name, slice_img, slice_centers))
                else:
                    neg_slices.append((slice_name, slice_img, []))
                    
    # Balance the dataset: match negative slice count to positive slice count
    random.shuffle(neg_slices)
    print(len(pos_slices))
    balanced_neg_slices = neg_slices[:len(pos_slices)]
    final_dataset = pos_slices + balanced_neg_slices
    
    # Save the balanced dataset
    for slice_name, slice_img, slice_centers in final_dataset:
        cv2.imwrite(os.path.join(output_dir, 'images', f"{slice_name}.jpg"), slice_img)
        
        # Save points for the Dataset loader (only if it has targets)
        if slice_centers:
            with open(os.path.join(output_dir, 'labels', f"{slice_name}.txt"), 'w') as f:
                for cx, cy in slice_centers:
                    f.write(f"{cx} {cy}\n")

import os
import cv2
import glob
import random

def process_yolo_to_slices_overlap(image_dir, label_dir, output_dir, slice_size=224, overlap_ratio=0.20):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    pos_slices = []
    neg_slices = []
    
    # Calculate the stride (step size) based on the overlap ratio
    # For a 224 slice with 20% overlap, stride will be int(224 * 0.8) = 179 pixels
    stride = int(slice_size * (1.0 - overlap_ratio))
    print(f"Slicing with window: {slice_size}x{slice_size} | Stride: {stride}px | Overlap: {slice_size - stride}px")
    
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) # Adjust extension if needed
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w, _ = img.shape
        
        # Parse YOLO labels into absolute center coordinates
        centers = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height
                        x_c, y_c = float(parts[1]), float(parts[2])
                        abs_x, abs_y = int(x_c * img_w), int(y_c * img_h)
                        centers.append((abs_x, abs_y))
        
        # Slide window across the image using the STRIDE instead of slice_size
        for y in range(0, img_h - slice_size + 1, stride):
            for x in range(0, img_w - slice_size + 1, stride):
                slice_img = img[y:y+slice_size, x:x+slice_size]
                
                # Find centers that fall within this specific overlapping slice
                slice_centers = []
                for (cx, cy) in centers:
                    if x <= cx < x + slice_size and y <= cy < y + slice_size:
                        # Adjust coordinates to be relative to the slice
                        slice_centers.append((cx - x, cy - y))
                
                slice_name = f"{base_name}_x{x}_y{y}"
                
                if len(slice_centers) > 0:
                    pos_slices.append((slice_name, slice_img, slice_centers))
                else:
                    neg_slices.append((slice_name, slice_img, []))
                    
    # Balance the dataset: match negative slice count to positive slice count
    random.shuffle(neg_slices)
    print(f"Total Positive Slices Generated: {len(pos_slices)}")
    
    balanced_neg_slices = neg_slices[:len(pos_slices)]
    final_dataset = pos_slices + balanced_neg_slices
    
    # Save the balanced dataset
    for slice_name, slice_img, slice_centers in final_dataset:
        cv2.imwrite(os.path.join(output_dir, 'images', f"{slice_name}.jpg"), slice_img)
        
        # Save points for the Dataset loader (only if it has targets)
        if slice_centers:
            with open(os.path.join(output_dir, 'labels', f"{slice_name}.txt"), 'w') as f:
                for cx, cy in slice_centers:
                    f.write(f"{cx} {cy}\n")

# Execute the script
process_yolo_to_slices_overlap('/raid/Colleges/gectcr/CA/Faculty/ktu-f41961/H.pylori-detection/datazip/merged_dataset/images', '/raid/Colleges/gectcr/CA/Faculty/ktu-f41961/H.pylori-detection/datazip/merged_dataset/labels', './balanced_slices',slice_size=224,overlap_ratio=0.20)
