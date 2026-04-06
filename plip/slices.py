import os
import cv2
import glob
import random
import numpy as np

def contains_tissue(patch, white_threshold=220, tissue_ratio=0.3):
    """Checks if the patch contains actual tissue and not just blank glass."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    tissue_pixels = np.sum(gray < white_threshold)
    total_pixels = patch.shape[0] * patch.shape[1]
    return (tissue_pixels / total_pixels) >= tissue_ratio

def build_balanced_dataset(pos_image_dir, label_dir, neg_image_dir, output_dir, slice_size=224, overlap_ratio=0.20):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    pos_slices = []
    neg_slices = []
    
    # Calculate the stride based on the overlap ratio
    stride = int(slice_size * (1.0 - overlap_ratio))
    print(f"Slicing with window: {slice_size}x{slice_size} | Stride: {stride}px")
    
    # ==========================================
    # 1. EXTRACT PURE POSITIVE SLICES
    # ==========================================
    print("\n--- Extracting Positive Slices ---")
    pos_image_paths = glob.glob(os.path.join(pos_image_dir, '*.jpg')) # Adjust extension if needed
    
    for img_path in pos_image_paths:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Skip if there is no annotation file for this image
        if not os.path.exists(label_path): continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w, _ = img.shape
        
        # Parse YOLO labels
        centers = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_c, y_c = float(parts[1]), float(parts[2])
                    abs_x, abs_y = int(x_c * img_w), int(y_c * img_h)
                    centers.append((abs_x, abs_y))
                    
        # Slide window across the infected image
        for y in range(0, img_h - slice_size + 1, stride):
            for x in range(0, img_w - slice_size + 1, stride):
                slice_centers = []
                for (cx, cy) in centers:
                    if x <= cx < x + slice_size and y <= cy < y + slice_size:
                        slice_centers.append((cx - x, cy - y))
                
                # ONLY save it if bacteria are present!
                if len(slice_centers) > 0:
                    slice_img = img[y:y+slice_size, x:x+slice_size]
                    slice_name = f"pos_{base_name}_x{x}_y{y}"
                    pos_slices.append((slice_name, slice_img, slice_centers))

    print(f"Total Positive Slices Generated: {len(pos_slices)}")

    # ==========================================
    # 2. EXTRACT PURE NEGATIVE SLICES
    # ==========================================
    print("\n--- Extracting Negative Slices ---")
    neg_image_paths = glob.glob(os.path.join(neg_image_dir, '*.jpg'))
    
    for img_path in neg_image_paths:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w, _ = img.shape
        
        # Slide window across the purely negative image
        for y in range(0, img_h - slice_size + 1, stride):
            for x in range(0, img_w - slice_size + 1, stride):
                slice_img = img[y:y+slice_size, x:x+slice_size]
                
                # ONLY save the negative patch if it actually contains tissue
                if contains_tissue(slice_img):
                    slice_name = f"neg_{base_name}_x{x}_y{y}"
                    neg_slices.append((slice_name, slice_img, []))

    print(f"Total Candidate Negative Slices Found: {len(neg_slices)}")

    # ==========================================
    # 3. BALANCE AND SAVE
    # ==========================================
    print("\n--- Balancing and Saving Dataset ---")
    # Shuffle the negatives so we get a random distribution across all patients
    random.shuffle(neg_slices)
    
    # Match the negative count to the positive count
    target_neg_count = min(len(pos_slices), len(neg_slices))
    balanced_neg_slices = neg_slices[:target_neg_count]
    
    # Combine and shuffle the final dataset
    final_dataset = pos_slices + balanced_neg_slices
    random.shuffle(final_dataset)
    
    for slice_name, slice_img, slice_centers in final_dataset:
        # Save the image
        cv2.imwrite(os.path.join(output_dir, 'images', f"{slice_name}.jpg"), slice_img)
        
        # Only save a .txt file if it is a positive slice (as we discussed!)
        if slice_centers:
            with open(os.path.join(output_dir, 'labels', f"{slice_name}.txt"), 'w') as f:
                for cx, cy in slice_centers:
                    f.write(f"{cx} {cy}\n")

    print(f"✅ Finished! Dataset created with {len(pos_slices)} Positive and {target_neg_count} Negative slices.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Folder containing your infected images
    POS_IMAGES = '/raid/Colleges/gectcr/CA/Faculty/ktu-f41961/H.pylori-detection/datazip/merged_dataset/images'
    
    # 2. Folder containing the YOLO labels for those infected images
    POS_LABELS = '/raid/Colleges/gectcr/CA/Faculty/ktu-f41961/H.pylori-detection/datazip/merged_dataset/labels'
    
    # 3. NEW: Folder containing your completely healthy/negative WSIs (Update this path!)
    NEG_IMAGES = '/raid/Colleges/gectcr/CA/Faculty/ktu-f41961/H.pylori-detection/datazip/negative_dataset/images'
    
    # 4. Where the final dataset will go
    OUTPUT_DIR = './balanced_slices'
    
    build_balanced_dataset(POS_IMAGES, POS_LABELS, NEG_IMAGES, OUTPUT_DIR, slice_size=224, overlap_ratio=0.20)