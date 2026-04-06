import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
# dataset augmentation
class AugmentedHistopathologyDataset(Dataset):
    def __init__(self, slice_dir, downsample_factor=4, is_training=True):
        self.image_dir = os.path.join(slice_dir, 'images')
        self.label_dir = os.path.join(slice_dir, 'labels')
        self.image_paths = glob.glob(os.path.join(self.image_dir, '*.jpg'))
        self.downsample_factor = downsample_factor
        
        # 1. Define the Joint Augmentation Pipeline
        if is_training:
            self.transform = A.Compose([
                # Spatial transformations (rotations/flips)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Stain variation simulation (crucial for medical imaging)
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.7),
                
                # Standard PLIP/CLIP Normalization
                A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                            std=[0.26862954, 0.26130258, 0.27577711]),
                ToTensorV2() # Converts numpy array to PyTorch tensor shape [C, H, W]
            ], 
            # 2. Tell Albumentations to treat our points as spatial coordinates
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
        else:
            # Validation/Testing pipeline (no random augmentations)
            self.transform = A.Compose([
                A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                            std=[0.26862954, 0.26130258, 0.27577711]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    def __len__(self):
        return len(self.image_paths)

    def _generate_gaussian_heatmap(self, target_size, points, sigma=1.0):
        # (Same Gaussian generation logic as before)
        h, w = target_size
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for cx, cy in points:
            scaled_cx = int(cx / self.downsample_factor)
            scaled_cy = int(cy / self.downsample_factor)
            
            if scaled_cx >= w or scaled_cy >= h or scaled_cx < 0 or scaled_cy < 0:
                continue
                
            radius = int(3 * sigma)
            x_min, x_max = max(0, scaled_cx - radius), min(w, scaled_cx + radius + 1)
            y_min, y_max = max(0, scaled_cy - radius), min(h, scaled_cy + radius + 1)
            
            x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            gaussian = np.exp(-((x_grid - scaled_cx)**2 + (y_grid - scaled_cy)**2) / (2 * sigma**2))
            heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], gaussian)
            
        return torch.from_numpy(heatmap).unsqueeze(0)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        
        # 3. Load image with OpenCV and convert BGR to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load center points
        points = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cx, cy = map(float, line.strip().split())
                    points.append([cx, cy]) # Albumentations expects lists or numpy arrays
        
        # 4. Apply transformations simultaneously
        transformed = self.transform(image=image, keypoints=points)
        img_tensor = transformed['image']
        transformed_points = transformed['keypoints']
        
        # Generate the heatmap using the newly augmented points
        target_size = (224 // self.downsample_factor, 224 // self.downsample_factor)
        heatmap_tensor = self._generate_gaussian_heatmap(target_size, transformed_points, sigma=1.0)
        
        return img_tensor, heatmap_tensor, img_tensor
