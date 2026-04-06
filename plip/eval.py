import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HPyloriEvaluationDataset(Dataset):
    def __init__(self, slice_dir, downsample_factor=8):
        """
        Loads images and YOLO labels, converting them into PLIP normalized tensors 
        and CenterNet spatial heatmaps.
        """
        self.slice_dir = slice_dir
        self.downsample_factor = downsample_factor
        self.image_size = 224
        
        # Because PLIP shrinks 224x224 by a factor of 8, the heatmap is 28x28
        self.heatmap_size = self.image_size // downsample_factor 
        
        # Grab all images in the directory
        self.image_paths = glob.glob(os.path.join(slice_dir, "*.png")) + \
                           glob.glob(os.path.join(slice_dir, "*.jpg"))
                           
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {slice_dir}!")

        # PLIP / CLIP Normalization pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def _generate_heatmap(self, yolo_coords):
        """
        The Mathematical Bridge: Converts a list of YOLO [x, y] center points 
        into a 2D Gaussian probability map.
        """
        # Start with a completely black 28x28 grid
        heatmap = np.zeros((1, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        # Sigma controls the "radius" of the peak. 
        # 1.0 is perfect for tiny 4-pixel bacteria in a 28x28 space.
        sigma = 1.0 
        
        for coord in yolo_coords:
            # YOLO coordinates are normalized [0.0 to 1.0]. 
            # Multiply by the grid size to get the exact pixel location on the 28x28 map
            x_center = coord[0] * self.heatmap_size
            y_center = coord[1] * self.heatmap_size
            
            # Generate the X and Y grid coordinates
            y, x = np.ogrid[0:self.heatmap_size, 0:self.heatmap_size]
            
            # The 2D Gaussian function
            h = np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma**2))
            
            # Use np.maximum to blend overlapping bacteria peaks smoothly
            heatmap[0] = np.maximum(heatmap[0], h)
            
        return torch.from_numpy(heatmap)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 1. Load and transform the image
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(image)
        
        # 2. Find the corresponding YOLO .txt file
        # Assumes the text file has the exact same name as the image but ends in .txt
        base_name = os.path.splitext(img_path)[0]
        txt_path = base_name + ".txt"
        
        yolo_coords = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # YOLO format: <class> <x_center> <y_center> <width> <height>
                        # We only extract x_center and y_center as floats
                        x_c, y_c = float(parts[1]), float(parts[2])
                        yolo_coords.append((x_c, y_c))
                        
        # 3. Paint the targets
        heatmap_tensor = self._generate_heatmap(yolo_coords)
        
        filename = os.path.basename(img_path)
        
        # Returns inputs, target_heatmaps, and filename (matching our evaluation loop structure)
        return img_tensor, heatmap_tensor, filename