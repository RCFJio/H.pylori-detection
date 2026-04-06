import torch
import torch.nn as nn
import math
from transformers import CLIPVisionModel
#model
class SRIntegratedPLIP(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        # 1. Foundation Model (Backbone)
        # We load the vision encoder of the PLIP model from Hugging Face
        self.backbone = CLIPVisionModel.from_pretrained("vinid/plip")
        
        # Freeze the PLIP backbone to retain its histopathology pre-training (optional but recommended initially)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # PLIP's ViT-Base hidden dimension is 768
        hidden_dim = 768 
        
        # 2. Super-Resolution Branch (The Enhancer)
        # Using PixelShuffle to upsample the feature maps. 
        # PLIP uses a 16x16 patch size, resulting in a 14x14 grid for 224x224 images.
        # An upscale_factor of 4 brings the spatial features up to 56x56 resolution.
        upscale_factor = 4
        self.sr_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 64 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True),
            # Final reconstruction layer for the SR loss (3 channels for RGB)
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        
        # 3. Detection Branch (The Predictor)
        # Takes the 56x56 super-resolved features to predict the CenterNet heatmap
        self.det_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.det_relu = nn.ReLU(inplace=True)
        self.det_head = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Extract features from the PLIP backbone
        # output_hidden_states=True forces the model to return features from all transformer layers
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        
        # Grab the hidden states from the second-to-last layer
        # This avoids the final layer norm/projection which can over-smooth spatial details
        # Shape: [Batch, Seq_Len, Hidden_Dim] (e.g., [B, 197, 768])
        hidden_states = outputs.hidden_states[-2] 
        
        # Remove the CLS token (the first token in the sequence)
        # Shape becomes [B, 196, 768]
        patch_tokens = hidden_states[:, 1:, :] 
        
        # Dynamically calculate the 2D grid size (e.g., sqrt(196) = 14)
        B, N, C = patch_tokens.shape
        grid_size = int(math.sqrt(N))
        
        # Reshape the 1D sequence back into a 2D spatial feature map
        # Shape becomes [Batch, Channels, Height, Width] -> [B, 768, 14, 14]
        spatial_features = patch_tokens.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)
        
        # --- SR Branch ---
        # Pass the 2D spatial features through the upsampling block
        x_sr_features = self.sr_branch[:-1](spatial_features) 
        
        # Final image reconstruction for calculating L_sr 
        sr_output = self.sr_branch[-1](x_sr_features) 
        
        # --- Detection Branch ---
        # Predict the high-resolution heatmap
        d_out = self.det_conv1(x_sr_features)
        d_out = self.det_relu(d_out)
        heatmap_output = self.det_head(d_out)
        
        return heatmap_output, sr_output