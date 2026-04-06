#loss function
import torch
import torch.nn as nn
import torch.nn.functional as F

def centernet_focal_loss(pred, target):
    """
    pred: predicted heatmap [Batch, Channels, Height, Width]. Must have Sigmoid applied.
    target: ground truth Gaussian heatmap [Batch, Channels, Height, Width].
    """
    # Clamp predictions to avoid log(0) errors
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
    
    # Identify exact center points (where Gaussian peak == 1) and background
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    # Reduce penalty for pixels near the center (where target is > 0 but < 1)
    neg_weights = torch.pow(1 - target, 4) 
    
    # Standard focal loss calculation (alpha=2, beta=4 are typical CenterNet hyperparams)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
        
    return loss