import torch
from torch.nn import functional as F

def centernet_loss(pred, heatmap, scalemap, classmap):
    heatmap_loss = sigmoid_focal_loss(pred[:,0:1,:,:], heatmap==1, heatmap)
    scale_loss = l1_loss(pred[:,1:3,:,:], scalemap)
    if pred.size(1) > 3:
        class_loss = F.cross_entropy(pred[:,3:,:,:], classmap.squeeze(1), ignore_index=-1)
        return heatmap_loss + 0.1*scale_loss + class_loss
    else:
        return heatmap_loss + 0.1*scale_loss

def l1_loss(pred, gt):
    valid = gt > 0
    n_points = valid.sum()//2
    loss = 0
    if n_points > 0:
        loss = abs(pred[valid] - gt[valid]).sum() / n_points
    
    return loss

def sigmoid_focal_loss(inputs, targets, alpha):
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** 2)
    
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = torch.pow(alpha_t, 4) * loss
    loss = loss.sum()

    n_targets = targets.sum()

    if n_targets > 0:
        loss = loss / n_targets 

    return loss