import numpy as np
import torch
from .losses import BCEFocal2WayLoss
from hydra.utils import instantiate

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


def cutmix_criterion(preds, new_targets, cfg, weight=None):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss(weights=cfg.losses.weights)
    loss = lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)
    if weight is not None:
        return ((loss * weight) / weight.sum()).sum()
    else:
        return loss.mean()


def mixup_criterion(preds, new_targets, cfg, weight=None):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss(weights=cfg.losses.weights)
    loss = lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)
    if weight is not None:
        return ((loss * weight) / weight.sum()).sum()
    else:
        return loss.mean()


def loss_fn(outputs, cfg):
    criterion = BCEFocal2WayLoss(weights=cfg.losses.weights)
    loss = criterion(outputs)
    
    weight = outputs["weight"]
    
    if weight is not None:
        return ((loss * weight) / weight.sum()).sum()
    return loss.mean()