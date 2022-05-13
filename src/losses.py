import numpy as np
import torch
import torch.nn as nn

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):        
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean(dim=1)
        return loss   


# class BCEFocal2WayLoss(nn.Module):
#     def __init__(self, weights=[1, 1], class_weights=None):
#         super().__init__()

#         self.focal = BCEFocalLoss()

#         self.weights = weights

#     def forward(self, input,):
#         input_ = input["logit"]
#         target = input["targets"]

#         framewise_output = input["framewise_logit"]
#         clipwise_output_with_max, _ = framewise_output.max(dim=1)

#         loss = self.focal(input_, target)
#         aux_loss = self.focal(clipwise_output_with_max, target)

#         return self.weights[0] * loss + self.weights[1] * aux_loss
    
class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input,):
        input_ = input["logit"]
        target = input["targets"]

        #framewise_output = input["framewise_logit"]
        #clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        #aux_loss = self.focal(clipwise_output_with_max, target_)

        return self.weights[0] * loss    