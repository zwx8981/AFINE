import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']

@LOSS_REGISTRY.register()
class FidelityLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(FidelityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-8
    def forward(self, p, g):
        p = p.view(-1)
        g = g.view(-1)
        loss = 1 - (torch.sqrt(p*g + self.eps) + torch.sqrt((1-p)*(1-g) + self.eps))
        return torch.mean(self.loss_weight*loss)


@LOSS_REGISTRY.register()
class MOSRegressionLoss(nn.Module):
    """Huber regression with an optional within-batch monotonic ranking term."""

    def __init__(self, loss_weight=1.0, ranking_weight=0.0, margin=0.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.ranking_weight = ranking_weight
        self.margin = margin

    def forward(self, prediction, target):
        prediction, target = prediction.view(-1), target.view(-1)
        loss = F.smooth_l1_loss(prediction, target)
        if self.ranking_weight and prediction.numel() > 1:
            target_diff = target[:, None] - target[None, :]
            mask = target_diff != 0
            prediction_diff = prediction[:, None] - prediction[None, :]
            rank = F.relu(self.margin - prediction_diff * target_diff.sign())
            loss = loss + self.ranking_weight * rank[mask].mean()
        return self.loss_weight * loss
