
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',
                 ignore_lb=101):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            label = label.clone().detach()
            ignore = label == self.ignore_lb
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_one_hot = torch.zeros_like(logits).scatter_(
                1, label.unsqueeze(1), 1).detach()
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[lb_one_hot == 1] = self.alpha

        # compute loss
        probs = torch.sigmoid(logits)
        pt = torch.where(lb_one_hot == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, lb_one_hot)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss).sum(dim=1)
        loss[ignore == 1] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss