import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        loss = torch.mean(
            torch.max(
                distance_positive - distance_negative + self.margin,
                torch.tensor([0.0]).to(anchor.device),
            )
        )
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, label):
        distance = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1))
        positive_loss = (1 - label) * distance**2
        negative_loss = label * torch.clamp(self.margin - distance, min=0) ** 2
        loss = 0.5 * torch.mean(positive_loss + negative_loss)
        return loss
