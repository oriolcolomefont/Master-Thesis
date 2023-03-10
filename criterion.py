import torch
import torch.nn as nn
import torch.nn.functional as F

"""
we are using the pairwise_distance function from PyTorch to calculate the Euclidean distance between 
the anchor, positive, and negative examples. 
We then calculate the loss using the triplet margin loss formula. 
We use a margin of 0.2 but will determine which is best
"""


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
