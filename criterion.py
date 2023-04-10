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
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, label):
        distance = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1))
        positive_loss = (1 - label) * distance**2
        negative_loss = label * torch.clamp(self.margin - distance, min=0) ** 2
        loss = 0.5 * torch.mean(positive_loss + negative_loss)
        return loss


"""
The TripletLoss module calculates the loss for a triplet of inputs (anchor, positive, negative), where anchor is an anchor sample, positive is a sample that is similar to the anchor, and negative is a sample that is dissimilar to the anchor. The loss is calculated as the margin between the distance of the anchor and the positive sample minus the distance between the anchor and the negative sample, and then clipped at zero to ensure that the loss is non-negative. The loss encourages the anchor and positive samples to be closer in the embedding space than the anchor and negative samples.

The ContrastiveLoss module, on the other hand, calculates the loss for a pair of inputs (anchor, positive) and a binary label that indicates whether the pair is similar or dissimilar. The loss is calculated as the squared distance between the anchor and positive samples if the label is 0 (indicating dissimilarity), or as the squared difference between the margin and the distance between the anchor and positive samples if the label is 1 (indicating similarity). The loss encourages the anchor and positive samples to be closer in the embedding space if they are similar, and farther apart if they are dissimilar.

In summary, TripletLoss and ContrastiveLoss are both used for training siamese and triplet networks, but they calculate the loss differently and are suitable for different types of input samples. TripletLoss is used for triplets of samples, while ContrastiveLoss is used for pairs of samples with a binary label indicating their similarity. The choice of loss function depends on the specific task and dataset being used.
    """
