# Import the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from criterion import TripletLoss

import wandb

wandb.login()

"""
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def initialize(self, m):
        if isinstance(m, (nn.Conv1d)):
            # nn.init.xavier_uniform_(m.weight)
            # if m.bias is not None:
            #     nn.init.xavier_uniform_(m.bias)
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
"""


class SampleCNN(nn.Module):
    def __init__(self, strides, supervised, out_dim, device=None):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.supervised = supervised
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sequential = [
            nn.Sequential(
                nn.Conv1d(
                    in_channels=1, out_channels=128, kernel_size=3, stride=3, padding=0
                ),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=h_in,
                        out_channels=h_out,
                        kernel_size=stride,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

        if self.supervised:
            self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        out = self.sequential(x)
        if self.supervised:
            out = self.dropout(out)
        out = torch.mean(out, dim=2)
        logit = self.fc(out)
        return logit


class TripletNet(pl.LightningModule):
    def __init__(self, encoder=SampleCNN, lr=0.001):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters(ignore=['encoder'])
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch  # batch is now a tuple
        anchor_embedding = self.encoder(anchor)
        positive_embedding = self.encoder(positive)
        negative_embedding = self.encoder(negative)
        train_loss = self.triplet_loss(
            anchor_embedding, positive_embedding, negative_embedding
        )
        self.log("val_loss", train_loss, sync_dist=True, rank_zero_only=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch  # batch is now a tuple
        anchor_embedding = self.encoder(anchor)
        positive_embedding = self.encoder(positive)
        negative_embedding = self.encoder(negative)
        val_loss = self.triplet_loss(
            anchor_embedding, positive_embedding, negative_embedding
        )
        self.log("val_loss", val_loss, sync_dist=True, rank_zero_only=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        anchor, positive, negative = batch  # batch is now a tuple
        anchor_embedding = self.encoder(anchor)
        positive_embedding = self.encoder(positive)
        negative_embedding = self.encoder(negative)
        test_loss = self.triplet_loss(
            anchor_embedding, positive_embedding, negative_embedding
        )
        self.log("test_loss", test_loss, sync_dist=True, rank_zero_only=True)
        return test_loss

    def triplet_loss(self, anchor, positive, negative):
        criterion = TripletLoss(margin=1.0)
        return criterion(anchor, positive, negative)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        return optimizer
