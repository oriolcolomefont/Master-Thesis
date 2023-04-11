# Import the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from criterion import TripletLoss, ContrastiveLoss

import wandb

wandb.login()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def initialize(self, m):
        if isinstance(m, (nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")


class SampleCNN(Model):
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
    def __init__(
        self, strides, supervised, out_dim, loss_type="triplet", *args, **kwargs
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = SampleCNN(strides, supervised, out_dim)
        self.loss_type = loss_type

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        if self.loss_type == "triplet":
            anchor, positive, negative = batch
            anchor_embedding = self.encoder(anchor)
            positive_embedding = self.encoder(positive)
            negative_embedding = self.encoder(negative)
            loss_function = self.get_loss_function()
            train_loss = loss_function(
                anchor_embedding, positive_embedding, negative_embedding
            )
        else:  # self.loss_type == "contrastive":
            sample1, sample2, label = batch
            sample1_embedding = self.encoder(sample1)
            sample2_embedding = self.encoder(sample2)
            loss_function = self.get_loss_function()
            train_loss = loss_function(sample1_embedding, sample2_embedding, label)
        self.log("train_loss", train_loss, sync_dist=True, rank_zero_only=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        if self.loss_type == "triplet":
            anchor, positive, negative = batch
            anchor_embedding = self.encoder(anchor)
            positive_embedding = self.encoder(positive)
            negative_embedding = self.encoder(negative)
            loss_function = self.get_loss_function()
            val_loss = loss_function(
                anchor_embedding, positive_embedding, negative_embedding
            )
        else:  # self.loss_type == "contrastive":
            sample1, sample2, label = batch
            sample1_embedding = self.encoder(sample1)
            sample2_embedding = self.encoder(sample2)
            loss_function = self.get_loss_function()
            val_loss = loss_function(sample1_embedding, sample2_embedding, label)
        self.log("val_loss", val_loss, sync_dist=True, rank_zero_only=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        if self.loss_type == "triplet":
            anchor, positive, negative = batch
            anchor_embedding = self.encoder(anchor)
            positive_embedding = self.encoder(positive)
            negative_embedding = self.encoder(negative)
            loss_function = self.get_loss_function()
            test_loss = loss_function(
                anchor_embedding, positive_embedding, negative_embedding
            )
        else:  # self.loss_type == "contrastive":
            sample1, sample2, label = batch
            sample1_embedding = self.encoder(sample1)
            sample2_embedding = self.encoder(sample2)
            loss_function = self.get_loss_function()
            test_loss = loss_function(sample1_embedding, sample2_embedding, label)
        self.log("test_loss", test_loss, sync_dist=True, rank_zero_only=True)
        return test_loss

    def get_loss_function(self):
        if self.loss_type == "triplet":
            return TripletLoss()
        elif self.loss_type == "contrastive":
            return ContrastiveLoss()
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.0003)
        return optimizer
