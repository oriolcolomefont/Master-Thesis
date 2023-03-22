# Import the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from criterion import TripletLoss

import wandb
wandb.login()


# Define the triplet network model by inheriting from pl.LightningModule.

"""
We are using a convolutional neural network to extract features from raw audio signals. 
The embedding_size parameter specifies the size of the output embedding space. 
We have defined a simple architecture with three convolutional layers, followed by max pooling, and two fully connected layers.
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


class SampleCNN(Model):
    def __init__(self, strides, supervised, out_dim):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.supervised = supervised
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
        #out = torch.avg_pool1d(out, axis=out.size(2)) #TODO check how to implement time distributed pooling layer
        #out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        out = torch.mean(out, dim=2)
        logit = self.fc(out)
        return logit

class TripletNet(pl.LightningModule):
    def __init__(self, encoder: nn.Module, lr=0.001):
        super().__init__()
        # log hyperparameters
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch  # batch is now a tuple
        anchor_embedding = self.encoder(anchor)
        positive_embedding = self.encoder(positive)
        negative_embedding = self.encoder(negative)
        loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        self.log("train_loss", loss)
        return loss

    def triplet_loss(self, anchor, positive, negative):
        criterion = TripletLoss(margin=1.0)
        return criterion(anchor, positive, negative)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        return optimizer
