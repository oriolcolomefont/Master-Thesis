import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

wandb_logger = WandbLogger()
trainer = Trainer(logger=wandb_logger)

from dataloader import *
from dataset import MyDataset
from loss_function import TripletLoss
from models import TripletNet
from wandb import *


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})
        if batch_idx % 50 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def main():
    # Initialize WandB
    wandb.init(project='triplet-network')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = load_data()  # Define this function to load your data (check /home/oriol_colome_font_epidemicsound_/Master-Thesis-1/dataloader/dataloader.py)
    dataset = MyDataset(data)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Initialize model, criterion, and optimizer
    model = TripletNet().to(device)
    criterion = TripletLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(10):
        train(model, train_loader, criterion, optimizer, device)

if __name__ == '__main__':
    main()