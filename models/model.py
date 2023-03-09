#Import the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio

from loss_function import TripletLoss
from dataset import MyDataset

#Define the triplet network model by inheriting from pl.LightningModule.

'''
In this example, we are using a convolutional neural network to extract features from raw audio signals. 
The embedding_size parameter specifies the size of the output embedding space. 
We have defined a simple architecture with three convolutional layers, followed by max pooling, and two fully connected layers.
'''

class TripletNet(pl.LightningModule):
    def __init__(self, embedding_size=128):
        super(TripletNet, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, self.embedding_size)
        self.save_hyperparameters()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    '''
    we are using the training_step function from pl.LightningModule to define our training step. 
    We are passing in batches of anchor, positive, and negative examples. 
    We then compute the embeddings for each example using our TripletNet model. 
    We calculate the loss using our TripletLoss function and log the loss using self.log.
    '''
    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor_embedding = self(anchor)
        positive_embedding = self(positive)
        negative_embedding = self(negative)
        loss = TripletLoss()(anchor_embedding, positive_embedding, negative_embedding)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def triplet_loss(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        loss = torch.mean(torch.max(distance_positive - distance_negative + self.margin, torch.tensor([0.]).to(anchor.device)))
        return loss

    def train_dataloader(self):
        audio_dataset = MyDataset(root_dir='data') #To be improved
        batch_size = 32
        train_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader