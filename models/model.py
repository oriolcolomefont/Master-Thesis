import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
#from typing_extensions import Protocol

import wandb

class MyModel(pl.LightningModule):
    
    def __init__(self, num_classes=10, conv_layers=[(32, 3), (64, 3)], loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam):
        super().__init__()
        # Define convolutional layers of the neural network
        self.conv_layers = nn.ModuleList()
        prev_channels = 1
        for channels, kernel_size in conv_layers:
            self.conv_layers.append(nn.Conv2d(prev_channels, channels, kernel_size))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2))
            self.conv_layers.append(nn.Dropout2d(p=0.25))
            prev_channels = channels
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # Define loss function and optimizer
        self.loss_func = loss_func
        self.optimizer = optimizer(self.parameters(), lr=1e-3)
        
        def forward(self, x):
        # Define forward pass of the neural network
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # Define training step of the neural network
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        # Log training loss to Wandb logger
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Define validation step of the neural network
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        # Log validation loss to Wandb logger
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        # Return optimizer for neural network
        return self.optimizer
    
    def prepare_data(self):
        # Define data preprocessing steps
        # Load and preprocess audio data
        
    def train_dataloader(self):
        # Define data loading for training set
        # Return DataLoader for training set
        
    def val_dataloader(self):
        # Define data loading for validation set
        # Return DataLoader for validation set


# Initialize model and Wandb logger
model = MyModel()
wandb_logger = pl.loggers.WandbLogger(project='master-thesis')

# Initialize PyTorch Lightning Trainer
trainer = pl.Trainer(gpus=1, logger=wandb_logger)

# Train model
trainer.fit(model)