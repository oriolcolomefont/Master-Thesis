from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from dataset import MyDataset
from model import TripletNet, SampleCNN

from collate_fn import collate_fn


# Create dataset
train_path = (
    "/home/oriol_colome_font_epidemicsound_/Master-Thesis/datasets/GTZAN/GTZAN train"
)
val_path = (
    "/home/oriol_colome_font_epidemicsound_/Master-Thesis/datasets/GTZAN/GTZAN validate"
)

train_set = MyDataset(root_dir=train_path, sample_rate=16000)
val_set = MyDataset(root_dir=val_path, sample_rate=16000)

# Create data/validation loader and setup data
batch_size = 16
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=16,
    drop_last=True,
)
validation_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=16,
    drop_last=True,
)

# Encoder
encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128)

# Initialize model
model = TripletNet(encoder)

# Initialize WandB logger
wandb_logger = pl.loggers.WandbLogger(
    project="master-thesis",  # Name of the project to log the run to (default: None)
    log_model=True,  # Log model topology (default: False)
    save_dir="/home/oriol_colome_font_epidemicsound_/Master-Thesis-1/runs/runs and checkpoints",  # Directory to save the logs and checkpoint files (default: None)
    config={
        "lr": 0.001,
        "batch_size": batch_size,
        "sample_rate": train_set.sample_rate,
    },  # Dictionary of hyperparameters and their values (default: None)
    tags=[],  # List of tags to apply to the run (default: None)
)

# add your batch size to the wandb config
# wandb_logger.experiment.config["batch_size"] = batch_size


# Create callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min"),
    ModelCheckpoint(dirpath="./runs wandb"),
]

# Initialize trainer and pass wandb_logger
trainer = pl.Trainer(
    accelerator="gpu",
    devices=2,
    callbacks=callbacks,
    log_every_n_steps=batch_size,
    logger=wandb_logger,
    max_epochs=10,
    strategy="ddp")

# Start training
trainer.fit(model, train_loader, validation_loader)