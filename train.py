import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

from dataset import MyDataset
from model import TripletNet, SampleCNN

# from lightning.pytorch.callbacks import Callback
"""
class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")
"""


def pad_waveform(waveform, length):
    padded_waveform = F.pad(waveform, (0, length - waveform.shape[-1]), "constant", 0)
    return padded_waveform


def collate_fn(batch):
    anchors = []
    positives = []
    negatives = []

    max_length = max(
        max(
            item["anchor"].shape[-1],
            item["positive"].shape[-1],
            item["negative"].shape[-1],
        )
        for item in batch
    )
    # Pad all waveforms to the maximum length in the batch
    for item in batch:
        anchors.append(pad_waveform(item["anchor"], max_length))
        positives.append(pad_waveform(item["positive"], max_length))
        negatives.append(pad_waveform(item["negative"], max_length))

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    # Online triplet mining: select the hardest negative for each anchor-positive pair
    anchor_positive_distance = (anchors - positives).pow(2).sum(dim=2).sqrt()
    anchor_negative_distance = (
        (anchors.unsqueeze(2) - negatives.unsqueeze(1)).pow(2).sum(dim=3).sqrt()
    )
    hardest_negative_indices = torch.argmax(
        anchor_negative_distance - anchor_positive_distance.unsqueeze(2), dim=2
    )
    hardest_negatives = torch.cat(
        [
            negatives[i, idx, :].unsqueeze(0)
            for i, idx in enumerate(hardest_negative_indices)
        ]
    )

    return anchors, positives, hardest_negatives


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
batch_size = 32
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
wandb_logger.experiment.config["batch_size"] = batch_size

# Initialize trainer and pass wandb_logger
trainer = pl.Trainer(
    max_epochs=100, logger=wandb_logger, callbacks=[ModelCheckpoint(dirpath="./runs")]
)

# Start training
trainer.fit(model, train_loader, validation_loader)
