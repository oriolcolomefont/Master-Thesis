import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import MyDataset
from model import TripletNet, SampleCNN

#from lightning.pytorch.callbacks import Callback
"""
class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")
"""

def pad_waveform(waveform, length):
    padded_waveform = torch.zeros(waveform.shape[0], length)
    padded_waveform[..., :waveform.shape[-1]] = waveform
    return padded_waveform

def collate_fn(batch):
    anchors = []
    positives = []
    negatives = []

    max_length = max(max(item['anchor'].shape[-1], item['positive'].shape[-1], item['negative'].shape[-1]) for item in batch)

    # Pad all waveforms to the maximum length in the batch
    for item in batch:
        anchors.append(pad_waveform(item['anchor'], max_length))
        positives.append(pad_waveform(item['positive'], max_length))
        negatives.append(pad_waveform(item['negative'], max_length))

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    # Online triplet mining: Select the hardest negative for each anchor-positive pair
    anchor_positive_distance = (anchors - positives).pow(2).sum(dim=2).sqrt()
    anchor_negative_distance = (anchors.unsqueeze(2) - negatives.unsqueeze(1)).pow(2).sum(dim=3).sqrt()
    hardest_negative_indices = torch.argmax(anchor_negative_distance - anchor_positive_distance.unsqueeze(2), dim=2)
    hardest_negatives = torch.cat([negatives[i, idx, :].unsqueeze(0) for i, idx in enumerate(hardest_negative_indices)])

    return anchors, positives, hardest_negatives

# Create dataset
data_path = "datasets/GTZAN/gtzan_genre"
dataset = MyDataset(root_dir=data_path, resample=22050)

# Create data loader and setup data
batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Encoder
encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128)

#print(summary(model=encoder))
# Initialize model
model = TripletNet(encoder)

# Initialize WandB logger
wandb_logger = pl.loggers.WandbLogger(
    name="second overnight run",  # Name of the run (default: None) torchu
    id=None,  # ID of the run (default: None)
    project="master-thesis",  # Name of the project to log the run to (default: None)
    save_dir="/home/oriol_colome_font_epidemicsound_/Master-Thesis-1/runs/runs and checkpoints",  # Directory to save the logs and checkpoint files (default: None)
    config={
        "lr": 0.001,
        "batch_size": batch_size,
    },  # Dictionary of hyperparameters and their values (default: None)
    tags=[
        "training",
        "pytorch",
        "deep learning",
    ],  # List of tags to apply to the run (default: None)
)

# add your batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = batch_size

# Initialize trainer and pass wandb_logger
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, accumulate_grad_batches=4)

# Start training
trainer.fit(model, train_loader)
