import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

#from pytorchsummary import summary

from dataset import MyDataset
from model import TripletNet, SampleCNN

def pad_waveform(waveform, target_length):
    current_length = waveform.shape[-1]
    if current_length < target_length:
        # Calculate the number of zeros to pad
        num_zeros = target_length - current_length
        # Pad the waveform with zeros
        padded_waveform = F.pad(waveform, (0, num_zeros), mode='constant', value=0)
        return padded_waveform
    else:
        return waveform

def collate_fn(batch):
    anchors = []
    positives = []
    negatives = []

    # Find the maximum waveform length in the batch
    max_length = 0
    for item in batch:
        max_length = max(max_length, item['anchor'].shape[-1], item['positive'].shape[-1], item['negative'].shape[-1])

    # Pad all waveforms to the maximum length
    for item in batch:
        anchors.append(pad_waveform(item['anchor'], max_length))
        positives.append(pad_waveform(item['positive'], max_length))
        negatives.append(pad_waveform(item['negative'], max_length))

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return {'anchor': anchors, 'positive': positives, 'negative': negatives}

# Create dataset
data_path = "datasets/GTZAN/gtzan_genre"
min_length = 44100  # Minimum audio length in samples

dataset = MyDataset(root_dir=data_path, min_length=min_length)

# Create data loader and setup data
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Encoder
encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128)

#print(summary(model=encoder))
# Initialize model
model = TripletNet(encoder)

# Initialize WandB logger
wandb_logger = pl.loggers.WandbLogger(
    name="my-run",  # Name of the run (default: None) torchu
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
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

# Start training
trainer.fit(model, train_loader)
