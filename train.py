import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

from pytorchsummary import summary

from dataset import MyDataset
from model import TripletNet, SampleCNN

def collate_fn(batch):
    def pad_audio(audio, max_length):
        padding = max_length - audio.shape[-1]
        return torch.nn.functional.pad(audio, (0, padding))

    max_length = 0

    # Find the maximum length in the batch
    for sample in batch:
        anchor_length = sample['anchor'].shape[-1]
        positive_length = sample['positive'].shape[-1]
        negative_length = sample['negative'].shape[-1]

        max_length = max(max_length, anchor_length, positive_length, negative_length)

    # Pad audios to have the same length
    padded_batch = []
    for sample in batch:
        padded_anchor = pad_audio(sample['anchor'], max_length)
        padded_positive = pad_audio(sample['positive'], max_length)
        padded_negative = pad_audio(sample['negative'], max_length)

        padded_sample = {'anchor': padded_anchor,
                         'positive': padded_positive,
                         'negative': padded_negative}

        padded_batch.append(padded_sample)

    return padded_batch

# Create dataset
data_path = "datasets/GTZAN/gtzan_genre"
min_length = 16000  # Minimum audio length in samples

dataset = MyDataset(root_dir=data_path, min_length)

# Create data loader and setup data
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Encoder
encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128)

print(summary(model=encoder, input_size=(1, 160857)))
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

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
