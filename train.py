import torch
import pytorch_lightning as pl
import wandb

from pytorchsummary import summary

from dataset import MyDataset
from model import TripletNet, SampleCNN

# Create dataset
dataset = MyDataset(root_dir="datasets/GTZAN/gtzan_genre", fixed_audio_len=44100*3)

# Create data loader and setup data
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    log_model=True,  # Whether to log the model to WandB (default: True)
    offline=False,  # Whether to run WandB in offline mode (default: False)
    save_dir="/home/oriol_colome_font_epidemicsound_/Master-Thesis-1/runs/runs and checkpoints",  # Directory to save the logs and checkpoint files (default: None)
    version=None,  # Version of the run (default: None)
    entity=None,  # WandB entity to log the run to (default: None)
    group=None,  # Name of the run group (default: None)
    config={
        "lr": 0.001,
        "batch_size": batch_size,
    },  # Dictionary of hyperparameters and their values (default: None)
    tags=[
        "training",
        "pytorch",
        "deep learning",
    ],  # List of tags to apply to the run (default: None)
    reinit=False,  # Whether to reinitialize the WandB run (default: False)
    resume=False,  # Whether to resume a previously stopped WandB run (default: False)
)

# add your batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = batch_size

# Initialize trainer and pass wandb_logger
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

# Start training
trainer.fit(model, train_loader)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
