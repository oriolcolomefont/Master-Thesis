from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset_msd import MyDatasetMSD
from model import TripletNet

from sklearn.model_selection import train_test_split

import datetime
import wandb

from collate_fn import collate_fn

# Create dataset
audio_df = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/msd_audio_files_limit=1000.csv"


# Assuming you have a DataFrame named 'audio_df'
train_df, val_df = train_test_split(audio_df, test_size=0.2, random_state=42)

train_set = MyDatasetMSD(input_df=train_df, sample_rate=16000, loss_type="triplet")
val_set = MyDatasetMSD(input_df=val_df, sample_rate=16000, loss_type="triplet")
# test_set = MyDataset(root_dir=test_path, sample_rate=16000)

# Create data/validation loader and setup data
batch_size = 16

train_loader = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, loss_type=train_set.loss_type),
    num_workers=16,
    drop_last=True,
)
validation_loader = DataLoader(
    dataset=val_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, loss_type=val_set.loss_type),
    num_workers=16,
    drop_last=True,
)

# Initialize model
model = TripletNet(
    strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
    supervised=False,
    out_dim=128,
    loss_type="triplet",
)

# Initialize WandB logger
wandb_logger = WandbLogger(
    experiment=None,
    project="master-thesis",  # Name of the project to log the run to (default: None)
    log_model=True,  # Log model checkpoints at the end of training
    save_dir="./wandb",  # Directory to save the logs to (default: None)
)

# log gradients, parameter histogram and model topology
wandb_logger.watch(model, log="gradients", log_graph=False)

# Get the current date
date = datetime.date.today().strftime("%Y-%m-%d")

# Get the name of the WandB run
run_name = wandb.run.name if wandb.run else "local-run"

# Define the filename for the checkpoint
filename = f"run-{run_name}-{date}-{{epoch:02d}}-{{val_loss:.2f}}-{model.loss_type}"

# Create callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=1000, verbose=True, mode="min"),
    ModelCheckpoint(
        dirpath="./checkpoints",
        filename=filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=False,
    ),
]

# Initialize trainer and pass wandb_logger
trainer = Trainer(
    accelerator="gpu",
    default_root_dir="./checkpoints",
    devices=4,
    enable_checkpointing=True,
    enable_progress_bar=True,
    callbacks=callbacks,
    logger=wandb_logger,
    log_every_n_steps=10,
    max_epochs=1000,
    precision="16-mixed",
    strategy="ddp",
)

# Start training
trainer.fit(model, train_loader, validation_loader)
