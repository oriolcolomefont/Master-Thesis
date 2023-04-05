from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
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

# test_path =

train_set = MyDataset(root_dir=train_path, sample_rate=16000)
val_set = MyDataset(root_dir=val_path, sample_rate=16000)
# test_set = MyDataset(root_dir=test_path, sample_rate=16000)

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

# test_dataloader

# Encoder
encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128)

# Initialize model
model = TripletNet(encoder)

# Initialize WandB logger
wandb_logger = WandbLogger(
    experiment=None,
    project="master-thesis",  # Name of the project to log the run to (default: None)
    log_model=True,  # Log model checkpoints at the end of training
    save_dir="/home/oriol_colome_font_epidemicsound_/Master-Thesis-1/runs/runs and checkpoints",
)

# log gradients, parameter histogram and model topology
wandb_logger.watch(model, log="all")

# Create callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min"),
    ModelCheckpoint(
        dirpath="./checkpoints",
        filename="example-{epoch:02d}-{val_loss:.2f}",
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
    devices=2,
    enable_checkpointing=True,
    enable_progress_bar=True,
    callbacks=callbacks,
    logger=wandb_logger,
    log_every_n_steps=10,
    max_epochs=10,
    precision="16-mixed",
    strategy="ddp",
)

# Start training
trainer.fit(model, train_loader, validation_loader)
#trainer.save_checkpoint(filepath="./runs wandb/example.ckpt", weights_only=False, storage_options=None)

# trainer.test(test_dataloader)
