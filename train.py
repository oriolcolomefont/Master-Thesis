import datetime
import os
import pandas as pd
import numpy as np
import torch
import multiprocessing
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
import wandb

from dataset import MyDataset
from model import TripletNet
from collate_fn import collate_fn

# Authenticate the account and initilize the project

wandb.login(
    key="b047312016c639a2d2ebe9948a5937668cd19066",
)

FILE_LIST_PATH = (
    "./datasets/MSD/MSD_audio_limit=all_progress100.csv"
)
DATASET_NAME = "Million Song Dataset"

BATCH_SIZE = 8
CLIP_DURATION = 15.0
SAMPLE_RATE = 16000
LOSS_TYPE = "triplet"
STRIDES = [3, 3, 3, 3, 3, 3, 3, 3, 3]
OUT_DIM = 128
SUPERVISED = False
MAX_EPOCHS = 1000
PATIENCE = MAX_EPOCHS
SAVE_TOP_K = 3
LOG_EVERY_N_STEPS = 10
PRECISION = "16-mixed"
STRATEGY = "ddp"
PROJECT_NAME = "MASTER THESIS"
ACCELERATOR = "gpu"
GPUS = torch.cuda.device_count()
CPU_COUNT = multiprocessing.cpu_count()

config = {
    "batch_size": BATCH_SIZE,
    "clip_duration": CLIP_DURATION,
    "sample_rate": SAMPLE_RATE,
    "loss_type": LOSS_TYPE,
    "strides": STRIDES,
    "out_dim": OUT_DIM,
    "supervised": SUPERVISED,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "save_top_k": SAVE_TOP_K,
    "log_every_n_steps": LOG_EVERY_N_STEPS,
    "precision": PRECISION,
    "strategy": STRATEGY,
    "dataset_name": DATASET_NAME,
}

# Get the global rank of the current process (0 for the main process)
global_rank = 0
if torch.cuda.is_available() and ACCELERATOR == "gpu" and STRATEGY == "ddp":
    global_rank = int(os.environ.get("LOCAL_RANK", 0))

# Initialize wandb only for the main process (rank 0)
if global_rank == 0:
    wandb.init(
        project=PROJECT_NAME,
        job_type="train",
        config=config,
    )
else:
    # Set the WANDB_MODE for non-main processes to "dryrun" to prevent them from logging
    os.environ["WANDB_MODE"] = "dryrun"


def load_file_list(file_list_path):
    _, file_extension = os.path.splitext(file_list_path)

    if file_extension == ".csv":
        file_list = pd.read_csv(file_list_path)["file_path"].tolist()
    elif file_extension == ".npy":
        file_list = np.load(file_list_path).tolist()
    else:
        raise ValueError(
            f"Unsupported file extension '{file_extension}'. Please use a CSV or NumPy file."
        )

    return file_list


def get_train_val_datasets(train_files, val_files):
    train_set = MyDataset(
        file_list=train_files,
        clip_duration=CLIP_DURATION,
        sample_rate=SAMPLE_RATE,
        loss_type=LOSS_TYPE,
    )
    val_set = MyDataset(
        file_list=val_files,
        clip_duration=CLIP_DURATION,
        sample_rate=SAMPLE_RATE,
        loss_type=LOSS_TYPE,
    )
    return train_set, val_set


def create_data_loaders(train_set, val_set):
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, loss_type=train_set.loss_type),
        num_workers=CPU_COUNT,
        drop_last=True,
    )
    validation_loader = DataLoader(
        dataset=val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, loss_type=val_set.loss_type),
        num_workers=CPU_COUNT,
        drop_last=True,
    )
    return train_loader, validation_loader


def init_model_and_logger(config):
    model = TripletNet(
        strides=STRIDES,
        supervised=SUPERVISED,
        out_dim=OUT_DIM,
        loss_type=LOSS_TYPE,
    )

    if torch.cuda.is_available() and ACCELERATOR == "gpu" and STRATEGY == "ddp":
        model = model.cuda()
        import torch.distributed as dist

        dist.init_process_group(
            backend="nccl", init_method="tcp://localhost:23456", world_size=1, rank=0
        )

    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        log_model=True,
        save_dir="./wandb",
        config=config,
        group=None,
    )

    wandb_logger.watch(model, log="gradients", log_graph=False)

    return model, wandb_logger


def get_checkpoint_filename(run_name):
    date = datetime.date.today().strftime("%Y-%m-%d")
    return f"run-{run_name}-{date}-{{epoch:02d}}-{{val_loss:.2f}}-{LOSS_TYPE}"


def create_callbacks(run_name):
    filename = get_checkpoint_filename(run_name)

    return [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, verbose=True, mode="min"),
        ModelCheckpoint(
            dirpath="./checkpoints",
            filename=filename,
            monitor="val_loss",
            mode="min",
            save_top_k=SAVE_TOP_K,
            save_weights_only=False,
        ),
    ]


def train_model(model, train_loader, validation_loader, wandb_logger):
    trainer = Trainer(
        accelerator=ACCELERATOR if torch.cuda.is_available() else "cpu",
        default_root_dir="./checkpoints",
        logger=wandb_logger,
        strategy=STRATEGY,
        max_epochs=MAX_EPOCHS,
        precision="16-mixed" if PRECISION == "16-mixed" else 32,
        sync_batchnorm=True,
        callbacks=create_callbacks(wandb.run.name),
        enable_checkpointing=True,
    )

    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    wandb_logger.log_metrics({"trainer_state_dict": trainer.state_dict()})
    wandb.finish()


def main():
    file_list = load_file_list(FILE_LIST_PATH)
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    train_set, val_set = get_train_val_datasets(train_files, val_files)
    train_loader, validation_loader = create_data_loaders(train_set, val_set)
    model, wandb_logger = init_model_and_logger(config)
    train_model(model, train_loader, validation_loader, wandb_logger)


if __name__ == "__main__":
    main()
