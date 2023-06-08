import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from collections import deque
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

FILE_LIST_PATH = "./datasets/MSD/MSD_audio_limit=all.csv"
DATASET_NAME = "Million Song Dataset"

BATCH_SIZE = 8
CLIP_DURATION = 15.0
SAMPLE_RATE = 16000
LOSS_TYPE = "triplet"
STRIDES = [3, 3, 3, 3, 3, 3, 3, 3, 3]
OUT_DIM = 128
SUPERVISED = False
MAX_EPOCHS = 1000
PATIENCE = 50
LOG_EVERY_N_STEPS = 10
PRECISION = "16-mixed"
PROJECT_NAME = "MASTER THESIS"
#CPU_COUNT = multiprocessing.cpu_count()
CPU_COUNT = 16


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
        pin_memory=True,
    )
    validation_loader = DataLoader(
        dataset=val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, loss_type=val_set.loss_type),
        num_workers=CPU_COUNT,
        drop_last=True,
        pin_memory=True,
    )
    return train_loader, validation_loader


def init_model_and_logger(config):
    model = TripletNet(
        strides=STRIDES,
        supervised=SUPERVISED,
        out_dim=OUT_DIM,
        loss_type=LOSS_TYPE,
    )

    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        log_model=True,
        save_dir="./wandb",
        config=config,
        group=None,
    )

    return model, wandb_logger


class CustomModelCheckpoint(ModelCheckpoint):

    def __init__(self, dirpath, monitor, mode, save_weights_only, save_top_k, save_last_k):
        super().__init__(dirpath=dirpath, monitor=monitor, mode=mode,
                         save_weights_only=save_weights_only, save_top_k=save_top_k)
        self.save_last_k = save_last_k
        self.last_k_paths = deque(maxlen=self.save_last_k)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        # Save last N checkpoints
        if len(self.last_k_paths) == self.save_last_k:
            try:
                os.remove(self.last_k_paths[0])
            except:
                pass
        self.last_k_paths.append(self.last_model_path)


def create_callbacks():
    checkpoint_callback = CustomModelCheckpoint(
        dirpath="./checkpoints",
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
        save_top_k=5,
        save_last_k=5,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, verbose=True, mode="min"
    )

    return [early_stopping_callback, checkpoint_callback], checkpoint_callback


def train_model(model, train_loader, validation_loader, wandb_logger):
    callbacks, checkpoint_callback = create_callbacks()

    trainer = Trainer(
        default_root_dir="./checkpoints",
        logger=wandb_logger,
        max_epochs=MAX_EPOCHS,
        precision="16-mixed" if PRECISION == "16-mixed" else 32,
        sync_batchnorm=True,
        callbacks=callbacks,
        enable_checkpointing=True,
    )

    fit = trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    best_model_path = checkpoint_callback.best_model_path

    return fit, best_model_path


def main():
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
        "log_every_n_steps": LOG_EVERY_N_STEPS,
        "precision": PRECISION,
        "dataset_name": DATASET_NAME,
    }

    file_list = load_file_list(FILE_LIST_PATH)
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    train_set, val_set = get_train_val_datasets(train_files, val_files)
    train_loader, validation_loader = create_data_loaders(train_set, val_set)
    model, wandb_logger = init_model_and_logger(config)
    fit, best_model_path = train_model(
        model, train_loader, validation_loader, wandb_logger
    )

    return fit, best_model_path


if __name__ == "__main__":
    main()
