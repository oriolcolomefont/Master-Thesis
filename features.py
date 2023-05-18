"""
Each feature must inherit from the base class :class:`msaf.base.Features` to be
included in the whole framework.

Here is a list of all the available features:

.. autosummary::
    :toctree: generated/

    CQT
    MFCC
    PCP
    Tonnetz
    Tempogram
    Features
    Embeddiogram
"""

from builtins import super
import librosa
import numpy as np

# Necessary packages for the embedding feature

import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Local stuff
from msaf import config
from msaf.base import Features
from msaf.exceptions import FeatureParamsError

import sys
import importlib.util

from msaf.base import features_registry
from msaf.configdefaults import AddConfigVar, IntParam

# embeddiogram Features
AddConfigVar(
    "embeddiogram.win_length",
    "The size of the window of the embeddiogram.",
    IntParam(8 * 22050),
)


class Embeddiogram(Features):
    """This class contains the implementation of the Embeddiogram feature."""

    def __init__(
        self,
        file_struct,
        feat_type,
        sr=config.sample_rate,
        hop_length=config.hop_size,
        win_length=config.embeddiogram.win_length,
    ):
        # Init the parent
        super().__init__(
            file_struct=file_struct, sr=sr, hop_length=hop_length, feat_type=feat_type
        )
        self.win_length = win_length

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "embeddiogram"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        embediogram: np.array(N, F)
            The features, each row representing a feature/embedding vector for a give
            time frame.
        """
        # Load and preprocess the input audio
        input_audio = self._audio  # time domain samples only
        print("Input audio loaded")
        print(f"Sample rate = {self.sr}")
        print(f"Number of samples = {input_audio.shape[0]}")

        window_size = 4 * self.sr  # number of samples in a window
        hop_size = self.hop_length  # number of samples between windows
        num_hops = 1 + (input_audio.shape[0] - window_size) // hop_size
        if (input_audio.shape[0] - window_size) % hop_size > 0:
            num_hops += 1  # add an extra hop for any remaining samples at the end
        print(f"Number of hops: {num_hops}")

        # Determine the number of CPUs and GPUs available
        num_cpus = cpu_count()
        num_gpus = torch.cuda.device_count()

        # Print the number of CPUs and GPUs
        print("Number of CPUs: ", num_cpus)
        print("Number of GPUs: ", num_gpus)

        # Initialize the ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=num_cpus)

        # Set up the input and output data
        embeddings = []
        slices = [
            (i * hop_size, min(i * hop_size + window_size, input_audio.shape[0]))
            for i in range(num_hops)
        ]

        gpu_device_ids = list(range(num_gpus))

        # Calculate the number of slices per GPU
        slices_per_gpu = len(slices) // num_gpus
        if len(slices) % num_gpus > 0:
            slices_per_gpu += 1

        # Divide the audio slices into chunks for each GPU
        slice_chunks = [
            slices[i : i + slices_per_gpu]
            for i in range(0, len(slices), slices_per_gpu)
        ]

        # Process the chunks of audio slices on each GPU
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            embeddings = list(
                executor.map(
                    self.process_audio_slices_chunk, gpu_device_ids, slice_chunks
                )
            )

        # Flatten the list of lists of embeddings
        embeddings = [embedding for chunk in embeddings for embedding in chunk]

        # Stack the embeddings
        embeddiogram = np.column_stack(embeddings)

        normalized_embeddiogram = (embeddiogram - embeddiogram.min()) / (
            embeddiogram.max() - embeddiogram.min()
        )

        return normalized_embeddiogram.T

    def process_audio_slices_chunk(self, gpu_device_id, slice_chunk):
        # Add the directory containing my model.py file to the system path
        sys.path.append("/home/jupyter/Master-Thesis/")

        # Import the model.py module using importlib
        spec = importlib.util.spec_from_file_location(
            "model", "/home/jupyter/Master-Thesis/model.py"
        )
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Load the checkpoint file
        CKPT_PATH = "/home/jupyter/Master-Thesis/checkpoints/run-solar-sound-307-2023-04-20-epoch=127-val_loss=0.03-triplet.ckpt"
        checkpoint = torch.load(CKPT_PATH)

        # Set the current device to the specified GPU
        torch.cuda.set_device(gpu_device_id)

        # Initialize the model on the current GPU
        model = model_module.TripletNet(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=False,
            out_dim=128,
            loss_type="triplet",
        ).to(f"cuda:{gpu_device_id}")

        # Load the model's state_dict
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # Process the audio slices and return the embeddings
        embeddings = []
        for start_end_tuple in tqdm(
            slice_chunk,
            desc=f"Processing audio slices on GPU {gpu_device_id}",
            leave=True,
        ):
            start, end = start_end_tuple
            audio_slice = self._audio[None, None, start:end]
            audio_slice = (
                torch.from_numpy(audio_slice).float().to(f"cuda:{gpu_device_id}")
            )

            with torch.no_grad():
                embedding = model(audio_slice).cpu().numpy().flatten()
            embeddings.append(embedding)

        return embeddings


# All available features
features_registry["embeddiogram"] = Embeddiogram
