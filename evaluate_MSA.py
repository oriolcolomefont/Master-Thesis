from model import SampleCNN, TripletNet
from msaf import config
from msaf.base import Features

import torch
import numpy as np


class Embedding(Features):
    """This class computes embeddings of an audio signal using a pre-trained SampleCNN model."""

    def __init__(
        self, file_struct, feat_type, sr=config.sample_rate, hop_length=config.hop_size
    ):
        super().__init__(
            file_struct=file_struct, sr=sr, hop_length=hop_length, feat_type=feat_type
        )

        self.model = TripletNet(SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128
        ))
        self.model.eval()

    @classmethod
    def get_id(cls):
        """Identifier of these features."""
        return "embedding"

    def compute_features(self):
        """Compute the embeddings of the audio signal using the SampleCNN model."""
        input_audio = self._audio
        # Convert the NumPy array to a PyTorch tensor
        input_audio_tensor = torch.from_numpy(input_audio)

        # Add a new dimension at position 0 to make it a 2D tensor
        input_torch_tensor = input_audio_tensor.unsqueeze(0)

        print(input_torch_tensor.shape)  # Output: torch.Size([1, N])

        audio_embedding = self.model(input_torch_tensor).detach().numpy()

        window_size = 128
        hop_length = self.hop_length
        num_frames = (len(audio_embedding) - window_size) // hop_length + 1

        features = np.zeros((num_frames, window_size))

        for i in range(num_frames):
            start = i * hop_length
            end = start + window_size
            features[i, :] = audio_embedding[start:end]

        return features
    
# Simple MSAF example
import msaf

# 1. Select audio file
audio_file = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/Track No01.mp3"

# 2. Segment the file using the default MSAF parameters (this might take a few seconds)
boundaries, labels = msaf.process(audio_file, feature="embedding")
print("Estimated boundaries:", boundaries)

# 3. Save segments using the MIREX format
out_file = "segments.txt"
print("Saving output to %s" % out_file)
msaf.io.write_mirex(boundaries, labels, out_file)

# 4. Evaluate the results
try:
    evals = msaf.eval.process(audio_file)
    print(evals)
except msaf.exceptions.NoReferencesError:
    file_struct = msaf.input_output.FileStruct(audio_file)
    print(
        "No references found in {}. No evaluation performed.".format(
            file_struct.ref_file
        )
    )
