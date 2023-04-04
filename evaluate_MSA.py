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

        self.encoder = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128
        )
        self.encoder.eval()

        self.model = TripletNet(self.encoder)
        self.model.eval()

    @classmethod
    def get_id(cls):
        """Identifier of these features."""
        return "embedding"

    def compute_features(self):
        """Compute the embeddings of the audio signal using the SampleCNN model."""
        # convert it to a 2D mono tensor of shape (1, samples), you could use the torch.from_numpy() and unsqueeze() functions as follows:
        audio_tensor = torch.from_numpy(self._audio)

        window_size = 5 * 16000
        embeddings = []
        for i in range(0, audio_tensor.shape[0] - window_size, config.hop_size):
            x = audio_tensor[i : i + window_size]
            if x.shape[0] < window_size:
                break
            embedding = self.model(x[None, :])[0]
            embedding_np = embedding.detach().cpu().numpy()
            embeddings.append(embedding_np)
        print(embeddings)
        embedding_np = np.stack(embeddings, axis=0)

        return embedding_np


# Simple MSAF example
import msaf

# 1. Select audio file
audio_file = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/datasets/GTZAN/gtzan_genre/genres/disco/disco.00006.wav"

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
