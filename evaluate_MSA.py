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

        self.model = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128
        )
        self.target_length = 729

        self.model.eval()

    @classmethod
    def get_id(cls):
        """Identifier of these features."""
        return "embedding"

    def pad_or_truncate_audio(self, audio, target_length):
        """Helper method to pad or truncate the input audio to the target length."""

        audio_length = len(audio)

        if audio_length < target_length:
            # Pad the audio with zeros at the end
            padding = np.zeros(target_length - audio_length)
            padded_audio = np.concatenate((audio, padding))
            return padded_audio

        elif audio_length > target_length:
            # Truncate the audio to the target length
            truncated_audio = audio[:target_length]
            return truncated_audio

        else:
            # If the audio length is already equal to the target length, return the original audio
            return audio

    def compute_features(self):
        """Compute the embeddings of the audio signal using the SampleCNN model."""
        audio = self.pad_or_truncate_audio(self._audio, self.target_length)
        # convert it to a 2D mono tensor of shape (1, samples), you could use the torch.from_numpy() and unsqueeze() functions as follows:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        embedding = self.model(audio_tensor)
        embedding_np = self.detach_to_cpu(embedding)

        N = embedding_np.shape[1] // config.hop_size
        F = embedding_np.shape[-1]
        embedding_reshape = embedding_np.reshape(N, F)

        return embedding_reshape

    def detach_to_cpu(self, embedding):
        embedding_np = embedding.detach().cpu().numpy()
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
