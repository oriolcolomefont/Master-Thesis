from model import TripletNet, SampleCNN
from msaf import config
from msaf.base import Features


class Embedding(Features):
    def __init__(
        self, file_struct, feat_type, sr=config.sample_rate, hop_length=config.hop_size
    ):
        super().__init__(
            file_struct=file_struct, sr=sr, hop_length=hop_length, feat_type=feat_type
        )

        self.model = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3], supervised=False, out_dim=128
        )
        # .load_from_checkpoint('/home/oriol_colome_font_epidemicsound_/Master-Thesis/runs/epoch=9-step=560.ckpt')
        # self.model = TripletNet.load_from_checkpoint('/home/oriol_colome_font_epidemicsound_/Master-Thesis/runs/epoch=9-step=560.ckpt')
        self.model.eval()

    @classmethod
    def get_id(cls):
        """Identifier of these features."""
        return "embedding"

    def compute_features(self):
        # extract embedding
        print(self._audio.shape)
        embedding = self.model(self._audio)
        return embedding


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
