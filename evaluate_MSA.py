from model import TripletNet, SampleCNN
from msaf import config
from msaf.base import Features
import torch


class Embedding(Features):
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size):
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        
        self.model = TripletNet(SampleCNN())
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.eval()
    
    @classmethod
    def get_id(cls):
        """Identifier of these features."""
        return "embedding"

    def compute_features(self):
        #extract embedding
        embedding = self.model(self._audio)
        return embedding

