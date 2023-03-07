import os
import torchaudio
from torchaudio.datasets.gtzan import gtzan_genres
from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple
class GTZAN(Dataset):
    NUM_LABELS = 50 ### ????
    MULTILABEL = True
    def __init__(self,
                 subset: str,
                 root: str = "data/processed/gtzan",
                 transforms=None,
                 ) -> None:
        super.__init__()
        self.root = root
        self.subset = subset
        self.genres = gtzan_genres
        self.transforms = transforms
        self._get_song_list()
    def _get_song_list(self):
        list_filename = os.path.join(self.root, f'{self.subset}_filtered.txt')
        with open(list_filename) as f:
            lines = f.readlines()
        self.song_list = [line.strip() for line in lines]
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        line = self.song_list[index]
        # get label
        genre = line.split('/')[0]
        label = self.genres.index(genre)
        # get audio
        audio_file = os.path.join(self.root, 'Data/genres_original', line)
        audio, _ = torchaudio.load(audio_file, format='wav')
        if self.transforms:
            audio = self.transforms(audio)
        return audio, label
    def __len__(self):
        return len(self.song_list)