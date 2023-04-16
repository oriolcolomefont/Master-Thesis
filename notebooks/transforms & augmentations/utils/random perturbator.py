import augment
import torch
import torchaudio
import torchaudio.sox_effects as sox
import numpy as np
from IPython.display import Audio


# Load data from file, apply random speed perturbation


class RandomPerturbationFile(torch.utils.data.Dataset):
    """Given flist, apply random speed perturbation
    Suppose all the input files are at least one second long.
    """

    def __init__(self, flist: List[str], sample_rate: int):
        super().__init__()
        self.flist = flist
        self.sample_rate = sample_rate
        self.rng = None

    def __getitem__(self, index):
        speed = self.rng.uniform(0.5, 2.0)
        effects = [
            ["gain", "-n", "-10"],  # apply 10 db attenuation
            ["remix", "-"],  # merge all the channels
            ["speed", f"{speed:.5f}"],  # duration is now 0.5 ~ 2.0 seconds.
            ["rate", f"{self.sample_rate}"],
            ["pad", "0", "1.5"],  # add 1.5 seconds silence at the end
            ["trim", "0", "2"],  # get the first 2 seconds
        ]
        waveform, _ = sox.apply_effects_file(self.flist[index], effects)
        return waveform

    def __len__(self):
        return len(self.flist)


dataset = RandomPerturbationFile(file_list, sample_rate=8000)

loader = torch.utils.data.DataLoader(dataset, batch_size=32)

for batch in loader:
    pass
