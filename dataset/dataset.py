import os
import torchaudio
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = os.path.join(self.root_dir, self.file_list[index])
        waveform, sample_rate = torchaudio.load(filename)
        return waveform, sample_rate
