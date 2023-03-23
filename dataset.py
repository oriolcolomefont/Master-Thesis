import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio.sox_effects as sox


class MyDataset(Dataset):
    def __init__(
        self,
        root_dir,
        resample: int = None,
        default_sample_rate: int = 44100,
        min_clip_duration: int = 3,
        max_clip_duration: int = 5,
        min_chunk_duration_sec: float = 0.05,
        max_chunk_duration_sec: float = 1.0,
        seed: int = 42
    ):
        self.root_dir = root_dir
        self.resample = resample
        self.default_sample_rate = default_sample_rate
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.min_chunk_duration_sec = min_chunk_duration_sec
        self.max_chunk_duration_sec = max_chunk_duration_sec
        self.file_list = self._load_files()

        np.random.seed(seed)

    @property
    def sample_rate(self):
        return self.resample or self.default_sample_rate
    
    def _load_files(self):
        #filter based on min_length
        filtered_file_list = []
    
        if self.resample is not None:
            min_length = int(self.min_clip_duration * self.resample)
        else:
            min_length = int(self.min_clip_duration * self.sample_rate)

        file_list = librosa.util.find_files(self.root_dir, ext=['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav'],)
        for file in file_list:
            try:
                waveform, _ = torchaudio.load(file)
                if waveform.shape[-1] >= min_length:
                    filtered_file_list.append(file)
            except RuntimeError as e:
                if "Invalid data found when processing input" in str(e):
                    print(f"Skipping invalid file: {file}")
                else:
                    raise e
        return filtered_file_list

    def __len__(self):
        return len(self.file_list)
    
    def _resample_waveform(self, waveform):
        if self.resample is not None:
            resampler = T.Resample(self.sample_rate, self.resample)
            waveform = resampler(waveform)
            return waveform, self.resample
        else:
            return waveform, self.sample_rate

    def __getitem__(self, index):
        filename = self.file_list[index]
        waveform, _ = torchaudio.load(filename)
        waveform = waveform.mean(dim=0, keepdim=True)  # convert stereo to mono

        # Resample the waveform if the resample parameter is set, otherwise use the default sample rate
        waveform, _ = self._resample_waveform(waveform)

        # generate anchor, positive from anchor and negative from positive
        anchor = waveform
        positive = self.generate_positive(anchor)
        negative = self.generate_negative(positive)

        print(f"Anchor shape: {anchor.shape}, Positive shape: {positive.shape}, Negative shape: {negative.shape}")
        return {'anchor': anchor, 'positive': positive, 'negative': negative}

    def generate_positive(self, anchor):
        # Define the effect parameters using numpy
        gain = np.random.randint(-12, 0)
        pitch = np.random.randint(-1200, 1200)
        reverb_params = [np.random.randint(0, 100)] * 3
        chorus_params = [
            round(np.random.uniform(0.1, 1.0), 1),
            round(np.random.uniform(0.1, 1.0), 1),
            55,
            round(np.random.uniform(0.1, 0.9), 1),
            round(np.random.uniform(0.1, 2.0), 2),
            np.random.randint(2, 5),
            np.random.choice(["-s", "-t"]),
        ]
        drive = np.random.randint(0, 30)
        stretch = round(np.random.uniform(0.8, 1.2), 1)
        speed = np.random.uniform(0.7, 1.3)
        tremolo_speed = np.random.uniform(0.1, 100)
        tremolo_depth = np.random.randint(1, 101)

        # Define the effect chain using f-strings
        effects = [
            ["gain", "-n", f"{gain}"],
            ["chorus", *map(str, chorus_params)],
            ["overdrive", f"{drive}"],
            ["pitch", f"{pitch}"],
            ["reverb", *[str(param) for param in reverb_params]],
            ["speed", f"{speed}"],
            ["stretch", f"{stretch}"],
            ["tremolo", f"{tremolo_speed}", f"{tremolo_depth}"],
        ]
        positive, _ = sox.apply_effects_tensor(anchor, self.sample_rate, effects)
        positive = positive.mean(dim=0, keepdim=True)  # convert stereo to mono
        return positive

    def generate_negative(self, positive):
        # Get positive length and duration
        positive_length = positive.shape[-1]
        positive_duration = positive_length / self.sample_rate

        # Determine the number of chunks based on minimum chunk duration
        n_chunks = int(positive_duration // self.min_chunk_duration_sec)

        # Calculate the minimum and maximum chunk lengths in samples
        min_chunk_length = int(self.min_chunk_duration_sec * self.sample_rate)
        max_chunk_length = int(self.max_chunk_duration_sec * self.sample_rate)

        # Generate random chunk lengths
        chunk_lengths = np.random.randint(min_chunk_length, max_chunk_length + 1, size=n_chunks - 1)
        chunk_lengths = np.append(chunk_lengths, positive_length - np.sum(chunk_lengths))

        # Split the positive clip into chunks and convert them to PyTorch tensors
        chunks = [torch.tensor(positive[..., start:start + length], dtype=torch.float32) for start, length in zip(np.cumsum(np.insert(chunk_lengths, 0, 0)), chunk_lengths)]

        # Shuffle the chunks
        np.random.shuffle(chunks)

        # Concatenate the shuffled chunks to create the negative example
        negative = torch.cat(chunks, dim=-1)

        # Check if the positive and negative examples have the same length
        if positive.shape != negative.shape:
            raise ValueError(f"Input positive and output negative have different shapes: {positive.shape} vs {negative.shape}")

        return negative