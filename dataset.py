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
        min_duration,
        resample: int = None,
        default_sample_rate: int = 44100,
        max_clip_duration: int = 5,
        min_chunk_duration_sec: float = 0.05,
        max_chunk_duration_sec: float = 1.0,
    ):
        self.root_dir = root_dir
        self.min_duration = min_duration
        self.resample = resample
        self.default_sample_rate = default_sample_rate
        self.max_clip_duration = max_clip_duration
        self.min_chunk_duration_sec = min_chunk_duration_sec
        self.max_chunk_duration_sec = max_chunk_duration_sec
        self.file_list = self._load_files()
    
    def _load_files(self):
        #filter based on min_length
        filtered_file_list = []
    
        if self.resample is not None:
             min_length = int(self.min_duration * self.resample)
        else:
            min_length = int(self.min_duration * self.default_sample_rate)

        file_list = librosa.util.find_files(
            self.root_dir, ext=['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav'],
        )
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
    
    def _resample_waveform(self, waveform, original_sample_rate):
        if self.resample is not None:
            resampler = T.Resample(original_sample_rate, self.resample)
            waveform = resampler(waveform)
            return waveform, self.resample
        else:
            return waveform, self.default_sample_rate

    def __getitem__(self, index):
        filename = self.file_list[index]
        num_frames = np.random.randint(self.min_duration * self.default_sample_rate, self.max_clip_duration * self.default_sample_rate)
        waveform, sample_rate = torchaudio.load(
            filename, frame_offset=0, num_frames=num_frames
        )
        waveform = waveform.mean(dim=0, keepdim=True)  # convert stereo to mono

        # Resample the waveform if the resample parameter is set, otherwise use the default sample rate
        waveform, sample_rate = self._resample_waveform(waveform, sample_rate)

        # Calculate min and max chunk lengths based on sample rate and duration in seconds
        min_chunk_length = int(min_chunk_duration_sec * sample_rate)
        max_chunk_length = int(max_chunk_duration_sec * sample_rate)

        # generate anchor, positive from anchor and negative from positive
        anchor = waveform
        positive = self.generate_positive(anchor, sample_rate)
        negative = self.generate_negative(
            positive,
            sample_rate,
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
        )

        #print(f"Anchor shape: {anchor.shape}, Positive shape: {positive.shape}, Negative shape: {negative.shape}")
        return {'anchor': anchor, 'positive': positive, 'negative': negative}

    def generate_positive(self, anchor, sample_rate):
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
        positive, _ = sox.apply_effects_tensor(anchor, sample_rate, effects)
        positive = positive.mean(dim=0, keepdim=True)  # convert stereo to mono
        return positive

    def generate_negative(
        self, positive, min_chunk_length, max_chunk_length
        ):
        # Add padding if the positive waveform length is smaller than the max_chunk_length
        if positive.shape[-1] < max_chunk_length:
            padding_size = max_chunk_length - positive.shape[-1]
            positive = torch.cat([positive, torch.zeros(positive.shape[:-1] + (padding_size,))], dim=-1)

        # Split the positive clip into chunks
        chunks = positive.unfold(-1, max_chunk_length, min_chunk_length)
        n_chunks = chunks.shape[-2]

        # Shuffle the chunks
        indices = torch.randperm(n_chunks)
        chunks = chunks[..., indices, :]

        # Concatenate the shuffled chunks
        negative = chunks.reshape(chunks.shape[:-2] + (-1,))
        return negative
