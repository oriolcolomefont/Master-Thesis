import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio.sox_effects as sox


class MyDataset(Dataset):
    def __init__(self, root_dir, min_duration, resample: int = None, default_sample_rate: int = 44100):
        self.root_dir = root_dir
        self.min_duration = min_duration
        self.resample = resample
        self.default_sample_rate = default_sample_rate
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

    def __getitem__(
        self,
        index,
        max_clip_duration:int = 5,
        min_chunk_duration_sec:float = 0.05,
        max_chunk_duration_sec:float = 1.0,
    ):
        filename = self.file_list[index]
        num_frames = np.random.randint(
            self.min_duration * self.default_sample_rate, max_clip_duration * self.default_sample_rate
        )
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

        print(f"Anchor shape: {anchor.shape}, Positive shape: {positive.shape}, Negative shape: {negative.shape}")
        return {'anchor': anchor, 'positive': positive, 'negative': negative}


    def generate_positive(self, anchor, sample_rate):
        # function to generate a positive sample from the anchor

        gain_min, gain_max = -12, 0
        speed = round(np.random.uniform(0.5, 2.0), 3)
        pitch_min, pitch_max = -1200, +1200  # it ruins the lyrics!!!
        pitch = np.random.randint(pitch_min, pitch_max)
        reverberance, damping_factor, room_size = (np.random.randint(0, 100),) * 3
        #chorus = round(np.random.uniform(0.01, 1.0), 1)
        effects = [
            [
                "gain",
                "-n",
                str(np.random.randint(gain_min, gain_max)),
            ],  # apply 10 db attenuation
            ["speed", f"{speed:.5f}"],  # duration is now 0.5 ~ 2.0 seconds.
            ["rate", f"{sample_rate}"],
            [
                "chorus",
                "0.9",
                "0.9",
                "55",
                "0.4",
                "0.25",
                "2",
                "-t",
            ],  #'chorus': 'gain-in gain-out delay decay speed depth [ -s | -t ]',
            ["overdrive", "30"],  #'overdrive': '[gain [colour]]',
            [
                "pitch",
                str(pitch),
            ],  #'pitch': 'semitones [octaves [cents]]',
            [
                "reverb",
                str(reverberance),
                str(damping_factor),
                str(room_size),
            ],  #'reverb': '[-w|--wet-only] [reverberance (50%) [HF-damping (50%) [room-scale (100%) [stereo-depth (100%) [pre-delay (0ms) [wet-gain (0dB)]]]]]]',
            ["speed", "2"],  #'speed': 'factor[c]',
            [
                "stretch",
                "1.5",
            ],  #'stretch': 'factor [window fade shift fading]\n       (expansion, frame in ms, lin/..., unit<1.0, unit<0.5)\n       (defaults: 1.0 20 lin ...)',
            ["tremolo", "10", "50"],  #'tremolo': 'speed_Hz [depth_percent]',
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
