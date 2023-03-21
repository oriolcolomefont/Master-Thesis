import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
import torchaudio.sox_effects as sox


class MyDataset(Dataset):
    def __init__(self, root_dir, min_length):
        self.root_dir = root_dir
        self.min_length = min_length
        self.file_list = self._load_files()
    
    def _load_files(self):
        filtered_file_list = []
        file_list = librosa.util.find_files(
            root_dir, ext=["mp3", "wav", "ogg", "flac", "aiff", "m4a"]
        )
        for file in file_list:
            waveform, _ = torchaudio.load(file)
                if waveform.shape[-1] >= self.min_length:
                    filtered_file_list.append(file)
        return filtered_file_list
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(
        self,
        index,
        sample_rate=44100,
        min_clip_duration=3,#seconds
        max_clip_duration=20, #seconds
        min_chunk_length=2205,
        max_chunk_length=44100,
    ):
        filename = self.file_list[index]
        num_frames = np.random.randint(
            min_clip_duration * sample_rate, max_clip_duration * sample_rate
        )
        waveform, sample_rate = torchaudio.load(
            filename, frame_offset=0, num_frames=num_frames
        )
        waveform = waveform.mean(dim=0, keepdim=True)  # convert stereo to mono

        # generate anchor, positive from anchor and negative from positive
        anchor = waveform
        positive = self.generate_positive(anchor, sample_rate)
        negative = self.generate_negative(
            positive,
            sample_rate,
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
        )

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
        self, positive, sample_rate, min_chunk_length=2205, max_chunk_length=44100
    ):
        # Calculate the number of chunks
        total_length = positive.shape[-1]
        chunk_lengths = np.random.randint(
            min_chunk_length, max_chunk_length, size=total_length // min_chunk_length
        )
        chunk_lengths = np.concatenate(
            [chunk_lengths, [total_length % min_chunk_length]]
        )
        n_chunks = len(chunk_lengths)

        # Calculate the start and end points of the chunks
        chunk_start = np.cumsum(np.concatenate([[0], chunk_lengths[:-1]]))
        chunk_end = chunk_start + chunk_lengths

        # Shuffle the list of chunks
        indices = np.arange(n_chunks)
        np.random.shuffle(indices)
        chunk_start = chunk_start[indices]
        chunk_end = chunk_end[indices]

        # Concatenate the shuffled chunks
        negative = np.concatenate(
            [positive[start:end] for start, end in zip(chunk_start, chunk_end)]
        )

        # Ensure the negative clip is the same length as the positive
        negative = librosa.util.fix_length(negative, size=total_length)
        return negative
