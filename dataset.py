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
        loss_type: str = "triplet",
        sample_rate: int = 44100,
        clip_duration: float = 15.0,
        min_chunk_duration_sec: float = 0.05,
        max_chunk_duration_sec: float = 1.0,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.loss_type = loss_type
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.min_chunk_duration_sec = min_chunk_duration_sec
        self.max_chunk_duration_sec = max_chunk_duration_sec
        self.file_list = self._load_files()

        np.random.seed(seed)

    def _load_files(self):
        # filter based on min_length
        filtered_file_list = []

        min_length = int(self.clip_duration * self.sample_rate)

        file_list = librosa.util.find_files(
            self.root_dir,
            ext=["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"],
        )
        for file in file_list:
            try:
                if torchaudio.info(file).num_frames >= min_length:
                    filtered_file_list.append(file)
            except RuntimeError as e:
                if "Invalid data found when processing input" in str(e):
                    print(f"Skipping invalid file: {file}")
                else:
                    raise e
        return filtered_file_list

    def __len__(self):
        return len(self.file_list)

    def _resample_waveform(self, waveform, current_sample_rate):
        if self.sample_rate != current_sample_rate:
            resampler = T.Resample(current_sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        return waveform, self.sample_rate

    def compute_average_bpm(self):
        total_bpm = 0
        n_files = 0

        for file in self.file_list:
            try:
                # Load the audio file
                waveform, sample_rate = librosa.load(file, sr=self.sample_rate)

                # Estimate the BPM using librosa
                bpm, _ = librosa.beat.tempo(waveform, sr=sample_rate, aggregate=None)
                total_bpm += bpm
                n_files += 1
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        # Calculate the average BPM
        average_bpm = total_bpm / n_files

        return average_bpm

    def __getitem__(self, index):
        filename = self.file_list[index]
        metadata = torchaudio.info(filename)
        num_frames = int(self.clip_duration * metadata.sample_rate)
        frame_offset = np.random.randint(0, metadata.num_frames - num_frames)

        waveform, _ = torchaudio.load(
            filename, frame_offset=frame_offset, num_frames=num_frames
        )
        waveform = waveform.mean(dim=0, keepdim=True)  # convert stereo to mono

        # Resample the waveform if the resample parameter is set, otherwise use the default sample rate
        waveform, _ = self._resample_waveform(waveform, metadata.sample_rate)

        # generate anchor, positive from anchor and negative from anchor
        anchor = waveform
        positive = self.generate_positive(anchor)
        negative = self.generate_negative(anchor)

        if self.loss_type == "triplet":
            return {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
            }
        elif self.loss_type == "contrastive":
            # return pairs of samples and a label indicating if they are similar or dissimilar
            return {
                "anchor": anchor,
                "positive": positive,
                "label": torch.tensor(0, dtype=torch.float32),  # Similar pair
                "negative": negative,
                "label_neg": torch.tensor(1, dtype=torch.float32),  # Dissimilar pair
            }
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

    def add_noise_with_snr(self, waveform, snr_range):
        # Generate white noise
        noise = torch.normal(0, 1, waveform.shape)

        # Calculate signal and noise power
        signal_power = torch.sum(waveform**2)
        noise_power = torch.sum(noise**2)

        # Calculate the scaling factor for the noise
        scale_factor = torch.sqrt(signal_power / (noise_power * 10 ** (snr_range / 10)))

        # Scale the noise
        scaled_noise = noise * scale_factor

        # Add noise to the signal
        noisy_waveform = waveform + scaled_noise

        return noisy_waveform

    def generate_positive(self, anchor):
        # Define the effect parameters using numpy
        gain = np.random.randint(-12, 0)
        pitch = np.random.randint(-1200, 1200)
        reverb_params = [np.random.randint(0, 100)] * 3
        chorus_params = [
            round(np.random.uniform(0.1, 1.0), 1),
            round(np.random.uniform(0.1, 1.0), 1),
            np.random.randint(20, 55),
            round(np.random.uniform(0.1, 0.9), 1),
            round(np.random.uniform(0.1, 2.0), 2),
            np.random.randint(2, 5),
            np.random.choice(["-s", "-t"]),
        ]
        drive = np.random.randint(0, 30)
        stretch = round(np.random.uniform(0.9, 1.1), 1)
        speed = np.random.uniform(0.9, 1.1)
        tremolo_speed = np.random.uniform(0.1, 100)
        tremolo_depth = np.random.randint(1, 101)
        snr_range = np.random.randint(12, 100)

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
        return self.add_noise_with_snr(positive, snr_range=snr_range)

    def generate_negative(self, anchor):
        # Get anchor length and duration
        anchor_length = anchor.shape[-1]
        anchor_duration = anchor_length / self.sample_rate

        # Determine the number of chunks based on minimum chunk duration
        n_chunks = int(anchor_duration // self.min_chunk_duration_sec)

        # Calculate the minimum and maximum chunk lengths in samples
        min_chunk_length = int(self.min_chunk_duration_sec * self.sample_rate)
        max_chunk_length = int(self.max_chunk_duration_sec * self.sample_rate)

        # Generate random chunk lengths
        chunk_lengths = np.random.randint(
            min_chunk_length, max_chunk_length + 1, size=n_chunks - 1
        )
        chunk_lengths = np.append(
            chunk_lengths, anchor_length - np.sum(chunk_lengths)
        )

        # Split the anchor clip into chunks
        chunks = [
            anchor[..., start : start + length].clone().detach()
            for start, length in zip(
                np.cumsum(np.insert(chunk_lengths, 0, 0)), chunk_lengths
            )
        ]

        # Shuffle the chunks
        np.random.shuffle(chunks)

        # Concatenate the shuffled chunks to create the negative example
        negative = torch.cat(chunks, dim=-1)

        # Check if the positive and negative examples have the same length
        if anchor.shape != negative.shape:
            raise ValueError(
                f"Input positive and output negative have different shapes. Scrambling the positive sample went wrong: {positive.shape} vs {negative.shape}"
            )

        return negative
