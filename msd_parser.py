import os
import pandas as pd
import librosa.util
import torchaudio
from tqdm import tqdm
from multiprocessing import Pool

# Set the folder to scan
base_directory = "/mnt/disks/msd/"

# List all directories in the base folder
directories = [
    os.path.join(base_directory, d)
    for d in os.listdir(base_directory)
    if os.path.isdir(os.path.join(base_directory, d))
]


class MSDParser:
    def __init__(self, clip_duration, sample_rate, limit):
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.limit = limit

    def _scan_files(self, directory):
        # Set the minimum length in frames for a valid audio file
        min_length = int(self.clip_duration * self.sample_rate)

        # Find all audio files in the directory
        audio_files = librosa.util.find_files(
            directory, ext=["mp3", "wav", "flac", "ogg", "m4a"], limit=self.limit
        )

        # Filter the audio files based on their duration
        filtered_files = []
        for file in audio_files:
            try:
                info = torchaudio.info(file)
                if info.num_frames >= min_length:
                    filtered_files.append(file)
            except RuntimeError as e:
                if "Invalid data found when processing input" in str(e):
                    print(f"Skipping invalid file: {file}")
                else:
                    raise e

        return filtered_files

    def parse(self):
        # Use multiprocessing to scan files in parallel
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self._scan_files, directories),
                    total=len(directories),
                    desc="Scanning audio files",
                )
            )

        # Flatten the list of filtered files
        filtered_files = [file for sublist in results for file in sublist]

        # Create a pandas DataFrame with the audio file paths
        audio_df = pd.DataFrame(filtered_files, columns=["file_path"])

        # Set the output directory for the CSV file
        output_directory = "/home/oriol_colome_font_epidemicsound_/Master-Thesis"

        # Set the CSV file name with the limit value and output directory
        csv_file_name = os.path.join(
            output_directory, f"msd_audio_files_limit={self.limit}.csv"
        )

        # Save the DataFrame to a CSV file
        audio_df.to_csv(csv_file_name, index=False)

        print(audio_df.head())
        return audio_df
