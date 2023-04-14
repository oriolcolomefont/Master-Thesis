import pandas as pd
import librosa.util
import torchaudio
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class MasterParser:
    def __init__(
        self, clip_duration: float, sample_rate: int, limit: int, base_directory: str
    ):
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.limit = limit
        self.base_directory = base_directory

    def parse(self, directories=None):
        if directories is None:
            # List all directories in the base folder
            directories = [
                os.path.join(self.base_directory, d)
                for d in os.listdir(self.base_directory)
                if os.path.isdir(os.path.join(self.base_directory, d))
            ]

        # Set the minimum length in frames for a valid audio file
        min_length = int(self.clip_duration * self.sample_rate)

        # Define a worker function for multiprocessing
        def worker(directory):
            filtered_files = []
            audio_files = librosa.util.find_files(
                directory, ext=["mp3", "wav", "flac", "ogg", "m4a"], limit=self.limit
            )
            for file in tqdm(audio_files, desc=f"Processing directory {directory}"):
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

        # Create a pool of worker processes
        with Pool(cpu_count()) as pool:
            results = pool.map(worker, directories)

        # Flatten the results
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
