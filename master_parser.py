import numpy as np
import pandas as pd
import librosa.util
import torchaudio
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class MasterParser:
    def __init__(
        self,
        name: str,
        min_duration: float,
        limit: int,
        base_directory: str,
        last_file_path: str = None,
    ):
        self.name = name
        self.min_duration = min_duration
        self.limit = limit
        self.base_directory = base_directory
        self.total_files = 0
        self.last_file_path = last_file_path

    def worker(self, directory, min_duration, limit):
        filtered_files = []
        audio_files = librosa.util.find_files(
            directory, ext=["mp3", "wav", "flac", "ogg", "m4a"], limit=limit
        )
        for i, file in tqdm(
            enumerate(audio_files),
            desc=f"Processing directory {directory}",
            total=len(audio_files),
        ):
            try:
                info = torchaudio.info(file)
                min_length = int(
                    min_duration * info.sample_rate
                )  # Calculate min_length based on the actual sample rate
                if info.num_frames >= min_length:
                    filtered_files.append(file)
            except Exception as e:
                print(f"Skipping invalid file: {file} due to error: {e}")
                continue

            if (i + 1) % (len(audio_files) // 10) == 0:
                self.total_files += len(filtered_files)
                progress = round((i + 1) / len(audio_files) * 100)
                script_directory = os.path.dirname(os.path.abspath(__file__))
                csv_file_name = os.path.join(
                    script_directory,
                    f"{self.name}_limit={self.limit if self.limit else 'all'}_progress{progress}.csv",
                )
                npy_file_name = os.path.join(
                    script_directory,
                    f"{self.name}_limit={self.limit if self.limit else 'all'}_progress{progress}.npy",
                )
                pd.DataFrame(filtered_files, columns=["file_path"]).to_csv(
                    csv_file_name, index=False
                )
                np.save(npy_file_name, filtered_files)

        self.total_files += len(filtered_files)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        csv_file_name = os.path.join(
            script_directory,
            f"{self.name}_limit={self.limit if self.limit else 'all'}_progress100.csv",
        )
        npy_file_name = os.path.join(
            script_directory,
            f"{self.name}_limit={self.limit if self.limit else 'all'}_progress100.npy",
        )
        pd.DataFrame(filtered_files, columns=["file_path"]).to_csv(
            csv_file_name, index=False
        )
        np.save(npy_file_name, filtered_files)

        return filtered_files

    def parse(self, directories=None):
        if self.last_file_path is not None:
            audio_df = pd.read_csv(self.last_file_path)
            processed_directories = list(
                set(os.path.dirname(f) for f in audio_df["file_path"].values)
            )
            if directories is not None:
                directories = list(set(directories) - set(processed_directories))
            else:
                directories = processed_directories
        else:
            directories = [
                os.path.join(self.base_directory, d)
                for d in os.listdir(self.base_directory)
                if os.path.isdir(os.path.join(self.base_directory, d))
            ]

        with Pool(cpu_count()) as pool:
            # Wrap the arguments for starmap with tqdm to display overall progress
            args = [
                (directory, self.min_duration, self.limit) for directory in directories
            ]
            results = list(tqdm(pool.starmap(self.worker, args), total=len(args)))

        filtered_files = [file for sublist in results for file in sublist]

        if self.last_file_path is not None:
            audio_df = audio_df.append(
                pd.DataFrame(filtered_files, columns=["file_path"]), ignore_index=True
            )
        else:
            audio_df = pd.DataFrame(filtered_files, columns=["file_path"])

        script_directory = os.path.dirname(os.path.abspath(__file__))
        csv_file_name = os.path.join(
            script_directory,
            f"{self.name}_limit={self.limit if self.limit else 'all'}.csv",
        )
        npy_file_name = os.path.join(
            script_directory,
            f"{self.name}_limit={self.limit if self.limit else 'all'}.npy",
        )

        audio_df.to_csv(csv_file_name, index=False)
        np.save(npy_file_name, filtered_files)

        print(f"Total files processed: {self.total_files}")
        print(audio_df.head())

        return audio_df, npy_file_name
