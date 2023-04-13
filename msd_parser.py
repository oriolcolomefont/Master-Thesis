import os
import pandas as pd
import librosa.util

# Set the folder to scan
base_directory = "/mnt/disks/msd/"

# List all directories in the base folder
directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]


# Limit the number of pieces to find
limit = 10000

# Find all audio files in the directories
audio_files = []
for directory in directories:
    audio_files.extend(librosa.util.find_files(directory, ext=['mp3', 'wav', 'flac', 'ogg', 'm4a'], limit=limit))

# Create a pandas DataFrame with the audio file paths
audio_df = pd.DataFrame(audio_files, columns=['file_path'])

#Set the output directory for the CSV file
output_directory = "/home/oriol_colome_font_epidemicsound_/Master-Thesis"

# Set the CSV file name with the limit value and output directory
csv_file_name = os.path.join(output_directory, f"msd_audio_files_limit={limit}.csv")

# Save the DataFrame to a CSV file
audio_df.to_csv(csv_file_name, index=False)

print(audio_df.head())
