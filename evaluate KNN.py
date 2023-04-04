import os
import torch
import torchaudio
import librosa
from torchaudio.transforms import Resample
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Audio, display

from model import TripletNet, SampleCNN

# Load the checkpoint file
CKPT_PATH = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/runs wandb/epoch=0-step=28.ckpt"

# all init args were saved to the checkpoint
checkpoint = torch.load(CKPT_PATH)
print(checkpoint.keys())

# Preprocess the input audio
input_audio_path = librosa.example('nutcracker')
input_audio, sampling_rate = torchaudio.load(input_audio_path)
input_audio = input_audio.unsqueeze(0)
print("Input audio loaded")

# Obtain embeddings for the input audio
input_audio_embedding = new_model(input_audio).detach().numpy()
print("Input audio embedding obtained")

# Calculate the similarity between the input audio and the audio files in the folder
folder_path = "path/to/your/folder/with/audio/files"
audio_files = librosa.util.find_files(
            folder_path,
            ext=["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"],
        )

similarity_scores = []
for audio_file in audio_files:
    audio_path = os.path.join(folder_path, audio_file)

    # Check if the file is an audio file
    try:
        torchaudio.info(audio_path)
    except RuntimeError:
        continue

    audio, sr = torchaudio.load(audio_path)

    # Resample the audio if the sampling rate is different
    if sr != sampling_rate:
        resampler = Resample(sr, sampling_rate)
        audio = resampler(audio)

    audio = audio.unsqueeze(0)
    audio_embedding = model(audio).detach().numpy()
    similarity = cosine_similarity(input_audio_embedding, audio_embedding)
    similarity_scores.append((audio_file, similarity[0][0]))

# Retrieve the audio file with the highest similarity
most_similar_audio_file = max(similarity_scores, key=lambda x: x[1])[0]
print(f"Most similar audio file: {most_similar_audio_file}")

# Display the most similar audio
most_similar_audio_path = os.path.join(folder_path, most_similar_audio_file)
display(Audio(most_similar_audio_path))
