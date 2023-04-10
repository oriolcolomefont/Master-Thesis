import torch
import torchaudio
import librosa
from sklearn.metrics.pairwise import cosine_similarity

from model import TripletNet

# Load the checkpoint file
CKPT_PATH = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/checkpoints/example-epoch=19-val_loss=0.07.ckpt"

# all init args were saved to the checkpoint
checkpoint = torch.load(CKPT_PATH)

# Create the model and move it to the GPU
model = TripletNet(
    strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
    supervised=False,
    out_dim=128,
    loss_type="contrastive",
)

# Copy parameters and buffers from state_dict into this module and its descendants.
model.load_state_dict(checkpoint["state_dict"])
print("Model loaded")

# Load and preprocess the input audio
input_audio_path = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/tests/positive_noisy_disco79.wav"
input_audio, sampling_rate = torchaudio.load(input_audio_path)
input_audio = input_audio.mean(dim=0, keepdim=True)  # convert stereo to mono
input_audio = input_audio.unsqueeze(0)

print("Input audio loaded")
print(input_audio.shape)

# Obtain embeddings for the input audio
input_audio_embedding = model(input_audio).detach().numpy()
print("Input audio embedding obtained: ", input_audio_embedding.shape)

# Calculate the similarity between the input audio and the audio files in the folder
folder_path = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/datasets/GTZAN/GTZAN train/disco"
audio_files = librosa.util.find_files(
    folder_path,
    ext=["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"],
)
print(f"Found {len(audio_files)} audio files in the specified folder path.")

similarities = []
for audio_file in audio_files:
    try:
        audio, sr = torchaudio.load(audio_file)
        print(audio.shape)
    except Exception as e:
        print(f"Error loading audio file '{audio_file}': {e}")
        continue

    # Calculate the duration of the audio
    duration = audio.shape[1] / sr

    # Skip the audio if its duration is less than 3 seconds
    if duration < 3:
        print(
            f"Discarding audio file '{audio_file}' due to {duration} shorter than 3 seconds."
        )
        continue

    audio = audio.unsqueeze(0)
    audio_embedding = model(audio).detach().numpy()
    similarity = cosine_similarity(input_audio_embedding, audio_embedding)
    similarities.append((audio_file, similarity[0][0]))

# Sort the audio files by their similarity to the input audio
similarities.sort(key=lambda x: x[1], reverse=True)

# Display the sorted audio files and their similarity scores
for audio_file, similarity in similarities:
    print(f"File: {audio_file}, Similarity: {similarity}")

# Print the most similar audio file and its similarity score
most_similar_audio, most_similar_similarity = similarities[0]
print(
    f"\nMost similar audio file: {most_similar_audio}, Similarity: {most_similar_similarity}"
)
