import torch
import torchaudio
import librosa
import concurrent.futures
from tqdm import tqdm
from model import TripletNet


def process_audio_file(audio_file):
    try:
        audio, sr = torchaudio.load(audio_file)
    except Exception as e:
        print(f"Error loading audio file '{audio_file}': {e}")
        return None

    # Calculate the duration of the audio
    duration = audio.shape[1] / sr

    # Skip the audio if its duration is less than 3 seconds
    if duration < 3:
        print(
            f"Discarding audio file '{audio_file}' due to {duration} shorter than 3 seconds."
        )
        return None

    audio = audio.unsqueeze(0)
    audio_embedding = model(audio)
    similarity = torch.cdist(input_audio_embedding, audio_embedding, p=2).item()
    return (audio_file, similarity)


# Load the checkpoint file
CKPT_PATH = "./checkpoints/run-valiant-darkness-40-2023-05-02-epoch=09-val_loss=0.18-triplet.ckpt"
checkpoint = torch.load(CKPT_PATH)

# Create the model
model = TripletNet(
    strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
    supervised=False,
    out_dim=128,
    loss_type="triplet",
)

# Copy parameters and buffers from state_dict into this module and its descendants.
model.load_state_dict(checkpoint["state_dict"])
print("Model loaded")

# Load and preprocess the input audio
input_audio_path = "./datasets/GTZAN/GTZAN train/blues/blues.00049.wav"
input_audio, sampling_rate = torchaudio.load(input_audio_path)
input_audio = input_audio.mean(dim=0, keepdim=True)  # convert stereo to mono
input_audio = input_audio.unsqueeze(0)

print("Input audio loaded")
print(input_audio.shape)

# Obtain embeddings for the input audio
input_audio_embedding = model(input_audio[:, :, : 15 * sampling_rate])
print("Input audio embedding obtained: ", input_audio_embedding.shape)

# Calculate the similarity between the input audio and the audio files in the folder
folder_path = "./datasets/GTZAN/GTZAN train/blues"
audio_files = librosa.util.find_files(
    folder_path,
    ext=["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"],
)
print(f"Found {len(audio_files)} audio files in the specified folder path.")

with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit the tasks to the executor and collect the futures
    futures = [
        executor.submit(process_audio_file, audio_file) for audio_file in audio_files
    ]

    # Create a tqdm progress bar
    progress_bar = tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Processing audio files",
    )

    # Retrieve the results from the futures
    results = [future.result() for future in progress_bar]

    # Filter out None results and collect the similarities
    similarities = [result for result in results if result is not None]

# Sort the audio files by their similarity to the input audio
similarities.sort(key=lambda x: x[1], reverse=False)

# Display the sorted audio files and their similarity scores
for audio_file, similarity in similarities:
    print(f"File: {audio_file}, Similarity score: {similarity}")

# Print the most similar audio file and its similarity score
most_similar_audio, most_similar_similarity = similarities[0]
print(
    f"\nMost similar audio file: {most_similar_audio}, Similarity score: {most_similar_similarity}"
)
