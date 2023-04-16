import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import TripletNet

# Load the checkpoint file
CKPT_PATH = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/checkpoints/run-super-universe-299-2023-04-15-epoch=158-val_loss=0.00-triplet.ckpt"

# Load the checkpoint
checkpoint = torch.load(CKPT_PATH)

# Create the model
model = TripletNet(
    strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
    supervised=False,
    out_dim=128,
    loss_type="triplet",
)

# Load the model's state_dict
model.load_state_dict(checkpoint["state_dict"])
model.eval()  # Set the model to evaluation mode
print("Model loaded")

# Load and preprocess the input audio
input_audio_path = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/datasets/SALAMI/audio/559.mp3"
input_audio, sampling_rate = torchaudio.load(input_audio_path)
input_audio = input_audio.mean(dim=0, keepdim=True)  # convert stereo to mono

# Check the minimum input length
min_length = model.strides[0] * model.num_pool_layers * 2
if input_audio.shape[1] < min_length:
    raise ValueError(f"Input audio is too short. Minimum length is {min_length} samples.")

# Extract embeddings
window_size = 3 * sampling_rate
hop_size = 512
num_hops = 1 + (input_audio.shape[1] - window_size) // hop_size

# Check the minimum input length
min_length = model.strides[0] * model.num_pool_layers * 2
if input_audio.shape[1] < min_length:
    raise ValueError(f"Input audio is too short. Minimum length is {min_length} samples.")

# Check compatibility of input length, hop size, and window size
if input_audio.shape[1] < window_size:
    raise ValueError(f"Input audio length ({input_audio.shape[1]}) is shorter than the window size ({window_size}).")

num_hops = 1 + (input_audio.shape[1] - window_size) // hop_size
if num_hops < 1:
    raise ValueError(f"Input audio is too short for the given hop size ({hop_size}) and window size ({window_size}).")

embeddings = []
for i in tqdm(range(num_hops)):
    start = i * hop_size
    end = start + window_size
    audio_slice = input_audio[:, start:end]
    audio_slice = audio_slice.unsqueeze(0)
    embedding = model(audio_slice).detach().numpy().flatten()
    embeddings.append(embedding)
embeddiogram = np.column_stack(embeddings)

# Visualize the embeddiogram
plt.imshow(embeddiogram, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Embedding Dimensions')
plt.title('Embeddiogram')
plt.colorbar()
plt.show()
