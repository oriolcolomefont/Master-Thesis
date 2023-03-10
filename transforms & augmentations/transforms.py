import augment
import torch
import torchaudio
import numpy as np

waveform, sample_rate = torchaudio.load("test.wav")

#######################AUGMENT EFFECT CHIAN#####################################

chain = augment.EffectChain()

#######################AUGMENT PARAMETERS#####################################

clip_min, clip_max, decimals = 0.0, 0.6, 2
pitch_min, pitch_max = -1200, +1200  # it ruins the lyrics!!!
reverberance, dumping_factor, room_size = (lambda: np.random.randint(0, 100),) * 3

###########################APPLY THEM TO AUDIO --> RETURN AUDIO DATA#################################

clipped_positive = chain.clip(
    np.round(np.random.uniform(clip_min, clip_max), decimals)
).apply(waveform, src_info={"rate": sample_rate})
pitch_shifted_positive = chain.pitch(
    "-q", lambda: np.random.randint(pitch_min, pitch_max)
).apply(waveform, src_info={"rate": sample_rate})
reverb_positive = (
    chain.reverb(reverberance, dumping_factor, room_size)
    .channels(1)
    .apply(waveform, src_info={"rate": sample_rate})
)

###########################CONCAT EFFECTS#################################
