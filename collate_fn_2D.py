import torch
import torch.nn.functional as F

def pad_waveform(waveform, shape):
    padded_waveform = F.pad(waveform, (0, shape[-1] - waveform.shape[-1], 0, shape[-2] - waveform.shape[-2]), "constant", 0)
    return padded_waveform

def collate_fn(batch):
    anchors = []
    positives = []
    negatives = []

    max_shape = tuple(
        max(
            item["anchor"].shape[i],
            item["positive"].shape[i],
            item["negative"].shape[i],
        )
        for i in range(3)  # assuming (channels, frequency bins, time frames) shape
    )
    # Pad all waveforms to the maximum length in the batch
    for item in batch:
        anchors.append(pad_waveform(item["anchor"], max_shape))
        positives.append(pad_waveform(item["positive"], max_shape))
        negatives.append(pad_waveform(item["negative"], max_shape))

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    # Online triplet mining: select the hardest negative for each anchor-positive pair
    anchor_positive_distance = (anchors - positives).pow(2).sum(dim=(1, 2)).sqrt()
    anchor_negative_distance = (
        (anchors.unsqueeze(2) - negatives.unsqueeze(1)).pow(2).sum(dim=(3, 4)).sqrt()
    )
    hardest_negative_indices = torch.argmax(
        anchor_negative_distance - anchor_positive_distance.unsqueeze(2), dim=2
    )
    hardest_negatives = torch.cat(
        [
            negatives[i, idx, :, :].unsqueeze(0)
            for i, idx in enumerate(hardest_negative_indices)
        ]
    )

    return anchors, positives, hardest_negatives
