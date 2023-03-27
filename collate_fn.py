import torch
import torch.nn.functional as F


def pad_waveform(waveform, length):
    padded_waveform = F.pad(waveform, (0, length - waveform.shape[-1]), "constant", 0)
    return padded_waveform


def collate_fn(batch):
    anchors = []
    positives = []
    negatives = []

    max_length = max(
        max(
            item["anchor"].shape[-1],
            item["positive"].shape[-1],
            item["negative"].shape[-1],
        )
        for item in batch
    )
    # Pad all waveforms to the maximum length in the batch
    for item in batch:
        anchors.append(pad_waveform(item["anchor"], max_length))
        positives.append(pad_waveform(item["positive"], max_length))
        negatives.append(pad_waveform(item["negative"], max_length))

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    # Online triplet mining: select the hardest negative for each anchor-positive pair
    anchor_positive_distance = (anchors - positives).pow(2).sum(dim=2).sqrt()
    anchor_negative_distance = (
        (anchors.unsqueeze(2) - negatives.unsqueeze(1)).pow(2).sum(dim=3).sqrt()
    )
    hardest_negative_indices = torch.argmax(
        anchor_negative_distance - anchor_positive_distance.unsqueeze(2), dim=2
    )
    hardest_negatives = torch.cat(
        [
            negatives[i, idx, :].unsqueeze(0)
            for i, idx in enumerate(hardest_negative_indices)
        ]
    )

    return anchors, positives, hardest_negatives