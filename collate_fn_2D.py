import torch
import torch.nn.functional as F


def collate_fn(batch, loss_type):
    if loss_type == "triplet":
        return collate_fn_triplet(batch)
    elif loss_type == "contrastive":
        return collate_fn_contrastive(batch)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


def pad_mel_spectrogram(mel_spectrogram, length):
    padded_mel_spectrogram = F.pad(mel_spectrogram, (0, length - mel_spectrogram.shape[-1]), "constant", 0)
    return padded_mel_spectrogram


def collate_fn_triplet(batch):
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
        anchors.append(pad_mel_spectrogram(item["anchor"], max_length))
        positives.append(pad_mel_spectrogram(item["positive"], max_length))
        negatives.append(pad_mel_spectrogram(item["negative"], max_length))

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


def collate_fn_contrastive(batch):
    samples1 = []
    samples2 = []
    labels = []

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
        samples1.append(pad_mel_spectrogram(item["anchor"], max_length))
        samples2.append(pad_mel_spectrogram(item["positive"], max_length))
        labels.append(item["label"])

        samples1.append(pad_mel_spectrogram(item["anchor"], max_length))
        samples2.append(pad_mel_spectrogram(item["negative"], max_length))
        labels.append(item["label_neg"])

    samples1 = torch.stack(samples1)
    samples2 = torch.stack(samples2)
    labels = torch.tensor(labels, dtype=torch.float32)

    return samples1, samples2, labels
