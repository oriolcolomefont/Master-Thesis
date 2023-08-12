import torch
import torch.nn.functional as F


def collate_fn(batch, loss_type):
    """
    A collate function for creating batches of data for training siamese/triplet network models.

    Args:
        batch (list): A list of dictionaries where each dictionary represents a data sample with keys:
            - "anchor" (torch.Tensor): The anchor waveform tensor.
            - "positive" (torch.Tensor): The positive waveform tensor.
            - "negative" (torch.Tensor): The negative waveform tensor.
            - "label" (int): Label for the anchor-positive pair.
            - "label_neg" (int): Label for the anchor-negative pair.
        loss_type (str): The type of loss function to use. Can be "triplet" or "contrastive".

    Returns:
        torch.Tensor or tuple: Depending on the loss type, returns either a tuple containing tensors for:
            - "triplet" loss: (anchors, positives, hardest_negatives)
            - "contrastive" loss: (samples1, samples2, labels)
        or a single tensor for "contrastive" loss: labels.

    Raises:
        ValueError: If an invalid loss type is provided.

    """
    if loss_type == "triplet":
        return collate_fn_triplet(batch)
    elif loss_type == "contrastive":
        return collate_fn_contrastive(batch)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


def pad_waveform(waveform, length):
    padded_waveform = F.pad(waveform, (0, length - waveform.shape[-1]), "constant", 0)
    return padded_waveform


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
        samples1.append(pad_waveform(item["anchor"], max_length))
        samples2.append(pad_waveform(item["positive"], max_length))
        labels.append(item["label"])

        samples1.append(pad_waveform(item["anchor"], max_length))
        samples2.append(pad_waveform(item["negative"], max_length))
        labels.append(item["label_neg"])

    samples1 = torch.stack(samples1)
    samples2 = torch.stack(samples2)
    labels = torch.tensor(labels, dtype=torch.float32)

    return samples1, samples2, labels
