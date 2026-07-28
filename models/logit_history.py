import torch


def normalize_sample_ids(sample_ids):
    """Convert a batch of IDs to stable Python integer keys."""
    if sample_ids is None:
        return None
    if torch.is_tensor(sample_ids):
        return sample_ids.detach().cpu().reshape(-1).tolist()
    return [
        int(sample_id.item()) if torch.is_tensor(sample_id) else int(sample_id)
        for sample_id in sample_ids
    ]


def load_history_batch(history, sample_ids, batch_size, num_classes, device):
    """Load matching historical logits and a mask for the IDs that were found."""
    ids = normalize_sample_ids(sample_ids)
    logits = torch.zeros(batch_size, num_classes, device=device)
    valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if ids is None:
        return logits, valid_mask

    for position, sample_id in enumerate(ids):
        previous = history.get(sample_id)
        if previous is not None:
            logits[position] = previous.to(device)
            valid_mask[position] = True
    return logits, valid_mask


def record_history_batch(history, sample_ids, ensemble_logits):
    """Record gathered batch logits on CPU, keyed by real dataset indices."""
    ids = normalize_sample_ids(sample_ids)
    if ids is None:
        return
    if len(ids) != ensemble_logits.size(0):
        raise ValueError(
            "sample_ids and ensemble_logits must contain the same batch size"
        )

    logits = ensemble_logits.detach().cpu()
    for sample_id, logit in zip(ids, logits):
        history[sample_id] = logit.clone()
