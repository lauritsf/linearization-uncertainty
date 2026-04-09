import torch


def compute_ece(logits, targets, n_bins=15):
    """Vectorized Expected Calibration Error (ECE) calculation.

    Args:
        logits: [N, V] tensor of raw scores
        targets: [N] tensor of ground truth class indices
        n_bins: Number of confidence bins

    Returns:
        ece: Scalar tensor representing the weighted average gap between confidence and accuracy.
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        # We use (lower, upper] bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
