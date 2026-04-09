import torch

from autograph.ece import compute_ece


def test_ece_basic():
    # Perfectly calibrated case
    # Logits for 2 classes, 100 samples
    # If prob is 0.8, accuracy should be 0.8
    n = 1000
    logits = torch.zeros((n, 2))
    logits[:, 1] = 1.38629  # logit for 0.8 prob (approx)
    # softmax([0, 1.38629]) -> [0.2, 0.8]

    targets = (torch.rand(n) < 0.8).long()

    ece = compute_ece(logits, targets, n_bins=10)
    print(f"Perfectly calibrated (target 0.8) ECE: {ece.item():.4f}")
    # Should be close to 0

    # Overconfident case
    targets_low_acc = (torch.rand(n) < 0.2).long()
    ece_overconf = compute_ece(logits, targets_low_acc, n_bins=10)
    print(f"Overconfident (conf 0.8, acc 0.2) ECE: {ece_overconf.item():.4f}")
    # Should be close to 0.6

    assert ece.item() < 0.1
    assert 0.5 < ece_overconf.item() < 0.7


if __name__ == "__main__":
    test_ece_basic()
