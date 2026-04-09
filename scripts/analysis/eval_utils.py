"""Shared utilities for invariance and generation evaluation scripts.

Provides reusable building blocks:
- Model loading with metrics bypass
- Strategy/seed extraction from checkpoint paths
- Forward pass with per-sequence NLL
- ECE computation with token-type breakdown (including new-node/revisit)
- Token-type mask construction
"""

import pathlib

import torch
import torch.serialization
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from autograph.ece import compute_ece
from autograph.models.seq_models import SequenceModel

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, device: torch.device | str = "cuda"):
    """Load a SequenceModel checkpoint with metrics instantiation bypassed."""
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    SequenceModel.instantiate_metrics = lambda self: None
    model = SequenceModel.load_from_checkpoint(checkpoint, weights_only=False)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

STRATEGY_NAMES = [
    "random_order", "min_degree_first", "max_degree_first", "anchor_expansion",
]


def extract_strategy_and_seed(checkpoint_path: str) -> tuple[str, int]:
    """Extract (strategy, seed) from checkpoint path.

    Expected format: logs/train/QM9/{strategy}/llama-s/{seed}/runs/...
    """
    parts = pathlib.Path(checkpoint_path).parts
    strategy = None
    seed = None

    for i, part in enumerate(parts):
        if part in STRATEGY_NAMES:
            strategy = part
            if i + 2 < len(parts):
                try:
                    seed = int(parts[i + 2])
                except ValueError:
                    pass
            break

    if strategy is None or seed is None:
        raise ValueError(
            f"Could not extract strategy and seed from: {checkpoint_path}"
        )
    return strategy, seed


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_pass_nll(model, batch, pad_id):
    """Run forward pass and return logits, per-sequence NLL, and targets.

    Args:
        model: SequenceModel (already on device, in eval mode).
        batch: [B, L] token tensor (already on device).
        pad_id: padding token id.

    Returns:
        logits: [B, L-1, V] raw logits.
        seq_nll: [B] sum-of-token NLL per sequence.
        targets: [B, L-1] target token ids.
    """
    x, y = batch[:, :-1], batch[:, 1:]

    with torch.no_grad():
        logits = model.model(x)  # [B, L-1, V]
        bsz, slen, vsz = logits.shape

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vsz),
            y.reshape(-1),
            ignore_index=pad_id,
            reduction="none",
        ).reshape(bsz, slen)

        seq_nll = loss.sum(dim=1)  # [B]

    return logits, seq_nll, y


# ---------------------------------------------------------------------------
# Token-type masks
# ---------------------------------------------------------------------------

def make_token_type_masks(targets, valid_mask, tokenizer):
    """Build boolean masks for each token type.

    Args:
        targets: [N] flat target tensor.
        valid_mask: [N] bool tensor (True = non-pad).
        tokenizer: model tokenizer with offset attributes.

    Returns:
        dict mapping name -> bool tensor of shape [N].
    """
    y = targets
    idx_off = tokenizer.idx_offset
    node_off = tokenizer.node_idx_offset
    edge_off = tokenizer.edge_idx_offset

    return {
        "node_index": (y >= idx_off) & (y < node_off) & valid_mask,
        "node_label": (y >= node_off) & (y < edge_off) & valid_mask,
        "edge_label": (y >= edge_off) & valid_mask,
        "special": (y < idx_off) & valid_mask,
    }


def make_new_node_revisit_masks(cat_targets, cat_valid, all_batch_shapes, tokenizer):
    """Detect new-node vs revisit within node_index tokens.

    A "new node" is the first occurrence of each node index in a sequence
    (detected via running-max heuristic).

    Returns:
        (new_node_mask, revisit_mask) — bool tensors of shape [N].
    """
    idx_off = tokenizer.idx_offset
    node_off = tokenizer.node_idx_offset

    new_node_mask = torch.zeros(cat_targets.shape[0], dtype=torch.bool)
    revisit_mask = torch.zeros(cat_targets.shape[0], dtype=torch.bool)

    flat_offset = 0
    for bsz, slen in all_batch_shapes:
        for seq_i in range(bsz):
            start = flat_offset + seq_i * slen
            seq_targets = cat_targets[start : start + slen]
            seen_max = -1
            for t_pos in range(slen):
                tok = seq_targets[t_pos].item()
                if idx_off <= tok < node_off:
                    node_idx = tok - idx_off
                    if node_idx > seen_max:
                        new_node_mask[start + t_pos] = True
                        seen_max = node_idx
                    else:
                        revisit_mask[start + t_pos] = True
        flat_offset += bsz * slen

    return new_node_mask & cat_valid, revisit_mask & cat_valid


# ---------------------------------------------------------------------------
# ECE breakdown
# ---------------------------------------------------------------------------

def compute_ece_breakdown(cat_logits, cat_targets, cat_valid, all_batch_shapes, tokenizer):
    """Compute overall + per-token-type ECE.

    Args:
        cat_logits: [N, V] concatenated logits (CPU).
        cat_targets: [N] concatenated targets (CPU).
        cat_valid: [N] valid mask (CPU).
        all_batch_shapes: list of (B, L) tuples for new-node detection.
        tokenizer: model tokenizer.

    Returns:
        dict with keys ECE_overall, ECE_node_index, ECE_node_label,
        ECE_edge_label, ECE_special, ECE_new_node, ECE_revisit.
    """
    ece_dict = {}

    # Overall
    ece_dict["ECE_overall"] = compute_ece(
        cat_logits[cat_valid], cat_targets[cat_valid]
    ).item()

    # Per token type
    type_masks = make_token_type_masks(cat_targets, cat_valid, tokenizer)
    for name, tmask in type_masks.items():
        key = f"ECE_{name}"
        if tmask.any():
            ece_dict[key] = compute_ece(cat_logits[tmask], cat_targets[tmask]).item()
        else:
            ece_dict[key] = float("nan")

    # New-node vs revisit
    new_node_valid, revisit_valid = make_new_node_revisit_masks(
        cat_targets, cat_valid, all_batch_shapes, tokenizer
    )
    for name, mask in [("ECE_new_node", new_node_valid), ("ECE_revisit", revisit_valid)]:
        if mask.any():
            ece_dict[name] = compute_ece(cat_logits[mask], cat_targets[mask]).item()
        else:
            ece_dict[name] = float("nan")

    return ece_dict
