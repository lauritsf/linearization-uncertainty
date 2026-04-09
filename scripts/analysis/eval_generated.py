"""Evaluate generated molecules with permutation-based scoring.

For each valid generated molecule:
  1. Compute "birth NLL" by scoring the original generated token sequence
  2. Decode to graph, re-linearize K times, compute permutation eval (NLL, CV, ECE)

Reads:
  - generated_tokens.pt (from generate_molecules.py)
  - generated_summary.csv (for validity flags)

Outputs:
  - generated_eval.csv — per-molecule birth NLL + permutation eval metrics
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Allow sibling imports (eval_utils)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_utils import (
    compute_ece_breakdown,
    extract_strategy_and_seed,
    forward_pass_nll,
    load_model,
)

from autograph.linearization import STRATEGIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with generated_tokens.pt and generated_summary.csv")
    parser.add_argument("--num_permutations", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to data_dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    train_strategy, seed = extract_strategy_and_seed(args.checkpoint)
    print(f"Model: {train_strategy}/seed_{seed} on {device}")

    # Use native strategy for eval
    model.tokenizer.linearization_strategy = train_strategy
    model.tokenizer.strategy_params = STRATEGIES[train_strategy].kwargs

    pad_id = model.tokenizer.pad

    # Load saved tokens and summary
    tokens_path = Path(args.data_dir) / "generated_tokens.pt"
    summary_path = Path(args.data_dir) / "generated_summary.csv"

    raw_sequences = torch.load(tokens_path, weights_only=False)
    print(f"Loaded {len(raw_sequences)} token sequences")

    # Read validity from summary
    valid_flags = []
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            valid_flags.append(row["valid"] == "True")

    collate_fn = model.tokenizer.batch_converter()
    results = []
    perm_nlls_list = []   # list of lists: one entry per audited molecule
    valid_idxs = []       # mol_idx for each audited entry

    for mol_idx in tqdm(range(len(raw_sequences)), desc="Auditing generated molecules"):
        seq = raw_sequences[mol_idx]
        if not isinstance(seq, torch.Tensor):
            continue

        # --- Birth NLL: score the original generated sequence ---
        birth_batch = collate_fn([seq]).to(device)
        _, birth_nll, _ = forward_pass_nll(model, birth_batch, pad_id)
        birth_nll_val = birth_nll[0].item()

        row = {
            "idx": mol_idx,
            "seq_len": len(seq),
            "valid": valid_flags[mol_idx] if mol_idx < len(valid_flags) else False,
            "birth_nll": birth_nll_val,
        }

        # --- Permutation evaluation (only for valid molecules) ---
        if row["valid"]:
            try:
                graph = model.tokenizer.decode(seq)
            except Exception:
                row["perm_mean_nll"] = float("nan")
                row["perm_cv"] = float("nan")
                results.append(row)
                continue

            # Generate K permutations
            tokens_list = []
            for _ in range(args.num_permutations):
                try:
                    tokens = model.tokenizer(graph)
                    tokens_list.append(tokens)
                except Exception:
                    pass

            if len(tokens_list) < 2:
                row["perm_mean_nll"] = float("nan")
                row["perm_cv"] = float("nan")
                results.append(row)
                continue

            perm_losses = []
            all_logits_flat = []
            all_targets_flat = []
            all_masks_flat = []
            all_batch_shapes = []

            for i in range(0, len(tokens_list), args.batch_size):
                batch_tokens = tokens_list[i : i + args.batch_size]
                batch = collate_fn(batch_tokens).to(device)
                logits, seq_nll, targets = forward_pass_nll(model, batch, pad_id)
                bsz, slen, vsz = logits.shape
                perm_losses.extend(seq_nll.cpu().numpy().tolist())

                valid = targets != pad_id
                all_logits_flat.append(logits.reshape(-1, vsz).cpu())
                all_targets_flat.append(targets.reshape(-1).cpu())
                all_masks_flat.append(valid.reshape(-1).cpu())
                all_batch_shapes.append((bsz, slen))

            perm_mean = np.mean(perm_losses)
            perm_std = np.std(perm_losses)
            row["perm_mean_nll"] = perm_mean
            row["perm_std_nll"] = perm_std
            row["perm_cv"] = perm_std / perm_mean if perm_mean > 0 else 0.0
            row["resamp_seq_len"] = np.mean([len(t) for t in tokens_list])

            # Accumulate raw per-permutation NLLs for K-sweep analysis
            perm_nlls_list.append(perm_losses)
            valid_idxs.append(mol_idx)

            # ECE breakdown
            cat_logits = torch.cat(all_logits_flat, dim=0)
            cat_targets = torch.cat(all_targets_flat, dim=0)
            cat_valid = torch.cat(all_masks_flat, dim=0)
            ece_dict = compute_ece_breakdown(
                cat_logits, cat_targets, cat_valid, all_batch_shapes, model.tokenizer
            )
            row.update(ece_dict)

        results.append(row)

    # Save
    out_path = Path(output_dir) / "generated_eval.csv"
    if results:
        fieldnames = list(results[0].keys())
        # Ensure all keys are captured
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Save raw per-permutation NLL matrix for K-sweep analysis
    output_dir_path = Path(output_dir)
    if perm_nlls_list:
        np.save(output_dir_path / "perm_nlls.npy",
                np.array(perm_nlls_list, dtype=np.float32))
        np.save(output_dir_path / "perm_nlls_idx.npy",
                np.array(valid_idxs, dtype=np.int64))
        print(f"Saved perm_nlls.npy: shape {np.array(perm_nlls_list).shape}")

    n_valid = sum(1 for r in results if r.get("valid"))
    n_audited = sum(1 for r in results if not np.isnan(r.get("perm_cv", float("nan"))))
    print(f"Saved {len(results)} rows to {out_path}")
    print(f"  Valid: {n_valid}, Audited: {n_audited}")
    mean_birth = np.mean([r["birth_nll"] for r in results if "birth_nll" in r])
    print(f"  Mean birth NLL: {mean_birth:.2f}")


if __name__ == "__main__":
    main()
