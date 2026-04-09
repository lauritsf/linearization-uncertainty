"""
Invariance analysis: evaluate a model's NLL consistency across multiple
random linearizations of the same graph. Computes Average NLL, CV, and
per-token-type ECE to quantify topological uncertainty and linearization bias.

Supports:
  --eval_strategy <name>   Evaluate under a specific linearization strategy.
  --num_graphs -1          Use the entire test set (no sampling).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

from autograph.data import GraphDataset
from autograph.linearization import STRATEGIES


def analyze_invariance():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="QM9")
    parser.add_argument("--num_graphs", type=int, default=50,
                        help="Number of graphs to evaluate (-1 = full test set)")
    parser.add_argument("--num_permutations", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="logs/analysis/invariance")
    parser.add_argument("--eval_strategy", type=str, default=None,
                        help="Linearization strategy for evaluation (e.g. random_order). "
                             "If not set, uses random_order + native.")
    parser.add_argument(
        "--all-eval-strategies", action="store_true",
        help="Evaluate on all 4 strategies (full 4x4 matrix) instead of just random_order + native",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    train_strategy, _ = extract_strategy_and_seed(args.checkpoint)

    # Setup datamodule to get the test dataset
    dm = GraphDataset(
        root="./datasets",
        dataset_names=args.dataset,
        labeled_graph=(args.dataset == "QM9"),
        init_tokenizer=False,
    )
    dm.prepare_data()
    dm.setup("test")

    raw_test_dataset = dm.test_dataset.datasets[0]

    # Select graphs
    if args.num_graphs == -1:
        indices = np.arange(len(raw_test_dataset))
        print(f"Using full test set: {len(indices)} graphs")
    else:
        np.random.seed(42)
        indices = np.random.choice(
            len(raw_test_dataset),
            min(args.num_graphs, len(raw_test_dataset)),
            replace=False,
        )

    # Determine eval strategies
    if args.eval_strategy:
        test_strategies = [args.eval_strategy]
    elif args.all_eval_strategies:
        test_strategies = list(STRATEGIES.keys())
    else:
        test_strategies = ["random_order"]
        if train_strategy != "random_order" and train_strategy in STRATEGIES:
            test_strategies.append(train_strategy)

    print(f"\nEvaluating Training Strategy: {train_strategy}")
    print(f"Test Strategies: {test_strategies}")
    print(
        f"| {'Test Strat':<20} | {'Avg NLL':<10} | {'Std NLL':<10} "
        f"| {'CV':<10} | {'ECE':<10} |"
    )
    print(f"|{'-' * 22}|{'-' * 12}|{'-' * 12}|{'-' * 12}|{'-' * 12}|")

    all_results = []
    pad_id = model.tokenizer.pad

    for test_strat in test_strategies:
        # Override tokenizer strategy
        model.tokenizer.linearization_strategy = test_strat
        model.tokenizer.strategy_params = STRATEGIES[test_strat].kwargs

        strat_nlls = []

        # Accumulate logits/targets for strategy-level ECE
        all_logits_flat = []
        all_targets_flat = []
        all_masks_flat = []
        all_batch_shapes = []

        for idx in tqdm(indices, desc=f"Testing {test_strat}", leave=False):
            orig_transform = raw_test_dataset.transform
            raw_test_dataset.transform = None
            data = raw_test_dataset[idx]
            raw_test_dataset.transform = orig_transform

            tokens_list = []
            token_lengths = []
            for _ in range(args.num_permutations):
                tokens = model.tokenizer(data)
                tokens_list.append(tokens)
                token_lengths.append(len(tokens))

            collate_fn = model.tokenizer.batch_converter()
            graph_losses = []

            for i in range(0, args.num_permutations, args.batch_size):
                batch_tokens = tokens_list[i : i + args.batch_size]
                batch = collate_fn(batch_tokens).to(device)

                logits, seq_nll, targets = forward_pass_nll(model, batch, pad_id)
                bsz, slen, vsz = logits.shape
                graph_losses.extend(seq_nll.cpu().numpy().tolist())

                # Accumulate for ECE (CPU)
                valid = targets != pad_id
                all_logits_flat.append(logits.reshape(-1, vsz).cpu())
                all_targets_flat.append(targets.reshape(-1).cpu())
                all_masks_flat.append(valid.reshape(-1).cpu())
                all_batch_shapes.append((bsz, slen))

            graph_mean = np.mean(graph_losses)
            graph_std = np.std(graph_losses)

            all_results.append(
                {
                    "Train_Strategy": train_strategy,
                    "Test_Strategy": test_strat,
                    "Graph_Idx": idx,
                    "Mean_NLL": graph_mean,
                    "Std_NLL": graph_std,
                    "CV": graph_std / graph_mean if graph_mean > 0 else 0.0,
                    "Mean_Seq_Len": np.mean(token_lengths),
                }
            )

            strat_nlls.extend(graph_losses)

        # --- Compute ECE breakdown ---
        cat_logits = torch.cat(all_logits_flat, dim=0)
        cat_targets = torch.cat(all_targets_flat, dim=0)
        cat_valid = torch.cat(all_masks_flat, dim=0)

        ece_dict = compute_ece_breakdown(
            cat_logits, cat_targets, cat_valid, all_batch_shapes, model.tokenizer
        )

        # Store ECE in each result row for this strategy
        for r in all_results:
            if r["Test_Strategy"] == test_strat:
                r.update(ece_dict)

        # Aggregate stats
        avg_nll = np.mean(strat_nlls)
        strat_df = pd.DataFrame(all_results)
        strat_df = strat_df[strat_df["Test_Strategy"] == test_strat]
        avg_cv = strat_df["CV"].mean()
        avg_std_per_graph = strat_df["Std_NLL"].mean()

        print(
            f"| {test_strat:<20} | {avg_nll:<10.2f} | {avg_std_per_graph:<10.2f} "
            f"| {avg_cv:<10.4f} | {ece_dict['ECE_overall']:<10.4f} |"
        )

        print(f"  ECE breakdown for {test_strat}:")
        for k, v in ece_dict.items():
            print(f"    {k}: {v:.4f}" if not np.isnan(v) else f"    {k}: N/A")

        # Free memory
        del cat_logits, cat_targets, cat_valid
        del all_logits_flat, all_targets_flat, all_masks_flat, all_batch_shapes

    # Save detailed results
    df = pd.DataFrame(all_results)
    csv_name = f"invariance_{args.dataset}_{train_strategy}.csv"
    out_path = Path(args.output_dir) / csv_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    analyze_invariance()
