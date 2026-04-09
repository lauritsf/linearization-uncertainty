"""Compute mean sequence length per linearization strategy on QM9 test graphs.

CPU-only — no GPU or model forward pass required. Runtime: < 1 minute.

Output: results/data/seq_lengths_qm9.csv
  columns: strategy, mean_seq_len, std_seq_len, n_graphs
"""

import csv
import pathlib
import statistics
import sys

import numpy as np

# Allow sibling imports (table_utils)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from table_utils import write_latex, write_markdown

from autograph.data import GraphDataset
from autograph.linearization import STRATEGIES

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "results" / "data" / "seq_lengths_qm9.csv"
TABLE_DIR = ROOT / "results" / "tables"

NUM_GRAPHS = 50  # same subset as invariance eval

GEN_SEQ_LEN_DIR = pathlib.Path("logs/analysis/generated_seq_lengths")


def compute_generated_seq_length_stats():
    """Aggregate per-sample generated seq lengths → one row per strategy.

    Reads from logs/analysis/generated_seq_lengths/{strategy}/seed_0/gen_seq_lengths.csv
    Writes to results/data/seq_lengths_generated_qm9.csv
    """
    results = []
    for strat in STRATEGIES:
        # Use seed 0 only (lengths are strategy-determined, not seed-determined)
        csv_path = GEN_SEQ_LEN_DIR / strat / "seed_0" / "gen_seq_lengths.csv"
        if not csv_path.exists():
            print(f"  WARN: Missing {csv_path}")
            continue

        lengths = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                lengths.append(float(row["seq_len"]))

        if lengths:
            mean_len = statistics.mean(lengths)
            std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
            results.append({
                "strategy": strat,
                "mean_seq_len": round(mean_len, 2),
                "std_seq_len": round(std_len, 2),
                "n_graphs": len(lengths),
            })
            print(f"  {strat:<22}  mean={mean_len:.1f}  std={std_len:.1f}  n={len(lengths)}")

    out = ROOT / "docs" / "manuscript" / "data" / "seq_lengths_generated_qm9.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "mean_seq_len", "std_seq_len", "n_graphs"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {out}")


def main():
    root = str(ROOT / "datasets")
    print("Initialising tokenizer (prepare_data reads train split for vocab)...")
    # init_tokenizer=True runs prepare_data() to set max_num_nodes / num_node_types
    dm_tok = GraphDataset(root=root, dataset_names="QM9", labeled_graph=True, init_tokenizer=True)
    dm_tok.prepare_data()          # sets tokenizer vocab sizes; does NOT apply transform yet
    tokenizer = dm_tok.tokenizer

    print("Loading raw QM9 test graphs (no transform)...")
    # init_tokenizer=False → dataset items are raw torch_geometric.data.Data objects
    dm_raw = GraphDataset(root=root, dataset_names="QM9", labeled_graph=True, init_tokenizer=False)
    dm_raw.setup("test")

    raw_test = dm_raw.test_dataset.datasets[0]
    np.random.seed(42)
    indices = np.random.choice(len(raw_test), min(NUM_GRAPHS, len(raw_test)), replace=False)
    graphs = [raw_test[int(i)] for i in indices]
    print(f"  Selected {len(graphs)} test graphs")
    results = []

    for strat_name in STRATEGIES:
        tokenizer.linearization_strategy = strat_name
        tokenizer.strategy_params = STRATEGIES[strat_name].kwargs

        lengths = []
        for g in graphs:
            # coalesce() sorts edge_index so tokenizer's is_coalesced() check
            # doesn't try to call .is_coalesced() on a dense (Strided) tensor,
            # which raises in PyTorch 2.x.
            tokens = tokenizer(g.coalesce())
            lengths.append(len(tokens))

        mean_len = statistics.mean(lengths)
        std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
        results.append({
            "strategy": strat_name,
            "mean_seq_len": round(mean_len, 2),
            "std_seq_len": round(std_len, 2),
            "n_graphs": len(lengths),
        })
        print(f"  {strat_name:<22}  mean={mean_len:.1f}  std={std_len:.1f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "mean_seq_len", "std_seq_len", "n_graphs"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {OUT_PATH}")

    t_headers = ["Strategy", "Mean tokens/graph", "Std", "N graphs"]
    t_rows = [[r["strategy"], r["mean_seq_len"], r["std_seq_len"], r["n_graphs"]] for r in results]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_markdown(TABLE_DIR / "seq_lengths_qm9.md", t_headers, t_rows)
    write_latex(
        TABLE_DIR / "seq_lengths_qm9.tex",
        t_headers, t_rows,
        caption="Mean sequence length per linearization strategy on 50 QM9 test graphs.",
        label="tab:seq_lengths",
    )

    # Aggregate generated sequence lengths if available
    print("\nAggregating generated sequence lengths...")
    compute_generated_seq_length_stats()


if __name__ == "__main__":
    main()
