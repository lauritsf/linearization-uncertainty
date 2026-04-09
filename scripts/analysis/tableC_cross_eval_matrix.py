"""Aggregate 4x4 cross-strategy eval results into Tables C1-C6.

Reads metrics.csv files written by evaluate.py via logs.prefix.

Off-diagonal layout:
  experiments/cross_eval/qm9/<train_strategy>/<eval_strategy>/seed_<s>/runs/*/csv_logs/version_0/metrics.csv

Diagonal (native eval) layout:
  experiments/invariance_eval/qm9/<strategy>/seed_<s>/native/runs/*/csv_logs/version_0/metrics.csv

Outputs:
  results/data/cross_eval_qm9_matrix.csv
  results/tables/{nll,ece_overall,ece_node_idx,...}_cross_eval_matrix.{md,tex}
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_utils import write_latex, write_markdown

STRATEGIES = ["random_order", "min_degree_first", "max_degree_first", "anchor_expansion"]
SEEDS = [0, 1, 2]

# All 12 off-diagonal pairs
OFF_DIAGONAL_PAIRS = [
    (train, eval_)
    for train in STRATEGIES
    for eval_ in STRATEGIES
    if train != eval_
]

# Metrics to extract (internal key -> evaluate.py metric name)
EXTRACT_METRICS = {
    "loss_pt":        "test/loss",
    "fcd":            "test/test_fcd",
    "pgd":            "test/test_mol_pgd",
    "validity":       "test/validity",
    "mol_stable":     "test/mol_stable",
    "atm_stable":     "test/atm_stable",
    "uniqueness":     "test/uniqueness",
    "node_idx_ece":   "test/node_idx_ece",
    "node_label_ece": "test/node_label_ece",
    "edge_label_ece": "test/edge_label_ece",
    "special_ece":    "test/special_ece",
}

ECE_COMPONENTS = ["node_idx_ece", "node_label_ece", "edge_label_ece", "special_ece"]

MATRIX_TABLES = [
    ("C1", "loss_pt",        "nll",            "Per-token NLL",                          3),
    ("C2", "ece_overall",    "ece_overall",    "Overall ECE (mean of 4 token-type ECEs)", 4),
    ("C3", "node_idx_ece",   "ece_node_idx",   "Node-index ECE",                         4),
    ("C4", "node_label_ece", "ece_node_label", "Node-label ECE",                         4),
    ("C5", "edge_label_ece", "ece_edge_label", "Edge-label ECE",                         4),
    ("C6", "special_ece",    "ece_special",    "Special-token ECE",                      4),
]


def read_metrics_csv(base_dir: pathlib.Path) -> dict:
    """Read final test metrics from a Lightning CSV log under base_dir."""
    candidates = sorted(base_dir.glob("runs/*/csv_logs/version_0/metrics.csv"))
    if not candidates:
        return {}
    result = {}
    with open(candidates[-1]) as f:
        last_row = None
        for row in csv.DictReader(f):
            last_row = row
    if last_row:
        for key, metric_name in EXTRACT_METRICS.items():
            val = last_row.get(metric_name, "").strip()
            if val:
                result[key] = float(val)
    return result


def add_ece_overall(metrics: dict):
    """Add ece_overall as mean of 4 token-type ECEs (in-place)."""
    vals = [metrics[k] for k in ECE_COMPONENTS if k in metrics]
    if len(vals) == len(ECE_COMPONENTS):
        metrics["ece_overall"] = sum(vals) / len(vals)


def load_all_pairs() -> dict:
    """Load and average all 16 (train, eval) pairs across seeds.

    Returns: {(train_strategy, eval_strategy): averaged_metrics_dict}
    """
    cross_root = pathlib.Path("experiments/cross_eval/qm9")
    inv_root = pathlib.Path("experiments/invariance_eval/qm9")
    results = {}

    # --- Diagonal: native eval from invariance eval dirs ---
    for strategy in STRATEGIES:
        seed_metrics = []
        for seed in SEEDS:
            m = read_metrics_csv(inv_root / strategy / f"seed_{seed}" / "native")
            if m:
                add_ece_overall(m)
                seed_metrics.append(m)
            else:
                print(f"  WARN: Missing native eval for {strategy}/seed_{seed}")
        if seed_metrics:
            avg = {}
            for key in list(EXTRACT_METRICS.keys()) + ["ece_overall"]:
                vals = [m[key] for m in seed_metrics if key in m]
                if vals:
                    avg[key] = sum(vals) / len(vals)
            results[(strategy, strategy)] = avg

    # --- Off-diagonal: cross-eval dirs ---
    for train_s, eval_s in OFF_DIAGONAL_PAIRS:
        seed_metrics = []
        for seed in SEEDS:
            base = cross_root / train_s / eval_s / f"seed_{seed}"
            m = read_metrics_csv(base)
            if m:
                add_ece_overall(m)
                seed_metrics.append(m)
            else:
                print(f"  WARN: Missing cross-eval for {train_s} x {eval_s} / seed_{seed}")
        if seed_metrics:
            avg = {}
            for key in list(EXTRACT_METRICS.keys()) + ["ece_overall"]:
                vals = [m[key] for m in seed_metrics if key in m]
                if vals:
                    avg[key] = sum(vals) / len(vals)
            results[(train_s, eval_s)] = avg
            loss_str = f"{avg.get('loss_pt', float('nan')):.4f}"
            print(f"  {train_s} x {eval_s}: loss_pt={loss_str} ({len(seed_metrics)}/3 seeds)")

    return results


def write_matrix(results, metric_key, file_prefix, title, prec, out_dir):
    SHORT = {
        "random_order":    "random_order",
        "min_degree_first": "min_degree",
        "max_degree_first": "max_degree",
        "anchor_expansion": "anchor_exp",
    }
    headers = ["Train \\ Eval"] + [SHORT[s] for s in STRATEGIES]
    rows = []
    for train_s in STRATEGIES:
        row = [SHORT[train_s]]
        for eval_s in STRATEGIES:
            key = (train_s, eval_s)
            if key in results and metric_key in results[key]:
                cell = f"{results[key][metric_key]:.{prec}f}"
            else:
                cell = "—"
            row.append(cell)
        rows.append(row)

    aligns = ["l"] + ["r"] * len(STRATEGIES)
    write_markdown(out_dir / f"{file_prefix}_cross_eval_matrix.md", headers, rows, alignments=aligns)
    write_latex(
        out_dir / f"{file_prefix}_cross_eval_matrix.tex",
        headers, rows,
        caption=f"{title} (averaged across 3 seeds) for all 16 train $\\times$ eval strategy combinations on QM9. "
                "Diagonal entries use the native linearization strategy; off-diagonal entries use cross-strategy evaluation.",
        label=f"tab:cross_eval_{file_prefix}",
        alignments=aligns,
        starred=False,
    )


def main():
    results = load_all_pairs()

    n_filled = sum(1 for v in results.values() if v)
    print(f"\nLoaded {n_filled}/16 cells.")

    # --- Print NLL matrix to stdout ---
    SHORT = {"random_order": "rand", "min_degree_first": "min",
             "max_degree_first": "max", "anchor_expansion": "anch"}
    print("\n=== 4x4 NLL/tok Matrix (Train down / Eval across) ===")
    print(f"{'':22s}" + "".join(f"{SHORT[s]:>10s}" for s in STRATEGIES))
    for train_s in STRATEGIES:
        row = f"{SHORT[train_s]:22s}"
        for eval_s in STRATEGIES:
            key = (train_s, eval_s)
            if key in results and "loss_pt" in results[key]:
                row += f"{results[key]['loss_pt']:>10.4f}"
            else:
                row += f"{'—':>10s}"
        print(row)

    # --- Write flat CSV ---
    data_dir = pathlib.Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_cols = ["train_strategy", "eval_strategy"] + list(EXTRACT_METRICS.keys()) + ["ece_overall"]
    out_path = data_dir / "tableC_cross_eval_qm9.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_cols)
        for (train_s, eval_s), metrics in sorted(results.items()):
            row = [train_s, eval_s]
            for col in csv_cols[2:]:
                row.append(f"{metrics[col]:.6f}" if col in metrics else "")
            writer.writerow(row)
    print(f"\nSaved {out_path}")

    # --- Write 6 matrix tables ---
    table_dir = pathlib.Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    for table_id, metric_key, file_prefix, title, prec in MATRIX_TABLES:
        print(f"\n--- Table {table_id}: {title} ---")
        write_matrix(results, metric_key, file_prefix, title, prec, table_dir)


if __name__ == "__main__":
    main()
