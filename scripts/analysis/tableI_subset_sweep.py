"""Aggregate subset sweep results (4 strategies × 3 sizes × 3 seeds = 36 runs).

Reads test metrics from eval CSV logs:
  logs/test/QM9/{strategy}/llama-s/{seed}/N_{size}/runs/{timestamp}/csv_logs/version_0/metrics.csv

Outputs:
  results/data/subset_sweep_results_qm9.csv
  results/tables/subset_sweep_qm9.md
  results/tables/subset_sweep_qm9.tex
"""

import csv
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_utils import fmt_mean_std, write_latex, write_markdown

STRATEGIES = ["random_order", "min_degree_first", "max_degree_first", "anchor_expansion"]
SEEDS = [0, 1, 2]
SIZES = [128, 1000, 10000]

METRICS = [
    "test/loss",
    "test/test_fcd",
    "test/test_mol_pgd",
    "test/validity",
    "test/mol_stable",
    "test/uniqueness",
    "test/novelty",
]

PRETTY_NAMES = {
    "random_order": "Random",
    "min_degree_first": "Min-Degree",
    "max_degree_first": "Max-Degree",
    "anchor_expansion": "Anchor",
}


def find_test_metrics(seed: int, strategy: str, size: int) -> dict:
    """Find test metrics from logs/test/QM9/{strategy}/llama-s/{seed}/N_{size}/runs/{timestamp}/csv_logs/version_0/metrics.csv"""
    base = pathlib.Path(f"logs/test/QM9/{strategy}/llama-s/{seed}/N_{size}/runs")
    if not base.exists():
        return {}

    candidates = sorted(base.glob("*/csv_logs/version_0/metrics.csv"), reverse=True)
    if not candidates:
        return {}

    return read_final_metrics(candidates[0])


def read_final_metrics(csv_path: pathlib.Path) -> dict:
    """Read the last row of a metrics CSV (final aggregated test metrics)."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        return {}

    last_row = rows[-1]
    result = {}
    for metric in METRICS:
        val = last_row.get(metric, "").strip()
        if val:
            try:
                result[metric] = float(val)
            except ValueError:
                pass
    return result


def main():
    data_dir = pathlib.Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir = pathlib.Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by strategy × size
    results = []

    for strategy in STRATEGIES:
        for size in SIZES:
            seed_metrics = {m: [] for m in METRICS}
            seed_info = []  # Track which seeds we found

            for seed in SEEDS:
                # Find test metrics from seed-specific eval directory
                metrics = find_test_metrics(seed, strategy, size)
                if not metrics:
                    print(f"  MISS: {strategy} N={size} seed={seed}")
                    continue

                for m in METRICS:
                    if m in metrics:
                        seed_metrics[m].append(metrics[m])

                seed_info.append(seed)
                print(f"  FOUND: {strategy} N={size} seed={seed}")

            # Aggregate across seeds
            agg = {}
            for m in METRICS:
                vals = seed_metrics[m]
                if len(vals) >= 2:
                    agg[m] = (statistics.mean(vals), statistics.stdev(vals))
                elif len(vals) == 1:
                    agg[m] = (vals[0], 0.0)
                else:
                    agg[m] = (None, None)

            results.append({
                "strategy": strategy,
                "subset_size": size,
                "metrics": agg,
                "seed_count": len(seed_info),
            })

    # Write CSV
    csv_path = data_dir / "tableI_subset_sweep_qm9.csv"
    csv_cols = ["strategy", "subset_size"] + METRICS
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for r in results:
            row = [r["strategy"], r["subset_size"]]
            for m in METRICS:
                mean, std = r["metrics"][m]
                if mean is not None:
                    row.append(f"{mean:.4f}")
                else:
                    row.append("")
            w.writerow(row)
    print(f"\nSaved: {csv_path}")

    # Write Markdown + LaTeX
    headers = [
        "Strategy", "$N_{\\text{train}}$", "NLL/tok", "Validity",
        "Mol. Stab.", "Uniqueness", "FCD", "PGD",
    ]

    table_rows = []
    for r in results:
        row = [
            PRETTY_NAMES[r["strategy"]],
            r["subset_size"],
        ]

        # NLL/tok
        mean, std = r["metrics"]["test/loss"]
        row.append(fmt_mean_std(mean, std) if mean is not None else "—")

        # Validity (as percentage)
        mean, std = r["metrics"]["test/validity"]
        row.append(fmt_mean_std(mean * 100, std * 100, prec=1) if mean is not None else "—")

        # Mol. Stab. (as percentage)
        mean, std = r["metrics"]["test/mol_stable"]
        row.append(fmt_mean_std(mean * 100, std * 100, prec=1) if mean is not None else "—")

        # Uniqueness (as percentage)
        mean, std = r["metrics"]["test/uniqueness"]
        row.append(fmt_mean_std(mean * 100, std * 100, prec=1) if mean is not None else "—")

        # FCD
        mean, std = r["metrics"]["test/test_fcd"]
        row.append(fmt_mean_std(mean, std) if mean is not None else "—")

        # PGD
        mean, std = r["metrics"]["test/test_mol_pgd"]
        row.append(fmt_mean_std(mean, std) if mean is not None else "—")

        table_rows.append(row)

    write_markdown(table_dir / "tableI_subset_sweep_qm9.md", headers, table_rows)
    write_latex(
        table_dir / "tableI_subset_sweep_qm9.tex",
        headers, table_rows,
        caption="QM9 subset-size sweep (3-seed mean ± std). "
                "Generation quality at $N \\in \\{128, 1000, 10000\\}$ training graphs "
                "for all four linearization strategies.",
        label="tab:subset-sweep",
        starred=True,
    )
    print(f"Saved: {table_dir / 'subset_sweep_qm9.md'}")
    print(f"Saved: {table_dir / 'subset_sweep_qm9.tex'}")

    # Pretty-print
    print("\n=== Subset Sweep Results ===")
    col_w = [max(len(str(headers[i])), *(len(str(row[i])) for row in table_rows))
             for i in range(len(headers))]
    hdr = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(hdr)
    print("-" * len(hdr))
    for row in table_rows:
        print("  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))


if __name__ == "__main__":
    main()
