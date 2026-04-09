"""Aggregate invariance eval data into Table 2 for the paper.

Reads:
  - experiments/invariance_eval/qm9/{strategy}/seed_{s}/invariance_*.csv
    Contains rows for both native eval (Test_Strategy == Train_Strategy)
    and randomized eval (Test_Strategy == "random_order").
    Mean_NLL is total-sequence NLL (summed across tokens).

  - experiments/invariance_eval/qm9/{strategy}/seed_{s}/native/...metrics.csv
  - experiments/invariance_eval/qm9/{strategy}/seed_{s}/random/...metrics.csv
    Full test-set eval with per-token loss and per-token-type ECE.

Outputs:
  - results/data/table2_permutation_consistency_qm9.csv
  - results/tables/table2_permutation_consistency_qm9.md   (paper-ready markdown)
  - results/tables/table2_permutation_consistency_qm9.tex  (paper-ready LaTeX fragment)
  - Pretty-printed tables to stdout
"""

import csv
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_utils import fmt_mean_std, write_latex_multiheader, write_markdown

ROOT = pathlib.Path("experiments/invariance_eval/qm9")
STRATEGIES = ["random_order", "min_degree_first", "max_degree_first", "anchor_expansion"]
SEEDS = [0, 1, 2]


def read_last_row(csv_path: str) -> dict:
    """Read the last non-empty row of a CSV as a dict."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows[-1]


def load_seq_lengths() -> dict:
    """Load tokens/graph from seq_lengths_qm9.csv → {strategy: mean_seq_len}."""
    path = pathlib.Path("results/data/seq_lengths_qm9.csv")
    out = {}
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                out[row["strategy"]] = float(row["mean_seq_len"])
    return out


def find_metrics_csv(base_dir: pathlib.Path) -> str:
    """Find the metrics.csv under a run directory."""
    candidates = sorted(base_dir.glob("runs/*/csv_logs/version_0/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv under {base_dir}")
    return str(candidates[-1])


ECE_TOKEN_TYPES = [
    "ECE_node_index", "ECE_node_label", "ECE_edge_label",
    "ECE_special", "ECE_new_node", "ECE_revisit",
]


def parse_invariance_csv(path: pathlib.Path, strategy: str):
    """Split invariance CSV into native and randomized rows.

    Returns per-graph averages for NLL/CV/ECE, split by eval condition.
    ECE_overall is read directly from the CSV (one value per graph).
    """
    native_nlls, native_cvs = [], []
    random_nlls, random_cvs = [], []
    native_ece_overall, random_ece_overall = [], []
    native_ece_by_type = {t: [] for t in ECE_TOKEN_TYPES}
    random_ece_by_type = {t: [] for t in ECE_TOKEN_TYPES}
    seq_lens = []

    with open(path) as f:
        reader = csv.DictReader(f)
        has_ece = "ECE_overall" in reader.fieldnames
        has_seq_len = "Mean_Seq_Len" in reader.fieldnames
        for row in reader:
            nll = float(row["Mean_NLL"])
            cv = float(row["CV"])
            if has_seq_len:
                seq_lens.append(float(row["Mean_Seq_Len"]))
            if row["Test_Strategy"] == strategy:
                native_nlls.append(nll)
                native_cvs.append(cv)
                if has_ece:
                    native_ece_overall.append(float(row["ECE_overall"]))
                    for t in ECE_TOKEN_TYPES:
                        if t in row:
                            native_ece_by_type[t].append(float(row[t]))
            if row["Test_Strategy"] == "random_order":
                random_nlls.append(nll)
                random_cvs.append(cv)
                if has_ece:
                    random_ece_overall.append(float(row["ECE_overall"]))
                    for t in ECE_TOKEN_TYPES:
                        if t in row:
                            random_ece_by_type[t].append(float(row[t]))

    result = {
        "native_nll": statistics.mean(native_nlls) if native_nlls else None,
        "native_cv": statistics.mean(native_cvs) if native_cvs else None,
        "randomized_nll": statistics.mean(random_nlls),
        "randomized_cv": statistics.mean(random_cvs),
        "native_ece_overall": statistics.mean(native_ece_overall) if native_ece_overall else None,
        "random_ece_overall": statistics.mean(random_ece_overall) if random_ece_overall else None,
        "n_graphs": len(random_nlls),
        "mean_seq_len": statistics.mean(seq_lens) if seq_lens else None,
    }
    # Per-token-type ECE
    for t in ECE_TOKEN_TYPES:
        short = t.replace("ECE_", "")
        result[f"native_ece_{short}"] = (
            statistics.mean(native_ece_by_type[t]) if native_ece_by_type[t] else None
        )
        result[f"random_ece_{short}"] = (
            statistics.mean(random_ece_by_type[t]) if random_ece_by_type[t] else None
        )
    return result


def parse_population_csvs(native_path, random_path, strategy):
    """Parse population-scale eval CSVs (one per eval strategy).

    Each population_*.csv has the same schema as invariance_*.csv but contains
    only one Test_Strategy per file.  We read native + random separately.
    """
    result = {
        "native_nll": None, "native_cv": None,
        "randomized_nll": None, "randomized_cv": None,
        "native_ece_overall": None, "random_ece_overall": None,
        "n_graphs": 0, "mean_seq_len": None,
    }
    for t in ECE_TOKEN_TYPES:
        short = t.replace("ECE_", "")
        result[f"native_ece_{short}"] = None
        result[f"random_ece_{short}"] = None

    for path, _prefix, is_native in [
        (native_path, "native", True),
        (random_path, "random", False),
    ]:
        if not path.exists():
            continue
        nlls, cvs, ece_overall_vals = [], [], []
        ece_by_type = {t: [] for t in ECE_TOKEN_TYPES}
        seq_lens = []

        with open(path) as f:
            reader = csv.DictReader(f)
            has_ece = "ECE_overall" in (reader.fieldnames or [])
            has_seq_len = "Mean_Seq_Len" in (reader.fieldnames or [])
            for row in reader:
                nlls.append(float(row["Mean_NLL"]))
                cvs.append(float(row["CV"]))
                if has_seq_len:
                    seq_lens.append(float(row["Mean_Seq_Len"]))
                if has_ece:
                    ece_overall_vals.append(float(row["ECE_overall"]))
                    for t in ECE_TOKEN_TYPES:
                        if t in row:
                            ece_by_type[t].append(float(row[t]))

        if is_native:
            result["native_nll"] = statistics.mean(nlls) if nlls else None
            result["native_cv"] = statistics.mean(cvs) if cvs else None
            result["native_ece_overall"] = statistics.mean(ece_overall_vals) if ece_overall_vals else None
            result["n_graphs"] = len(nlls)
            result["mean_seq_len"] = statistics.mean(seq_lens) if seq_lens else None
            for t in ECE_TOKEN_TYPES:
                short = t.replace("ECE_", "")
                result[f"native_ece_{short}"] = (
                    statistics.mean(ece_by_type[t]) if ece_by_type[t] else None
                )
        else:
            result["randomized_nll"] = statistics.mean(nlls) if nlls else None
            result["randomized_cv"] = statistics.mean(cvs) if cvs else None
            result["random_ece_overall"] = statistics.mean(ece_overall_vals) if ece_overall_vals else None
            if not result["n_graphs"]:
                result["n_graphs"] = len(nlls)
            if result["mean_seq_len"] is None and seq_lens:
                result["mean_seq_len"] = statistics.mean(seq_lens)
            for t in ECE_TOKEN_TYPES:
                short = t.replace("ECE_", "")
                result[f"random_ece_{short}"] = (
                    statistics.mean(ece_by_type[t]) if ece_by_type[t] else None
                )

    return result


def aggregate():
    results = []

    for strategy in STRATEGIES:
        seed_data = []
        for seed in SEEDS:
            seed_dir = ROOT / strategy / f"seed_{seed}"

            # --- Permutation audit (K=32 full test set) ---
            k32_dir = seed_dir / "k32"
            pop_native = k32_dir / f"population_{strategy}.csv"
            pop_random = k32_dir / "population_random_order.csv"

            if pop_native.exists() or pop_random.exists():
                perm = parse_population_csvs(pop_native, pop_random, strategy)
            else:
                print(f"  WARN: No K=32 audit CSV for {strategy}/seed_{seed}")
                continue

            native_nll = perm["native_nll"]
            randomized_nll = perm["randomized_nll"]

            # --- Full test-set eval: Native ---
            native_dir = seed_dir / "native"
            native_csv = find_metrics_csv(native_dir)
            native_row = read_last_row(native_csv)

            # --- Full test-set eval: Random ---
            random_dir = seed_dir / "random"
            random_csv = find_metrics_csv(random_dir)
            random_row = read_last_row(random_csv)

            sd = {
                "seed": seed,
                "native_nll": native_nll,
                "randomized_nll": randomized_nll,
                "cv": perm["randomized_cv"],
                "mean_seq_len": perm["mean_seq_len"],
                "native_cv": perm["native_cv"],
                "n_graphs": perm["n_graphs"],
                # Overall ECE from invariance eval
                "native_ece_overall": perm["native_ece_overall"],
                "random_ece_overall": perm["random_ece_overall"],
                # Per-token loss (full test set)
                "native_loss_pt": float(native_row["test/loss"]),
                "random_loss_pt": float(random_row["test/loss"]),
                # ECE from full test-set eval (per token type)
                "native_ece_node_idx": float(native_row["test/node_idx_ece"]),
                "native_ece_node_label": float(native_row["test/node_label_ece"]),
                "native_ece_edge": float(native_row["test/edge_label_ece"]),
                "native_ece_special": float(native_row["test/special_ece"]),
                "random_ece_node_idx": float(random_row["test/node_idx_ece"]),
                "random_ece_node_label": float(random_row["test/node_label_ece"]),
                "random_ece_edge": float(random_row["test/edge_label_ece"]),
                "random_ece_special": float(random_row["test/special_ece"]),
                # Validity & stability
                "native_validity": float(native_row["test/validity"]),
                "random_validity": float(random_row["test/validity"]),
                "native_mol_stable": float(native_row["test/mol_stable"]),
                "random_mol_stable": float(random_row["test/mol_stable"]),
                # Generation quality
                "native_fcd": float(native_row["test/test_fcd"]),
                "random_fcd": float(random_row["test/test_fcd"]),
            }
            # Per-token-type ECE from invariance eval
            for t in ECE_TOKEN_TYPES:
                short = t.replace("ECE_", "")
                sd[f"inv_native_ece_{short}"] = perm.get(f"native_ece_{short}")
                sd[f"inv_random_ece_{short}"] = perm.get(f"random_ece_{short}")
            seed_data.append(sd)

        if not seed_data:
            continue

        def mean_std(key, seed_data=seed_data):
            vals = [d[key] for d in seed_data if d[key] is not None]
            if not vals:
                return None, None
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return m, s

        row = {"strategy": strategy}
        for key in seed_data[0]:
            if key in ("seed", "n_graphs"):
                continue
            m, s = mean_std(key)
            row[f"{key}_mean"] = m
            row[f"{key}_std"] = s
        row["n_graphs"] = seed_data[0]["n_graphs"]

        results.append(row)

    # --- Patch mean_seq_len from fallback source ---
    seq_lens = load_seq_lengths()
    for r in results:
        if r.get("mean_seq_len_mean") is None and r["strategy"] in seq_lens:
            r["mean_seq_len_mean"] = seq_lens[r["strategy"]]
            r["mean_seq_len_std"] = 0.0

    # --- Output CSV ---
    out_path = pathlib.Path("results/data/table2_permutation_consistency_qm9.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_path}")

    # --- Paper-ready markdown + LaTeX (compact subset of columns) ---
    STRATEGY_LABELS = {
        "random_order": "random\\_order",
        "min_degree_first": "min\\_degree",
        "max_degree_first": "max\\_degree",
        "anchor_expansion": "anchor\\_exp",
    }
    STRATEGY_LABELS = {
        "random_order": "Random",
        "min_degree_first": "Min-Degree",
        "max_degree_first": "Max-Degree",
        "anchor_expansion": "Anchor",
    }
    t1_headers = [
        "Strategy", "Tok/graph",
        "NLL/tok (N) ↓", "NLL/tok (R) ↓",
        "Topo. Unc. (N) ↓", "Topo. Unc. (R) ↓",
        "ECE (N) ↓", "ECE (R) ↓",
    ]
    t1_col_groups = [
        (None,                        ["Strategy"]),
        (None,                        ["Tok/graph"]),
        ("NLL/token ↓",               ["Native", "Random"]),
        ("Linearization Uncertainty ↓", ["Native", "Random"]),
        ("ECE ↓",                     ["Native", "Random"]),
    ]
    t1_rows = []
    for r in results:
        _tok_mean = r.get("mean_seq_len_mean")
        tok = f"${float(_tok_mean):.1f}$" if _tok_mean is not None else "—"
        ece_n = fmt_mean_std(r.get("native_ece_overall_mean"), r.get("native_ece_overall_std"), prec=3)
        ece_r = fmt_mean_std(r.get("random_ece_overall_mean"), r.get("random_ece_overall_std"), prec=3)
        topo_n = fmt_mean_std(r.get("native_cv_mean"), r.get("native_cv_std"), prec=3)
        topo_r = fmt_mean_std(r.get("cv_mean"), r.get("cv_std"), prec=3)
        t1_rows.append([
            STRATEGY_LABELS.get(r["strategy"], r["strategy"]),
            tok,
            fmt_mean_std(r["native_loss_pt_mean"], r["native_loss_pt_std"]),
            fmt_mean_std(r["random_loss_pt_mean"], r["random_loss_pt_std"]),
            topo_n,
            topo_r,
            ece_n,
            ece_r,
        ])
    table_dir = pathlib.Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(table_dir / "table2_permutation_consistency_qm9.md", t1_headers, t1_rows)
    write_latex_multiheader(
        table_dir / "table2_permutation_consistency_qm9.tex", t1_col_groups, t1_rows,
        caption="Test-set robustness on QM9 ($K{=}32$ permutations, full test set, 3 seeds). "
                "Native = model's own training strategy; Random = evaluated under the Random Order strategy.",
        label="tab:robustness",
        starred=True,
    )

    # --- Pretty print: Main table ---
    print("\n" + "=" * 140)
    print("TABLE 1: Invariance Eval (QM9, K=32 permutations, 3 seeds)")
    print("Total-sequence NLL from permutation eval subset; ECE from full test-set eval")
    print("=" * 140)

    def fmt(r, key, prec=2):
        m = r[f"{key}_mean"]
        s = r[f"{key}_std"]
        return f"{m:.{prec}f} ± {s:.{prec}f}"

    header = (
        f"{'Strategy':<22} | {'Tok/graph':>10} | {'Native NLL':>16} | {'Random. NLL':>16} | "
        f"{'CV':>14} | "
        f"{'ECE Nat.':>10} | {'ECE Rand.':>10} | "
        f"{'Val Nat.':>10} | {'Val Rand.':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        tok_str = fmt(r, 'mean_seq_len', 1) if r.get('mean_seq_len_mean') is not None else "—"
        print(
            f"{r['strategy']:<22} | "
            f"{tok_str:>10} | "
            f"{fmt(r, 'native_nll'):>16} | "
            f"{fmt(r, 'randomized_nll'):>16} | "
            f"{fmt(r, 'cv', 3):>14} | "
            f"{fmt(r, 'native_ece_overall', 3):>10} | "
            f"{fmt(r, 'random_ece_overall', 3):>10} | "
            f"{r['native_validity_mean']:>10.3f} | "
            f"{r['random_validity_mean']:>10.3f}"
        )

    # --- ECE breakdown (full test-set eval) ---
    print("\n" + "=" * 100)
    print("ECE BREAKDOWN BY TOKEN TYPE (mean across 3 seeds, full test-set eval)")
    print("=" * 100)
    header2 = (
        f"{'Strategy':<22} | {'Eval':>8} | {'Node Idx':>10} | {'Node Lbl':>10} | "
        f"{'Edge Lbl':>10} | {'Special':>10}"
    )
    print(header2)
    print("-" * len(header2))
    for r in results:
        for eval_type in ["native", "random"]:
            print(
                f"{r['strategy']:<22} | {eval_type:>8} | "
                f"{r[f'{eval_type}_ece_node_idx_mean']:>10.4f} | "
                f"{r[f'{eval_type}_ece_node_label_mean']:>10.4f} | "
                f"{r[f'{eval_type}_ece_edge_mean']:>10.4f} | "
                f"{r[f'{eval_type}_ece_special_mean']:>10.4f}"
            )

    # --- ECE breakdown (invariance eval, includes new_node/revisit) ---
    token_shorts = ["node_index", "node_label", "edge_label", "special", "new_node", "revisit"]
    print("\n" + "=" * 130)
    print("ECE BREAKDOWN BY TOKEN TYPE (mean across 3 seeds, invariance eval — for Appendix C)")
    print("=" * 130)
    header3 = (
        f"{'Strategy':<22} | {'Eval':>8} | {'Node Idx':>10} | {'Node Lbl':>10} | "
        f"{'Edge Lbl':>10} | {'Special':>10} | {'New Node':>10} | {'Revisit':>10}"
    )
    print(header3)
    print("-" * len(header3))
    for r in results:
        for eval_type in ["native", "random"]:
            prefix = f"inv_{eval_type}_ece_"
            vals = []
            for t in token_shorts:
                key = f"{prefix}{t}_mean"
                v = r.get(key)
                vals.append(f"{v:>10.4f}" if v is not None else f"{'—':>10}")
            print(
                f"{r['strategy']:<22} | {eval_type:>8} | " + " | ".join(vals)
            )

    # --- Per-token loss comparison ---
    print("\n" + "=" * 80)
    print("PER-TOKEN LOSS (full test set, for reference)")
    print("=" * 80)
    for r in results:
        print(
            f"{r['strategy']:<22} | "
            f"Native: {fmt(r, 'native_loss_pt', 4)} | "
            f"Random: {fmt(r, 'random_loss_pt', 4)}"
        )


if __name__ == "__main__":
    aggregate()
