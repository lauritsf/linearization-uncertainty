"""Build all appendix figures and tables from existing experiment data.

Produces:
  - Figure A1 check: Verify existing diversity plot
  - Table/Figure C1: ECE by token type (bar chart)
  - Table D1: Valid vs. Invalid NLL/CV comparison
  - Figure D1: NLL vs. CV scatter colored by validity
  - Figure D3: Rejection curves (validity & mol_stable vs. rejection threshold)
  - Table D2: Sequence length stats (valid vs. invalid)
  - Figure G1: NLL vs. CV scatter with correlation coefficient

Outputs go to plots/ (figures) and experiments/appendix/ (tables as CSVs).
"""

import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_utils import write_latex, write_markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = ROOT / "docs" / "manuscript" / "figures"
APPENDIX_DIR = ROOT / "docs" / "manuscript" / "tables"
DATA_DIR = ROOT / "docs" / "manuscript" / "data"
TABLE1_PATH = DATA_DIR / "permutation_consistency_qm9.csv"
GEN_INV_DIR = ROOT / "logs" / "analysis" / "generated_invariance"

STRATEGIES = ["random_order", "min_degree_first", "max_degree_first", "anchor_expansion"]
STRATEGY_LABELS = {
    "random_order": "Random",
    "min_degree_first": "Min-deg",
    "max_degree_first": "Max-deg",
    "anchor_expansion": "Anchor",
}
STRATEGY_COLORS = {
    "random_order": "#C44E52",
    "min_degree_first": "#55A868",
    "max_degree_first": "#DD8452",
    "anchor_expansion": "#4C72B0",
}


def load_table1():
    """Load the aggregated permutation_consistency_qm9.csv."""
    rows = {}
    with open(TABLE1_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["strategy"]] = row
    return rows


def load_generated_invariance():
    """Load all generated invariance CSVs into a dict of lists."""
    data = {}
    for strat in STRATEGIES:
        path = GEN_INV_DIR / f"generated_invariance_{strat}.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "strategy": strat,
                    "smiles": row["SMILES"],
                    "is_valid": row["Is_Valid"] == "True",
                    "is_stable": row["Is_Stable"] == "True",
                    "nll_mean": float(row["NLL_Mean"]),
                    "nll_std": float(row["NLL_Std"]),
                    "cv": float(row["CV"]),
                })
        data[strat] = rows
    return data


# ---------------------------------------------------------------------------
# 1. Figure A1 Check
# ---------------------------------------------------------------------------
def check_figure_a1():
    """Verify Figure A1 diversity plot exists."""
    print("\n=== Figure A1: Diversity Saturation ===")
    pdf_path = PLOTS_DIR / "diversity_all_datasets.pdf"
    png_path = PLOTS_DIR / "diversity_all_datasets.png"
    if pdf_path.exists() and png_path.exists():
        print(f"  OK: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")
        print(f"  OK: {png_path} ({png_path.stat().st_size / 1024:.0f} KB)")
    else:
        print("  MISSING: No diversity_all_datasets plot found!")


# ---------------------------------------------------------------------------
# 2. Table/Figure C1: ECE by Token Type
# ---------------------------------------------------------------------------
def build_ece_by_token_type(table1):
    """Bar chart: ECE by token type × strategy × eval mode.

    Uses 6-type inv-ECE columns from the invariance eval (inv_native_ece_* /
    inv_random_ece_*) rather than the 4-type columns from the full test-set eval.
    """
    print("\n=== Figure C1: ECE by Token Type ===")

    # 6 token types from invariance eval (matches ECE_TOKEN_TYPES in table2_permutation_consistency.py)
    token_types = ["node_index", "node_label", "edge_label", "special", "new_node", "revisit"]
    token_labels = ["Node Index", "Node Label", "Edge Label", "Special", "New Node", "Revisit"]
    eval_modes = ["native", "random"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax_idx, mode in enumerate(eval_modes):
        ax = axes[ax_idx]
        x = np.arange(len(token_types))
        width = 0.18
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

        for i, strat in enumerate(STRATEGIES):
            row = table1[strat]
            vals = []
            errs = []
            for tt in token_types:
                key_mean = f"inv_{mode}_ece_{tt}_mean"
                key_std = f"inv_{mode}_ece_{tt}_std"
                vals.append(float(row[key_mean]))
                errs.append(float(row[key_std]))
            ax.bar(
                x + offsets[i], vals, width, yerr=errs,
                label=STRATEGY_LABELS[strat], color=STRATEGY_COLORS[strat],
                capsize=3, edgecolor="white", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(token_labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("ECE", fontsize=11)
        ax.set_title(f"{'Native' if mode == 'native' else 'Randomized'} Eval", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("ECE by Token Type and Evaluation Mode", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"ece_by_token_type.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ece_by_token_type.{pdf,png}")

    # Also save as CSV / markdown / LaTeX
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "ece_by_token_type.csv"
    t_headers = ["Strategy", "Eval Mode"] + token_labels
    t_rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "Eval_Mode"] + token_labels)
        for strat in STRATEGIES:
            row = table1[strat]
            for mode in eval_modes:
                vals = [float(row[f"inv_{mode}_ece_{tt}_mean"]) for tt in token_types]
                writer.writerow([STRATEGY_LABELS[strat], mode] + [f"{v:.4f}" for v in vals])
                t_rows.append([STRATEGY_LABELS[strat], mode] + [f"{v:.4f}" for v in vals])
    print(f"  Saved: {csv_path}")
    write_markdown(APPENDIX_DIR / "ece_by_token_type.md", t_headers, t_rows)
    write_latex(
        APPENDIX_DIR / "ece_by_token_type.tex",
        t_headers, t_rows,
        caption="ECE by token type from the permutation invariance eval ($K{=}32$ permutations, "
                "3 seeds). Native = model's own training strategy; Random. = \\texttt{random\\_order}.",
        label="tab:ece_by_type",
        alignments=["l", "l"] + ["r"] * len(token_labels),
    )


# ---------------------------------------------------------------------------
# 3. Table D1: Valid vs. Invalid NLL/CV
# ---------------------------------------------------------------------------
def build_validity_nll_table(gen_data):
    """Table comparing NLL and CV for valid vs. invalid generated molecules."""
    print("\n=== Table D1: Valid vs. Invalid NLL/CV ===")

    csv_path = APPENDIX_DIR / "validity_nll_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Strategy", "Group", "Count", "NLL_Mean", "NLL_Std",
            "CV_Mean", "CV_Std",
        ])

        for strat in STRATEGIES:
            if strat not in gen_data:
                continue
            rows = gen_data[strat]
            for group_name, filter_fn in [
                ("Valid", lambda r: r["is_valid"]),
                ("Invalid", lambda r: not r["is_valid"]),
                ("Stable", lambda r: r["is_stable"]),
                ("Unstable", lambda r: not r["is_stable"]),
            ]:
                subset = [r for r in rows if filter_fn(r)]
                if not subset:
                    continue
                nlls = [r["nll_mean"] for r in subset]
                cvs = [r["cv"] for r in subset]
                writer.writerow([
                    STRATEGY_LABELS[strat], group_name, len(subset),
                    f"{np.mean(nlls):.2f}", f"{np.std(nlls):.2f}",
                    f"{np.mean(cvs):.4f}", f"{np.std(cvs):.4f}",
                ])

    print(f"  Saved: {csv_path}")

    # Print summary for verification
    if "random_order" in gen_data:
        rows = gen_data["random_order"]
        stable = [r for r in rows if r["is_valid"] and r["is_stable"]]
        unstable = [r for r in rows if r["is_valid"] and not r["is_stable"]]
        invalid = [r for r in rows if not r["is_valid"]]
        if stable and unstable:
            s_nll = np.mean([r["nll_mean"] for r in stable])
            u_nll = np.mean([r["nll_mean"] for r in unstable])
            s_cv = np.mean([r["cv"] for r in stable])
            u_cv = np.mean([r["cv"] for r in unstable])
            print(f"  Random Order: Stable NLL={s_nll:.2f}, Unstable NLL={u_nll:.2f}")
            print(f"  Random Order: Stable CV={s_cv:.4f}, Unstable CV={u_cv:.4f}")
            print(f"  CV ratio (unstable/stable): {u_cv / s_cv:.2f}x")
        if invalid:
            i_nll = np.mean([r["nll_mean"] for r in invalid])
            i_cv = np.mean([r["cv"] for r in invalid])
            print(f"  Random Order: Invalid (empty SMILES) n={len(invalid)}, "
                  f"NLL={i_nll:.2f}, CV={i_cv:.4f}")


# ---------------------------------------------------------------------------
# 4. Figure D1: NLL vs. CV Scatter
# ---------------------------------------------------------------------------
def build_nll_cv_scatter(gen_data):
    """Scatter plot: NLL_Mean vs. CV, colored by validity."""
    print("\n=== Figure D1: NLL vs. CV Scatter ===")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, strat in enumerate(STRATEGIES):
        ax = axes[idx]
        if strat not in gen_data:
            ax.set_title(f"{STRATEGY_LABELS[strat]} (no data)")
            continue
        rows = gen_data[strat]
        valid = [r for r in rows if r["is_valid"]]
        invalid = [r for r in rows if not r["is_valid"]]

        if invalid:
            ax.scatter(
                [r["nll_mean"] for r in invalid],
                [r["cv"] for r in invalid],
                c="#d62728", alpha=0.3, s=8, label=f"Invalid ({len(invalid)})",
                rasterized=True,
            )
        if valid:
            ax.scatter(
                [r["nll_mean"] for r in valid],
                [r["cv"] for r in valid],
                c="#2ca02c", alpha=0.3, s=8, label=f"Valid ({len(valid)})",
                rasterized=True,
            )

        ax.set_xlabel("NLL Mean", fontsize=10)
        ax.set_ylabel("CV", fontsize=10)
        ax.set_title(STRATEGY_LABELS[strat], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, markerscale=3)

    fig.suptitle("NLL vs. CV by Validity Status", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"nll_cv_scatter_validity.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: nll_cv_scatter_validity.{pdf,png}")


# ---------------------------------------------------------------------------
# 5. Figure D3: Rejection Curves
# ---------------------------------------------------------------------------
def build_rejection_curves(gen_data):
    """Line plot: rejection threshold vs. validity/stability after rejection."""
    print("\n=== Figure D3: Rejection Curves ===")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, strat in enumerate(["random_order", "anchor_expansion"]):
        ax = axes[idx]
        if strat not in gen_data:
            ax.set_title(f"{STRATEGY_LABELS[strat]} (no data)")
            continue
        rows = gen_data[strat]
        # Sort by CV descending (highest CV = most uncertain → reject first)
        sorted_rows = sorted(rows, key=lambda r: r["cv"], reverse=True)
        n = len(sorted_rows)

        reject_fracs = np.linspace(0, 0.95, 100)
        validity_rates = []
        stability_rates = []

        for frac in reject_fracs:
            n_reject = int(frac * n)
            remaining = sorted_rows[n_reject:]
            if not remaining:
                validity_rates.append(np.nan)
                stability_rates.append(np.nan)
                continue
            validity_rates.append(sum(r["is_valid"] for r in remaining) / len(remaining))
            stability_rates.append(sum(r["is_stable"] for r in remaining) / len(remaining))

        ax.plot(reject_fracs * 100, np.array(validity_rates) * 100,
                label="Validity", color="#2ca02c", linewidth=2)
        ax.plot(reject_fracs * 100, np.array(stability_rates) * 100,
                label="Mol Stable", color="#1f77b4", linewidth=2)
        ax.set_xlabel("Rejection Fraction (%)", fontsize=11)
        ax.set_ylabel("Rate (%)", fontsize=11)
        ax.set_title(STRATEGY_LABELS[strat], fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(0, 95)
        ax.grid(alpha=0.3)

        # Print key thresholds
        base_valid = sum(r["is_valid"] for r in rows) / n
        base_stable = sum(r["is_stable"] for r in rows) / n
        print(f"  {STRATEGY_LABELS[strat]}: base validity={base_valid:.3f}, "
              f"base stability={base_stable:.3f}")

    fig.suptitle("Rejection Curves: CV-Based Filtering", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"rejection_curves.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: rejection_curves.{pdf,png}")

    # Also save full rejection curve data for random_order
    if "random_order" in gen_data:
        rows = gen_data["random_order"]
        sorted_rows = sorted(rows, key=lambda r: r["cv"], reverse=True)
        n = len(sorted_rows)
        csv_path = APPENDIX_DIR / "rejection_curves.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Reject_Pct", "Remaining", "Validity_Pct", "MolStable_Pct"])
            for pct in range(0, 96, 5):
                n_reject = int(pct / 100.0 * n)
                remaining = sorted_rows[n_reject:]
                val_rate = sum(r["is_valid"] for r in remaining) / len(remaining) * 100
                stab_rate = sum(r["is_stable"] for r in remaining) / len(remaining) * 100
                writer.writerow([pct, len(remaining), f"{val_rate:.1f}", f"{stab_rate:.1f}"])
        print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# 6. Table D2: Sequence Length Analysis
# ---------------------------------------------------------------------------
def build_sequence_length_analysis(gen_data):
    """Compare SMILES token lengths for valid vs. invalid molecules."""
    print("\n=== Table D2: Sequence Length Analysis ===")

    def smiles_token_length(smiles):
        """Approximate SMILES token count (bracket atoms, two-char elements, etc.)."""
        # Simple heuristic: count non-bracket characters + bracket groups
        length = 0
        i = 0
        while i < len(smiles):
            if smiles[i] == '[':
                # Find closing bracket
                j = smiles.find(']', i)
                if j == -1:
                    j = len(smiles)
                length += 1
                i = j + 1
            else:
                length += 1
                i += 1
        return length

    csv_path = APPENDIX_DIR / "seq_len_valid_vs_invalid.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Strategy", "Group", "Count", "SMILES_Len_Mean", "SMILES_Len_Std",
            "SMILES_Len_Median",
        ])

        for strat in STRATEGIES:
            if strat not in gen_data:
                continue
            rows = gen_data[strat]
            for group_name, filter_fn in [
                ("All", lambda r: True),
                ("Valid+Stable", lambda r: r["is_valid"] and r["is_stable"]),
                ("Valid+Unstable", lambda r: r["is_valid"] and not r["is_stable"]),
                ("Invalid", lambda r: not r["is_valid"]),
            ]:
                subset = [r for r in rows if filter_fn(r)]
                if not subset:
                    continue
                lengths = [smiles_token_length(r["smiles"]) for r in subset]
                writer.writerow([
                    STRATEGY_LABELS[strat], group_name, len(subset),
                    f"{np.mean(lengths):.1f}", f"{np.std(lengths):.1f}",
                    f"{np.median(lengths):.0f}",
                ])

    print(f"  Saved: {csv_path}")

    # Print key comparison for random_order (stable vs unstable)
    if "random_order" in gen_data:
        rows = gen_data["random_order"]
        stable_lens = [
            smiles_token_length(r["smiles"])
            for r in rows if r["is_valid"] and r["is_stable"]
        ]
        unstable_lens = [
            smiles_token_length(r["smiles"])
            for r in rows if r["is_valid"] and not r["is_stable"]
        ]
        if stable_lens and unstable_lens:
            diff = np.mean(stable_lens) - np.mean(unstable_lens)
            print(f"  Random Order: Stable avg len={np.mean(stable_lens):.1f}, "
                  f"Unstable avg len={np.mean(unstable_lens):.1f}, "
                  f"diff={diff:.1f} tokens")

    # Histogram figure: stable vs unstable (valid only)
    if "random_order" in gen_data:
        rows = gen_data["random_order"]
        stable_lens = [
            smiles_token_length(r["smiles"])
            for r in rows if r["is_valid"] and r["is_stable"]
        ]
        unstable_lens = [
            smiles_token_length(r["smiles"])
            for r in rows if r["is_valid"] and not r["is_stable"]
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        max_len = max(max(stable_lens, default=0), max(unstable_lens, default=0))
        bins = np.arange(0, max_len + 5, 2)
        ax.hist(stable_lens, bins=bins, alpha=0.6, color="#2ca02c",
                label=f"Stable (n={len(stable_lens)})", density=True)
        ax.hist(unstable_lens, bins=bins, alpha=0.6, color="#d62728",
                label=f"Unstable (n={len(unstable_lens)})", density=True)
        ax.set_xlabel("SMILES Token Length", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("SMILES Length: Stable vs. Unstable Molecules (Random Order)",
                      fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(PLOTS_DIR / f"seq_len_valid_vs_invalid.{ext}",
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: seq_len_valid_vs_invalid.{pdf,png}")


# ---------------------------------------------------------------------------
# 7. Figure G1: NLL vs. CV Correlation
# ---------------------------------------------------------------------------
def build_nll_cv_correlation(gen_data):
    """Scatter plot + correlation: NLL vs. CV within generated samples."""
    print("\n=== Figure G1: NLL vs. CV Correlation ===")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    csv_path = APPENDIX_DIR / "nll_cv_correlation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "N", "Pearson_r", "Spearman_rho"])

        for idx, strat in enumerate(STRATEGIES):
            ax = axes[idx]
            if strat not in gen_data:
                ax.set_title(f"{STRATEGY_LABELS[strat]} (no data)")
                writer.writerow([STRATEGY_LABELS[strat], 0, "N/A", "N/A"])
                continue

            rows = gen_data[strat]
            nlls = np.array([r["nll_mean"] for r in rows])
            cvs = np.array([r["cv"] for r in rows])

            # Pearson correlation
            pearson_r = np.corrcoef(nlls, cvs)[0, 1]

            # Spearman (rank correlation)
            from scipy.stats import spearmanr
            spearman_rho, _ = spearmanr(nlls, cvs)

            writer.writerow([
                STRATEGY_LABELS[strat], len(rows),
                f"{pearson_r:.4f}", f"{spearman_rho:.4f}",
            ])

            ax.scatter(nlls, cvs, alpha=0.2, s=8, c=STRATEGY_COLORS[strat], rasterized=True)
            ax.set_xlabel("NLL Mean", fontsize=10)
            ax.set_ylabel("CV", fontsize=10)
            ax.set_title(
                f"{STRATEGY_LABELS[strat]}  (r={pearson_r:.3f}, ρ={spearman_rho:.3f})",
                fontsize=11, fontweight="bold",
            )
            ax.grid(alpha=0.3)

            print(f"  {STRATEGY_LABELS[strat]}: Pearson r={pearson_r:.4f}, "
                  f"Spearman ρ={spearman_rho:.4f}")

    fig.suptitle("NLL vs. CV Correlation (Generated Samples)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"nll_cv_correlation.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: nll_cv_correlation.{pdf,png}")
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Building Appendix Artifacts")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    table1 = load_table1()
    print(f"  permutation_consistency_qm9.csv: {len(table1)} strategies")

    gen_data = load_generated_invariance()
    for strat, rows in gen_data.items():
        print(f"  generated_invariance_{strat}: {len(rows)} samples")

    # Build artifacts
    check_figure_a1()
    build_ece_by_token_type(table1)

    print("\n" + "=" * 60)
    print("Done! All artifacts written.")
    print(f"  Figures: {PLOTS_DIR}")
    print(f"  Tables:  {APPENDIX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
