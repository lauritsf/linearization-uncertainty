"""Master diversity saturation grid: 3 rows (datasets) × 4 columns (metrics).

Rows: Planar (N=128), QM9 Subset (N=128), QM9 Full (N=97k)
Columns: Unique Ratio, Unique 3-grams, 3-gram Entropy, Compression (bits/char)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Style & palette
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.constrained_layout.use": False,
    }
)

STRATEGY_PALETTE = {
    "random_order": "#C44E52",
    "min_degree_first": "#55A868",
    "max_degree_first": "#DD8452",
    "anchor_expansion": "#4C72B0",
}
STRATEGY_LABELS = {
    "random_order": "Random",
    "min_degree_first": "Min-deg",
    "max_degree_first": "Max-deg",
    "anchor_expansion": "Anchor",
}
STRATEGY_ORDER = ["random_order", "min_degree_first", "max_degree_first", "anchor_expansion"]

# Mix percentages to evaluate
MIX_PERCENTS = [0.1, 0.3]
MIX_LINESTYLES = {0.1: "--", 0.3: ":"}
MIX_LABELS = {0.1: "10% Random", 0.3: "30% Random"}

# We focus on the 4 most intuitive metrics
METRICS = [
    ("Unique_Ratio", "Unique Ratio"),
    ("Ngram_Coverage", "Unique 3-grams"),
    ("Entropy", "3-gram Entropy"),
    ("Compression_BPC", "Compression (bits/char)"),
]


# ---------------------------------------------------------------------------
# Helper: human-readable formatter
# ---------------------------------------------------------------------------
def human_format(x, _pos):
    if x >= 1_000_000:
        return f"{int(x / 1_000_000)}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k"
    return f"{int(x)}"


formatter = FuncFormatter(human_format)


# ---------------------------------------------------------------------------
# Helper: saturation curve with 95% CI
# ---------------------------------------------------------------------------
def plot_saturation(ax, df, metric="Unique_Ratio"):
    for strat in STRATEGY_ORDER:
        sub = df[df["Strategy"] == strat]
        if sub.empty:
            continue
        agg = sub.groupby("Total_Samples")[metric].agg(["mean", "std", "count"])
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        x = agg.index.values
        y = agg["mean"].values
        ci = 1.96 * agg["se"].values
        ax.plot(x, y, label=STRATEGY_LABELS[strat], color=STRATEGY_PALETTE[strat], lw=1.8)
        ax.fill_between(x, y - ci, y + ci, color=STRATEGY_PALETTE[strat], alpha=0.15)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="both", ls="--", alpha=0.5)


# ---------------------------------------------------------------------------
# Helper: mixed strategy builder
# ---------------------------------------------------------------------------
def build_mixed_df(df, base_strat, random_frac=0.1):
    """Simulate a corpus where random_frac of graphs use random_order,
    the rest use base_strat. Returns rows relabelled as 'mix_{base_strat}'."""
    rand_rows = df[df["Strategy"] == "random_order"]
    base_rows = df[df["Strategy"] == base_strat]
    parts = []
    for spg, grp_rand in rand_rows.groupby("Samples_Per_Graph"):
        grp_base = base_rows[base_rows["Samples_Per_Graph"] == spg]
        if grp_base.empty:
            continue
        n = len(grp_rand)
        n_rand = max(1, round(random_frac * n))
        n_base = n - n_rand
        sample = pd.concat([
            grp_rand.sample(n=n_rand, replace=False, random_state=42),
            grp_base.sample(n=n_base, replace=False, random_state=42),
        ])
        sample = sample.copy()
        sample["Strategy"] = f"mix_{base_strat}"
        parts.append(sample)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Helper: mixed strategy saturation plot
# ---------------------------------------------------------------------------
def plot_saturation_mixed(ax, df_pure, dfs_mixed_by_pct, metric="Unique_Ratio"):
    """Plot pure strategies (solid), 10% mixed (dashed), and 30% mixed (dotted).
    Line style indicates mix percentage: solid=pure, --=10%, :=30%."""
    # All 4 pure strategies (solid lines)
    plot_saturation(ax, df_pure, metric=metric)

    # Mixed strategy curves for each percentage
    for mix_pct in MIX_PERCENTS:
        df_mixed = dfs_mixed_by_pct[mix_pct]
        ls = MIX_LINESTYLES[mix_pct]
        label = MIX_LABELS[mix_pct]

        for i, strat in enumerate(STRATEGY_ORDER[1:]):  # Skip "random_order"
            sub = df_mixed[df_mixed["Strategy"] == f"mix_{strat}_{mix_pct}"]
            if sub.empty:
                continue
            agg = sub.groupby("Total_Samples")[metric].agg(["mean", "std", "count"])
            agg["se"] = agg["std"] / np.sqrt(agg["count"])
            x, y, ci = agg.index.values, agg["mean"].values, 1.96 * agg["se"].values

            # Label once per mix_pct (on first strategy for that percentage)
            plot_label = label if i == 0 else None
            ax.plot(x, y, label=plot_label,
                    color=STRATEGY_PALETTE[strat], lw=1.8, ls=ls)
            ax.fill_between(x, y - ci, y + ci, color=STRATEGY_PALETTE[strat], alpha=0.15)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def _load_optional(path):
    """Load CSV if it exists, otherwise return None."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


try:
    planar = pd.read_csv("experiments/diversity/diversity_saturation_planar.csv")
    qm9_sub = pd.read_csv("experiments/diversity/diversity_saturation_qm9_subset.csv")
    qm9_full = pd.read_csv("experiments/diversity/diversity_saturation_qm9_full.csv")
except Exception as e:
    print(f"Error loading mandatory CSVs: {e}")
    print(
        "Ensure diversity_saturation_{planar,qm9_subset,qm9_full}.csv exist in experiments/diversity/"
    )
    exit(1)

# Optional CSVs (generated by additional eval runs for new subset sizes)
qm9_1k = _load_optional("experiments/diversity/diversity_saturation_qm9_1k.csv")
qm9_10k = _load_optional("experiments/diversity/diversity_saturation_qm9_10k.csv")

# Build datasets and row labels dynamically (skip None entries)
_all = [
    (planar, "Planar (N=128)"),
    (qm9_sub, "QM9 Subset (N=128)"),
    (qm9_1k, "QM9 Subset (N=1k)"),
    (qm9_10k, "QM9 Subset (N=10k)"),
    (qm9_full, "QM9 Full (N=97k)"),
]
datasets, ROW_LABELS = zip(*[(df, lbl) for df, lbl in _all if df is not None], strict=True)
datasets, ROW_LABELS = list(datasets), list(ROW_LABELS)

# Transform Compression_Ratio to bits per character
for df in datasets:
    df["Compression_BPC"] = df["Compression_Ratio"] * 8

# Build mixed strategy datasets (10% and 30% random + base)
BASE_STRATS = ["max_degree_first", "min_degree_first", "anchor_expansion"]
datasets_mixed_by_pct = {0.1: [], 0.3: []}

for df in datasets:
    for mix_pct in MIX_PERCENTS:
        parts = []
        for strat in BASE_STRATS:
            mixed_df = build_mixed_df(df, strat, random_frac=mix_pct)
            # Relabel strategy to include mix percentage
            mixed_df = mixed_df.copy()
            mixed_df["Strategy"] = mixed_df["Strategy"].str.replace(
                f"mix_{strat}", f"mix_{strat}_{mix_pct}"
            )
            parts.append(mixed_df)
        datasets_mixed_by_pct[mix_pct].append(pd.concat(parts, ignore_index=True))

# ---------------------------------------------------------------------------
# Figure: Dynamic rows × 4 columns
# ---------------------------------------------------------------------------
n_rows = len(datasets)
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3 * n_rows), squeeze=False)

for row, (df, row_label) in enumerate(zip(datasets, ROW_LABELS, strict=True)):
    for col, (metric, metric_label) in enumerate(METRICS):
        ax = axes[row, col]
        plot_saturation(ax, df, metric=metric)

        # Reference line only for Unique Ratio
        if metric == "Unique_Ratio":
            ax.axhline(1.0, ls="-", color="0.3", lw=0.8, alpha=0.5)

        # Column titles on top row
        if row == 0:
            ax.set_title(metric_label, fontsize=12, pad=10, fontweight="bold")

        # X-axis label only on bottom row
        if row == n_rows - 1:
            ax.set_xlabel("Total training samples", labelpad=10)
        else:
            ax.set_xlabel("")

        # Row label on leftmost column
        if col == 0:
            # Use text annotation for row label to avoid squashing y-axis
            ax.annotate(
                row_label,
                xy=(-0.45, 0.5),
                xycoords="axes fraction",
                rotation=90,
                va="center",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
            ylabel_map = {
                "Unique_Ratio": "Unique Ratio",
                "Ngram_Coverage": "Unique 3-grams",
                "Entropy": "3-gram Entropy",
                "Compression_BPC": "Bits / char",
            }
            ax.set_ylabel(ylabel_map.get(metric, metric_label))
        else:
            ax.set_ylabel("")

# Single shared legend at the bottom
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=True,
    framealpha=1.0,
    edgecolor="0.5",
    bbox_to_anchor=(0.5, -0.05),
    fontsize=11,
)

# Adjust spacing manually to account for the legend and row labels
plt.subplots_adjust(bottom=0.1, left=0.15, top=0.92, right=0.95, hspace=0.3, wspace=0.25)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

fig.savefig(f"{output_dir}/figA2_diversity_grid.pdf", bbox_inches="tight")
fig.savefig(f"{output_dir}/figA2_diversity_grid.png", dpi=300, bbox_inches="tight")
print(f"Saved {output_dir}/figA2_diversity_grid.{{pdf,png}}")

# ---------------------------------------------------------------------------
# Figure 2: Dynamic rows × 4 columns — mixed strategies (10% and 30% random blends)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(n_rows, 4, figsize=(16, 3 * n_rows), squeeze=False)

ylabel_map = {
    "Unique_Ratio": "Unique Ratio",
    "Ngram_Coverage": "Unique 3-grams",
    "Entropy": "3-gram Entropy",
    "Compression_BPC": "Bits / char",
}

for row, (df_pure, row_label) in enumerate(zip(datasets, ROW_LABELS, strict=True)):
    # Build dfs_mixed_by_pct for this row
    dfs_mixed_by_pct = {
        pct: datasets_mixed_by_pct[pct][row] for pct in MIX_PERCENTS
    }

    for col, (metric, metric_label) in enumerate(METRICS):
        ax = axes2[row, col]
        plot_saturation_mixed(ax, df_pure, dfs_mixed_by_pct, metric=metric)

        if metric == "Unique_Ratio":
            ax.axhline(1.0, ls="-", color="0.3", lw=0.8, alpha=0.5)
        if row == 0:
            ax.set_title(metric_label, fontsize=12, pad=10, fontweight="bold")
        if row == n_rows - 1:
            ax.set_xlabel("Total training samples", labelpad=10)
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.annotate(row_label, xy=(-0.45, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", ha="center",
                        fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel_map.get(metric, metric_label))
        else:
            ax.set_ylabel("")

handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc="lower center", ncol=7, frameon=True,
            framealpha=1.0, edgecolor="0.5", bbox_to_anchor=(0.5, -0.05), fontsize=10)
plt.subplots_adjust(bottom=0.1, left=0.15, top=0.92, right=0.95, hspace=0.3, wspace=0.25)

fig2.savefig(f"{output_dir}/figA3_diversity_mixed.pdf", bbox_inches="tight")
fig2.savefig(f"{output_dir}/figA3_diversity_mixed.png", dpi=300, bbox_inches="tight")
print(f"Saved {output_dir}/figA3_diversity_mixed.{{pdf,png}}")
