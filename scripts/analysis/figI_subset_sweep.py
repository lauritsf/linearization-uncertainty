"""Figure: QM9 subset-size sweep across all four linearization strategies.

Eight-panel figure (2 rows x 4 cols):
  Row 1: Validity, Uniqueness, Novelty, NLL/tok
  Row 2: Atm. Stable, Mol. Stable, FCD, PGD

X-axis: N_train in {128, 1000, 10000} on a log scale.
Each strategy is a line with mean +/- std shading across 3 seeds,
plus individual seed lines shown as thin dashed traces.

Output: plots/subset_sweep_qm9.{pdf,png}
"""

import csv
import pathlib
import statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

STRATEGIES = {
    "random_order": {"color": "#C44E52", "label": "Random", "zorder": 4},
    "min_degree_first": {"color": "#55A868", "label": "Min-deg", "zorder": 3},
    "max_degree_first": {"color": "#DD8452", "label": "Max-deg", "zorder": 2},
    "anchor_expansion": {"color": "#4C72B0", "label": "Anchor", "zorder": 1},
}
SEEDS = [0, 1, 2]
SIZES = [128, 1000, 10000]

PANELS = [
    ("test/validity", "Validity (%)", True),
    ("test/uniqueness", "Uniqueness (%)", True),
    ("test/novelty", "Novelty (%)", True),
    ("test/loss", "NLL / tok", False),
    ("test/atm_stable", "Atm. Stable (%)", True),
    ("test/mol_stable", "Mol. Stable (%)", True),
    ("test/test_fcd", "FCD", False),
    ("test/test_mol_pgd", "PGD", False),
]


def find_metrics_csv(strategy: str, seed: int, size: int) -> pathlib.Path | None:
    """Find the most recent metrics CSV for a given (strategy, seed, size)."""
    base = ROOT / f"logs/test/QM9/{strategy}/llama-s/{seed}/N_{size}/runs"
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/csv_logs/version_0/metrics.csv"), reverse=True)
    return candidates[0] if candidates else None


def read_metric(csv_path: pathlib.Path, key: str) -> float | None:
    """Read the last recorded value of a metric from a CSV log."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    val = rows[-1].get(key, "").strip()
    return float(val) if val else None


def load_data() -> dict:
    """Load per-seed metric values.

    Returns:
        {strategy: {size: {metric: [val_s0, val_s1, val_s2]}}}
    """
    data = {}
    for strategy in STRATEGIES:
        data[strategy] = {}
        for size in SIZES:
            seed_vals = {m: [] for m, *_ in PANELS}
            for seed in SEEDS:
                path = find_metrics_csv(strategy, seed, size)
                if path is None:
                    print(f"  MISS: {strategy} N={size} seed={seed}")
                    for metric, *_ in PANELS:
                        seed_vals[metric].append(None)
                    continue
                for metric, *_ in PANELS:
                    v = read_metric(path, metric)
                    seed_vals[metric].append(v)

            data[strategy][size] = seed_vals
    return data


def plot():
    """Generate the subset sweep figure."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.constrained_layout.use": True,
    })

    data = load_data()
    x = np.array(SIZES)

    fig, axes_arr = plt.subplots(2, 4, figsize=(11, 4.2))
    axes = axes_arr.flatten()

    handles = []
    for ax, (metric, ylabel, as_pct) in zip(axes, PANELS, strict=False):
        for strategy, style in STRATEGIES.items():
            # Collect per-seed arrays
            per_seed = []  # list of (size_array,) per seed
            for seed_idx in range(len(SEEDS)):
                vals = []
                for size in SIZES:
                    v = data[strategy][size][metric][seed_idx]
                    if v is not None and as_pct:
                        v *= 100
                    vals.append(v)
                per_seed.append(vals)

            # Plot individual seed lines (thin, dashed, low alpha)
            for seed_vals in per_seed:
                y_seed = np.array([v if v is not None else np.nan for v in seed_vals])
                ax.plot(
                    x,
                    y_seed,
                    color=style["color"],
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.35,
                    zorder=style["zorder"] - 0.5,
                )

            # Compute mean/std from non-None values
            means, stds = [], []
            for size in SIZES:
                raw = data[strategy][size][metric]
                valid = [v for v in raw if v is not None]
                if valid:
                    m = statistics.mean(valid)
                    s = statistics.stdev(valid) if len(valid) >= 2 else 0.0
                    if as_pct:
                        m *= 100
                        s *= 100
                    means.append(m)
                    stds.append(s)
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            means = np.array(means)
            stds = np.array(stds)

            # Plot aggregate (thick solid line + shading)
            (line,) = ax.plot(
                x,
                means,
                marker="o",
                markersize=4,
                color=style["color"],
                linewidth=1.8,
                zorder=style["zorder"],
                label=style["label"],
            )
            ax.fill_between(
                x,
                means - stds,
                means + stds,
                color=style["color"],
                alpha=0.15,
                zorder=0,
            )

            if ax is axes[0]:
                handles.append(line)

        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels(["128", "1k", "10k"])
        ax.set_xlabel("$N_{\\mathrm{train}}$")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)
        if as_pct:
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(max(ylo, 0), min(yhi, 105))

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.06),
        framealpha=0.9,
        fontsize=9,
    )

    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out_path = out_dir / f"figI_subset_sweep_qm9.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    plot()
