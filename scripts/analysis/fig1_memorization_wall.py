"""Plot Figure 1: Planar training curves showing the Memorization Wall.

Uses seed 0 (200k baseline) to show full divergence between strategies.
Dual-axis: val/loss (left) and val/vun_score (right) vs. training step.
Smoothed with exponential moving average for readability.

VUN = Validity × Uniqueness × Novelty. More informative than Validity alone:
biased strategies can match random Validity while having far lower VUN due to
repetition (low Uniqueness) and memorisation of training graphs (low Novelty).

Output: results/figures/planar_memorization_wall.{pdf,png}
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLANAR_ROOT = pathlib.Path("logs/train/planar")
STRATEGIES = {
    "random_order": {"color": "#C44E52", "label": "Random", "zorder": 4},
    "min_degree_first": {"color": "#55A868", "label": "Min-deg", "zorder": 3},
    "max_degree_first": {"color": "#DD8452", "label": "Max-deg", "zorder": 2},
    "anchor_expansion": {"color": "#4C72B0", "label": "Anchor", "zorder": 1},
}
SEED = 0


def ema(values, alpha=0.05):
    """Exponential moving average."""
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def load_training_metrics(strategy: str) -> dict:
    """Load val/loss and val/vun_score from training CSV."""
    base = PLANAR_ROOT / strategy / "llama2-s" / str(SEED)
    candidates = sorted(base.glob("runs/*/csv_logs/version_0/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics for {strategy} seed {SEED}")

    steps_loss, val_loss = [], []
    steps_vun, val_vun = [], []

    with open(candidates[-1]) as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if row.get("val/loss", ""):
                steps_loss.append(step)
                val_loss.append(float(row["val/loss"]))
            if row.get("val/vun_score", ""):
                steps_vun.append(step)
                val_vun.append(float(row["val/vun_score"]) * 100)

    return {
        "steps_loss": np.array(steps_loss),
        "val_loss": np.array(val_loss),
        "steps_vun": np.array(steps_vun),
        "val_vun": np.array(val_vun),
    }


def plot():
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax2 = ax1.twinx()

    nll_handles = []
    for strategy, style in STRATEGIES.items():
        try:
            data = load_training_metrics(strategy)
        except FileNotFoundError as e:
            print(f"  WARN: {e}")
            continue

        # Smooth NLL
        smoothed_loss = ema(data["val_loss"], alpha=0.02)
        line, = ax1.plot(
            data["steps_loss"], smoothed_loss,
            color=style["color"], linewidth=1.8, label=style["label"],
            zorder=style["zorder"],
        )
        nll_handles.append(line)

        # Smooth VUN
        if len(data["val_vun"]) > 0:
            smoothed_vun = ema(data["val_vun"], alpha=0.1)
            ax2.plot(
                data["steps_vun"], smoothed_vun,
                color=style["color"], linewidth=1.2, linestyle="--",
                alpha=0.6, zorder=style["zorder"],
            )

    # Memorization wall annotation
    ax1.axvline(x=5000, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ylim = ax1.get_ylim()
    ax1.annotate(
        "Memorization\nWall (~5k)",
        xy=(5000, ylim[0] + (ylim[1] - ylim[0]) * 0.75),
        fontsize=8, color="gray", ha="left",
        xytext=(8000, ylim[0] + (ylim[1] - ylim[0]) * 0.85),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation NLL (per token)")
    ax2.set_ylabel("VUN (%)")
    ax1.set_xlim(0, 200000)
    ax2.set_ylim(-2, 105)

    # Legend
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="black", linewidth=1.2, label="NLL (solid)"),
        Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", alpha=0.6,
               label="VUN (dashed)"),
    ]
    legend1 = ax1.legend(handles=nll_handles, loc="upper left", framealpha=0.9)
    ax1.add_artist(legend1)
    ax1.legend(handles=style_handles, loc="center left", framealpha=0.9,
               bbox_to_anchor=(0.0, 0.45))

    plt.title("Planar ($N{=}128$): Deterministic Strategies Overfit", fontsize=11, pad=8)
    plt.tight_layout()

    out_dir = pathlib.Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out_path = out_dir / f"fig1_memorization_wall.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")

    plt.close()


if __name__ == "__main__":
    plot()
