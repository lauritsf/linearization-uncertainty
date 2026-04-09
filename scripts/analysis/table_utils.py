"""Utilities for writing paper-ready tables in multiple formats.

Provides write_markdown() and write_latex() for consistent multi-format output
alongside the canonical CSV files.
"""

import pathlib
import re
from collections.abc import Sequence


def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters outside of $...$ and \\cmd{...} spans."""
    s = str(s)
    # Split on math spans ($...$) and LaTeX commands with braces (\cmd{...})
    parts = re.split(r"(\$[^$]*\$|\\[a-zA-Z]+\{[^}]*\})", s)
    result = []
    for part in parts:
        if part.startswith("$") or part.startswith("\\"):
            result.append(part)  # math or command — leave untouched
        else:
            result.append(part.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&"))
    return "".join(result)


def write_markdown(path, headers: Sequence[str], rows: Sequence[Sequence], alignments=None):
    """Write a GitHub-flavored markdown table.

    Args:
        path: Output path (.md).
        headers: Column header strings.
        rows: Iterable of row iterables (values will be str-converted).
        alignments: List of 'l', 'c', or 'r' per column (default: first col left, rest right).
    """
    path = pathlib.Path(path)
    n = len(headers)
    if alignments is None:
        alignments = ["l"] + ["r"] * (n - 1)

    sep_map = {"l": ":---", "c": ":---:", "r": "---:"}
    sep_row = [sep_map.get(a, "---") for a in alignments]

    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(sep_row) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


def write_latex(path, headers: Sequence[str], rows: Sequence[Sequence],
                caption: str = "", label: str = "", alignments=None, starred: bool = False):
    r"""Write a LaTeX booktabs table fragment (can be \input'd directly).

    Args:
        path: Output path (.tex).
        headers: Column header strings.
        rows: Iterable of row iterables.
        caption: Table caption (used in \\caption{}).
        label: Table label (used in \\label{}).
        alignments: LaTeX column spec chars per column (default: 'l' + 'r'*(n-1)).
        starred: If True, use \begin{table*} instead of \begin{table}.
    """
    path = pathlib.Path(path)
    n = len(headers)
    if alignments is None:
        alignments = ["l"] + ["r"] * (n - 1)
    col_spec = "".join(alignments)

    esc = _latex_escape

    env_name = "table*" if starred else "table"
    lines = []
    if caption or label:
        lines.append(rf"\begin{{{env_name}}}[!ht]")
        lines.append(r"  \centering")
        if caption:
            lines.append(f"  \\caption{{{esc(caption)}}}")
        if label:
            lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    lines.append("    " + " & ".join(esc(h) for h in headers) + r" \\")
    lines.append(r"    \midrule")
    for row in rows:
        lines.append("    " + " & ".join(esc(v) for v in row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    if caption or label:
        lines.append(rf"\end{{{env_name}}}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


def write_latex_multiheader(path, col_groups, rows,
                            caption: str = "", label: str = "", alignments=None, starred: bool = False):
    r"""Write a LaTeX booktabs table with a two-row multi-level header.

    Args:
        path: Output path (.tex).
        col_groups: List of (group_label, [subcolumn_labels]) tuples.
                    Use group_label=None for a standalone column with no group header.
                    Example: [(None, ["Strategy"]), ("NLL/tok ↓", ["(N)", "(R)"]), ...]
        rows: Iterable of row iterables (must match total number of leaf columns).
        caption, label, alignments, starred: same as write_latex().
    """
    path = pathlib.Path(path)

    # Flatten to get all leaf column labels and total count
    leaf_headers = []
    for _, subcols in col_groups:
        leaf_headers.extend(subcols)
    n = len(leaf_headers)

    if alignments is None:
        alignments = ["l"] + ["r"] * (n - 1)
    col_spec = "".join(alignments)

    esc = _latex_escape

    env_name = "table*" if starred else "table"
    lines = []
    if caption or label:
        lines.append(rf"\begin{{{env_name}}}[!ht]")
        lines.append(r"  \centering")
        if caption:
            lines.append(f"  \\caption{{{esc(caption)}}}")
        if label:
            lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # Top header row: group labels (multicolumn for groups, empty for singletons)
    top_cells = []
    cmidrules = []
    col_idx = 1  # 1-based for \cmidrule
    for group_label, subcols in col_groups:
        span = len(subcols)
        if group_label is None:
            top_cells.append(r"\multicolumn{" + str(span) + r"}{l}{}")
        else:
            top_cells.append(r"\multicolumn{" + str(span) + r"}{c}{" + esc(group_label) + "}")
            if span > 1:
                cmidrules.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + span - 1}}}")
        col_idx += span
    lines.append("    " + " & ".join(top_cells) + r" \\")
    if cmidrules:
        lines.append("    " + " ".join(cmidrules))

    # Bottom header row: leaf column labels
    lines.append("    " + " & ".join(esc(h) for h in leaf_headers) + r" \\")
    lines.append(r"    \midrule")

    for row in rows:
        lines.append("    " + " & ".join(esc(v) for v in row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    if caption or label:
        lines.append(rf"\end{{{env_name}}}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


def fmt_mean_std(mean, std, prec=3):
    """Format a mean ± std pair as a string."""
    try:
        m, s = float(mean), float(std)
        return f"${m:.{prec}f}_{{\\pm {s:.{prec}f}}}$"
    except (TypeError, ValueError):
        return "—"
