"""Aggregate generation quality metrics into Table 1a and Table 1b for the paper.

Table 1a — Generative Quality (chemistry metrics on generated molecules):
  Validity ↑ | Unique ↑ | Novelty ↑ | Atm. Stable ↑ | Mol. Stable ↑ | FCD ↓ | PGD ↓

Table 1b — Model Self-Assessment of Generated Sequences:
  Tok/graph (gen) | Tok/graph (resamp) | Self-NLL ↓ | Resamp. NLL ↓ | NLL Gap ↓ | Topo. Unc. ↓

Reads:
  experiments/invariance_eval/qm9/{strategy}/seed_{s}/generated_quality.csv
  experiments/invariance_eval/qm9/{strategy}/seed_{s}/k32/generated_eval.csv

Outputs:
  results/data/generation_quality_qm9.csv
  results/tables/generation_quality_qm9.{md,tex}   (Table 1a)
  results/tables/generation_self_assess_qm9.{md,tex}  (Table 1b)
"""

import csv
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_utils import fmt_mean_std, write_latex, write_latex_multiheader, write_markdown

ROOT = pathlib.Path("experiments/invariance_eval/qm9")
STRATEGIES = ["random_order", "min_degree_first", "max_degree_first", "anchor_expansion"]
SEEDS = [0, 1, 2]

QUALITY_METRICS = [
    "test/validity",
    "test/uniqueness",
    "test/novelty",
    "test/atm_stable",
    "test/mol_stable",
    "test/test_fcd",
    "test/test_mol_pgd",
]

PRETTY_NAMES = {
    "random_order": "Random",
    "min_degree_first": "Min-Degree",
    "max_degree_first": "Max-Degree",
    "anchor_expansion": "Anchor",
}


def find_metrics_csv(base_dir: pathlib.Path) -> str:
    candidates = sorted(base_dir.glob("runs/*/csv_logs/version_0/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv under {base_dir}")
    return str(candidates[-1])


def read_last_row(csv_path: str) -> dict:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows[-1]


def load_gen_quality(seed_dir: pathlib.Path) -> dict | None:
    """Load aggregate generation quality from generated_quality.csv if it exists."""
    quality_path = seed_dir / "generated_quality.csv"
    if not quality_path.exists():
        return None
    with open(quality_path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def load_gen_eval(seed_dir: pathlib.Path) -> dict:
    """Load per-molecule eval data from generated_eval.csv.

    Returns dict with lists: birth_nll, perm_mean_nll, perm_cv, seq_len,
    resamp_seq_len (valid only).
    """
    out = {"birth_nll": [], "perm_mean_nll": [], "perm_cv": [], "seq_len": [], "resamp_seq_len": []}
    audit_path = seed_dir / "k32" / "generated_eval.csv"
    if not audit_path.exists():
        return out
    with open(audit_path) as f:
        for row in csv.DictReader(f):
            valid = row.get("valid", "").strip().lower()
            if valid not in ("true", "1"):
                continue
            for key in out:
                val = row.get(key, "")
                if val:
                    try:
                        out[key].append(float(val))
                    except ValueError:
                        pass
    return out


def _agg(vals):
    """Return (mean, std) or (None, None)."""
    if len(vals) >= 2:
        return statistics.mean(vals), statistics.stdev(vals)
    elif len(vals) == 1:
        return vals[0], 0.0
    return None, None


def main():
    results = []

    for strategy in STRATEGIES:
        quality_vals = {m: [] for m in QUALITY_METRICS}
        birth_nlls, resamp_nlls, perm_cvs = [], [], []
        gen_seq_lens, resamp_seq_lens = [], []

        for seed in SEEDS:
            seed_dir = ROOT / strategy / f"seed_{seed}"

            # --- Quality metrics ---
            gen_quality = load_gen_quality(seed_dir)
            if gen_quality is not None:
                key_map = {
                    "test/validity": "validity",
                    "test/atm_stable": "atm_stable",
                    "test/mol_stable": "mol_stable",
                    "test/uniqueness": "uniqueness",
                    "test/novelty": "novelty",
                    "test/test_fcd": "test_fcd",
                    "test/test_mol_pgd": "test_mol_pgd",
                }
                for m in QUALITY_METRICS:
                    gen_key = key_map.get(m, m.replace("test/", ""))
                    val = gen_quality.get(gen_key, "")
                    if val:
                        quality_vals[m].append(float(val))
            else:
                try:
                    native_dir = seed_dir / "native"
                    csv_path = find_metrics_csv(native_dir)
                    row = read_last_row(csv_path)
                except (FileNotFoundError, IndexError):
                    print(f"  WARN: Missing quality data for {strategy}/seed_{seed}")
                    continue
                for m in QUALITY_METRICS:
                    val = row.get(m, "")
                    if val:
                        quality_vals[m].append(float(val))

            # --- Self-assessment from generated_eval.csv ---
            result = load_gen_eval(seed_dir)
            if result["birth_nll"]:
                birth_nlls.append(statistics.mean(result["birth_nll"]))
            if result["perm_mean_nll"]:
                resamp_nlls.append(statistics.mean(result["perm_mean_nll"]))
            if result["perm_cv"]:
                perm_cvs.append(statistics.mean(result["perm_cv"]))
            if result["seq_len"]:
                gen_seq_lens.append(statistics.mean(result["seq_len"]))
            if result["resamp_seq_len"]:
                resamp_seq_lens.append(statistics.mean(result["resamp_seq_len"]))

        if not quality_vals["test/validity"]:
            print(f"  SKIP: No data for {strategy}")
            continue

        quality_agg = {}
        for m in QUALITY_METRICS:
            quality_agg[m] = _agg(quality_vals[m])

        birth_nll_agg = _agg(birth_nlls)
        resamp_nll_agg = _agg(resamp_nlls)
        perm_cv_agg = _agg(perm_cvs)
        gen_seq_len_agg = _agg(gen_seq_lens)
        resamp_seq_len_agg = _agg(resamp_seq_lens)

        # NLL Gap = Resamp. NLL - Self-NLL (per seed, then aggregate)
        nll_gap_vals = []
        for seed in SEEDS:
            seed_dir = ROOT / strategy / f"seed_{seed}"
            result = load_gen_eval(seed_dir)
            if result["birth_nll"] and result["perm_mean_nll"] and len(result["birth_nll"]) == len(result["perm_mean_nll"]):
                gaps = [
                    r - b
                    for r, b in zip(result["perm_mean_nll"], result["birth_nll"], strict=True)
                ]
                nll_gap_vals.append(statistics.mean(gaps))
        nll_gap_agg = _agg(nll_gap_vals)

        results.append({
            "strategy": strategy,
            "gen_seq_len": gen_seq_len_agg,
            "resamp_seq_len": resamp_seq_len_agg,
            "birth_nll": birth_nll_agg,
            "resamp_nll": resamp_nll_agg,
            "nll_gap": nll_gap_agg,
            "perm_cv": perm_cv_agg,
            **{m: quality_agg[m] for m in QUALITY_METRICS},
        })

    # --- Write CSV ---
    data_dir = pathlib.Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "table1_generation_quality_qm9.csv"
    csv_cols = (
        ["strategy", "gen_seq_len", "resamp_seq_len", "birth_nll",
         "resamp_nll", "nll_gap", "perm_cv"] + QUALITY_METRICS
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for r in results:
            def fmt_pair(pair):
                m, s = pair
                return f"{m:.4f}" if m is not None else ""
            row = [
                r["strategy"],
                fmt_pair(r["gen_seq_len"]),
                fmt_pair(r["resamp_seq_len"]),
                fmt_pair(r["birth_nll"]),
                fmt_pair(r["resamp_nll"]),
                fmt_pair(r["nll_gap"]),
                fmt_pair(r["perm_cv"]),
            ]
            for m in QUALITY_METRICS:
                mean, _ = r[m]
                row.append(f"{mean:.4f}" if mean is not None else "")
            w.writerow(row)
    print(f"  Saved: {csv_path}")

    table_dir = pathlib.Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 1a: Generative Quality ---
    headers_1a = [
        "Strategy",
        "Validity ↑", "Unique ↑", "Novelty ↑",
        "Atm. Stable ↑", "Mol. Stable ↑",
        "FCD ↓", "PGD ↓",
    ]
    PCT_METRICS = {"test/validity", "test/atm_stable", "test/mol_stable",
                   "test/uniqueness", "test/novelty"}
    rows_1a = []
    for r in results:
        row = [PRETTY_NAMES[r["strategy"]]]
        for m in QUALITY_METRICS:
            mean, std = r[m]
            if mean is None:
                row.append("—")
            elif m in PCT_METRICS:
                row.append(fmt_mean_std(mean * 100, std * 100, prec=1))
            else:
                row.append(fmt_mean_std(mean, std))
        rows_1a.append(row)

    write_markdown(table_dir / "table1a_generation_quality_qm9.md", headers_1a, rows_1a)
    write_latex(
        table_dir / "table1a_generation_quality_qm9.tex", headers_1a, rows_1a,
        caption=(
            "Generative quality on QM9 (3-seed mean$_{\\pm\\text{std}}$; 10{,}000 generated "
            "molecules per model). Metric definitions in Appendix~\\ref{app:metric_defs}."
        ),
        label="tab:generation-quality",
        starred=True,
    )

    # --- Table 1b: Model Self-Assessment ---
    headers_1b = [
        "Strategy",
        "Tok/graph (gen)", "Tok/graph (resamp)",
        "Self-NLL ↓", "Resamp. NLL ↓",
        "Lin. Unc. ↓",
    ]
    col_groups_1b = [
        (None,         ["Strategy"]),
        ("Tok/graph",  ["(Gen.)", "(Resamp.)"]),
        ("NLL ↓",      ["(Gen.)", "(Resamp.)"]),
        (None,         ["Lin. Unc. ↓"]),
    ]
    rows_1b = []
    for r in results:
        gen_tok = fmt_mean_std(*r["gen_seq_len"], prec=1) if r["gen_seq_len"][0] is not None else "—"
        resamp_tok = fmt_mean_std(*r["resamp_seq_len"], prec=1) if r["resamp_seq_len"][0] is not None else "—"
        self_nll = fmt_mean_std(*r["birth_nll"]) if r["birth_nll"][0] is not None else "—"
        resamp_nll = fmt_mean_std(*r["resamp_nll"]) if r["resamp_nll"][0] is not None else "—"
        topo_unc = fmt_mean_std(*r["perm_cv"], prec=3) if r["perm_cv"][0] is not None else "—"
        rows_1b.append([
            PRETTY_NAMES[r["strategy"]],
            gen_tok, resamp_tok,
            self_nll, resamp_nll,
            topo_unc,
        ])

    write_markdown(table_dir / "table1b_self_assessment_qm9.md", headers_1b, rows_1b)
    write_latex_multiheader(
        table_dir / "table1b_self_assessment_qm9.tex", col_groups_1b, rows_1b,
        caption="Model self-assessment of generated sequences on QM9 (3-seed mean$_{\\pm\\text{std}}$, "
                "valid molecules only). Gen.\\ = model's own generated trajectory; "
                "Resamp.\\ = same molecule re-linearized via the native strategy ($K{=}32$).",
        label="tab:generation-self-assess",
        starred=True,
    )

    # --- Pretty print ---
    print("\n=== Table 1a: Generative Quality (QM9) ===")
    col_w = [max([len(str(headers_1a[i]))] + [len(str(row[i])) for row in rows_1a])
             for i in range(len(headers_1a))]
    print("  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers_1a)))
    print("-" * sum(col_w))
    for row in rows_1a:
        print("  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))

    print("\n=== Table 1b: Model Self-Assessment (QM9) ===")
    col_w = [max([len(str(headers_1b[i]))] + [len(str(row[i])) for row in rows_1b])
             for i in range(len(headers_1b))]
    print("  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers_1b)))
    print("-" * sum(col_w))
    for row in rows_1b:
        print("  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))


if __name__ == "__main__":
    main()
