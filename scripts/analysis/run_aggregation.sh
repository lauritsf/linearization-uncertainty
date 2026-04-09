#!/usr/bin/env bash
# run_aggregation.sh — Run all CPU-side analysis and figure generation from existing eval outputs.
# Prerequisites: the evaluation scripts listed in README.md ("Reproducing Paper Results") must
# have been run and their outputs written under experiments/ and logs/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
ANALYSIS="$ROOT/scripts/analysis"

echo "[1/8] Aggregating permutation consistency (→ Table 2)..."
$PYTHON "$ANALYSIS/table2_permutation_consistency.py"

echo "[2/8] Computing sequence lengths per strategy (→ Tokens/graph)..."
$PYTHON "$ANALYSIS/compute_seq_lengths.py"

echo "[3/8] Aggregating generation quality (→ Tables 1a/1b)..."
$PYTHON "$ANALYSIS/table1_generation_quality.py"

echo "[4/8] Aggregating 4×4 cross-eval matrices (→ Tables C1–C6)..."
$PYTHON "$ANALYSIS/tableC_cross_eval_matrix.py"

echo "[5/8] Aggregating dataset-size subset sweep (→ Table I1)..."
$PYTHON "$ANALYSIS/tableI_subset_sweep.py"

echo "[6/8] Building appendix artifacts (ECE by token type)..."
$PYTHON "$ANALYSIS/appendix_ece_analysis.py"

echo "[7/8] Plotting main figures..."
$PYTHON "$ANALYSIS/fig1_memorization_wall.py"
$PYTHON "$ANALYSIS/figA1_diversity_saturation.py"
$PYTHON "$ANALYSIS/figA2_A3_diversity_grid.py"

echo "[8/8] Plotting cross-evaluation heatmaps..."
$PYTHON "$ANALYSIS/figC_cross_eval_heatmaps.py"

echo ""
echo "Done. Outputs in results/:"
echo "  Tables:"
echo "    results/data/table2_permutation_consistency_qm9.csv"
echo "    results/tables/table2_permutation_consistency_qm9.{md,tex}"
echo "    results/data/table1_generation_quality_qm9.csv"
echo "    results/tables/table1a_generation_quality_qm9.{md,tex}"
echo "    results/tables/table1b_self_assessment_qm9.{md,tex}"
echo "    results/data/tableC_cross_eval_qm9.csv"
echo "    results/tables/tableC{1-6}_cross_eval_matrix.{md,tex}"
echo "    results/data/tableI_subset_sweep_qm9.csv"
echo "    results/tables/tableI_subset_sweep_qm9.{md,tex}"
echo "  Figures:"
echo "    results/figures/fig1_memorization_wall.{pdf,png}"
echo "    results/figures/figA1_diversity_saturation.{pdf,png}"
echo "    results/figures/figA2_diversity_grid.{pdf,png}"
echo "    results/figures/figA3_diversity_mixed.{pdf,png}"
echo "    results/figures/figC_cross_eval_heatmaps.{pdf,png}"
echo "  Supporting:"
echo "    results/data/seq_lengths_qm9.csv"
echo "    results/tables/seq_lengths_qm9.{md,tex}"
echo "    results/data/ece_by_token_type.csv"
echo "    results/tables/ece_by_token_type.{md,tex}"
