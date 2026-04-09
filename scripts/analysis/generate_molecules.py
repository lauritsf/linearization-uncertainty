"""Generate molecules from a trained model and save token sequences + quality metrics.

Outputs (in --output_dir):
  - generated_tokens.pt   — list of variable-length token tensors (raw model output)
  - generated_summary.csv — per-molecule quality metrics + SMILES
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Allow sibling imports (eval_utils)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_utils import extract_strategy_and_seed, load_model

from autograph.data import GraphDataset
from autograph.evaluation.metrics import MoleculeMetrics
from autograph.mol import build_molecule_with_partial_charges, mol2smiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="QM9")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)
    train_strategy, seed = extract_strategy_and_seed(args.checkpoint)
    print(f"Model: {train_strategy}/seed_{seed} on {device}")

    # Optimize batch size for generation (tested: batch_size=64 → 26 samples/sec)
    model.cfg.sampling.batch_size = 64

    # Generate with return_sent=True to get raw token sequences
    print(f"Generating {args.num_samples} molecules...")
    raw_sequences, avg_time = model.generate(
        num_samples=args.num_samples,
        return_sent=True,
        max_length=args.max_length,
    )
    print(f"Generated {len(raw_sequences)} sequences (avg {avg_time:.4f}s/sample)")

    # Save raw token tensors
    tokens_path = str(Path(args.output_dir) / "generated_tokens.pt")
    torch.save(raw_sequences, tokens_path)
    print(f"Saved token tensors to {tokens_path}")

    # Decode to graphs and compute quality metrics
    print("Decoding graphs and computing quality metrics...")
    dm = GraphDataset(
        root="./datasets",
        dataset_names=args.dataset,
        labeled_graph=(args.dataset == "QM9"),
        init_tokenizer=False,
    )
    dm.prepare_data()
    dm.setup("fit")  # Setup both train and test
    dm.setup("test")  # Also setup test explicitly for MoleculeMetrics compatibility

    # Decode token sequences to graph objects
    decoded_graphs = []
    for seq in tqdm(raw_sequences, desc="Decoding"):
        try:
            graph = model.tokenizer.decode(seq)
            decoded_graphs.append(graph)
        except Exception:
            decoded_graphs.append(None)

    # Convert to SMILES and build summary
    atom_decoder = getattr(dm, "atom_decoder", None)
    rows = []
    for i, (seq, graph) in enumerate(zip(raw_sequences, decoded_graphs, strict=True)):
        seq_len = len(seq) if isinstance(seq, torch.Tensor) else 0
        smiles = ""
        valid = False

        if graph is not None and atom_decoder is not None:
            try:
                mol = build_molecule_with_partial_charges(graph, atom_decoder)
                s = mol2smiles(mol)
                if s:
                    smiles = s
                    valid = True
            except Exception:
                pass

        rows.append({
            "idx": i,
            "seq_len": seq_len,
            "smiles": smiles,
            "valid": valid,
        })

    # Compute quality metrics
    valid_graphs = [g for g in decoded_graphs if g is not None]
    print(f"Valid graphs: {len(valid_graphs)}/{len(decoded_graphs)}")
    print("Computing generation quality metrics...")

    quality = {}
    if len(valid_graphs) >= 5:
        try:
            mol_metrics = MoleculeMetrics(dm)
            quality = mol_metrics(valid_graphs, split="test", fast=False)
            print(f"  Validity: {quality.get('validity', 'N/A'):.4f}")
            print(f"  Uniqueness: {quality.get('uniqueness', 'N/A'):.4f}")
            print(f"  Novelty: {quality.get('novelty', 'N/A'):.4f}")
            print(f"  FCD: {quality.get('test_fcd', 'N/A'):.4f}")
        except Exception as e:
            print(f"  WARNING: Quality metrics computation failed: {e}")
            quality = {}
    else:
        print(f"  WARNING: Too few valid graphs ({len(valid_graphs)}) for quality metrics")

    # Save summary CSV
    import csv
    summary_path = str(Path(args.output_dir) / "generated_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "seq_len", "smiles", "valid"])
        writer.writeheader()
        writer.writerows(rows)

    # Save aggregate quality metrics
    quality_path = str(Path(args.output_dir) / "generated_quality.csv")
    with open(quality_path, "w", newline="") as f:
        metric_keys = [k for k, v in quality.items()
                       if isinstance(v, (int, float)) and k != "error"]
        writer = csv.DictWriter(f, fieldnames=["strategy", "seed"] + metric_keys)
        writer.writeheader()
        row = {"strategy": train_strategy, "seed": seed}
        row.update({k: quality[k] for k in metric_keys})
        writer.writerow(row)

    print(f"Saved summary to {summary_path}")
    print(f"Saved quality metrics to {quality_path}")


if __name__ == "__main__":
    main()
