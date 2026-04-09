# AutoGraph: Linearization Uncertainty

Code for "Same Graph, Different Likelihoods: Calibration of Autoregressive Graph Generators via Permutation-Equivalent Encodings", *Towards Trustworthy Predictions* workshop at AISTATS 2026.

- Paper: [arXiv:2604.05613](https://arxiv.org/abs/2604.05613)
- Authors: Laurits Fredsgaard, Aaron Thomas, Michael Riis Andersen, Mikkel N. Schmidt, Mahito Sugiyama

## Overview

Autoregressive graph generators encode a graph as a token sequence and fit a next-token language model. A single graph has many valid linearizations, so the model assigns different likelihoods to equivalent encodings of the same graph. We measure this inconsistency with Linearization Uncertainty (LU), the coefficient of variation of per-graph NLL across random linearizations. Models trained on deterministic orderings tend to memorize the ordering rather than the graph; stochastic (`random_order`) training lowers LU and improves calibration. On QM9, LU predicts generated-sample quality better than the model's own likelihood.

The four strategies we compare are defined in `autograph/linearization.py`:

| Strategy           | Start bias | Jump bias  | Neighbor bias |
|--------------------|------------|------------|---------------|
| `random_order`     | random     | random     | random        |
| `max_degree_first` | max degree | max degree | max degree    |
| `min_degree_first` | min degree | min degree | min degree    |
| `anchor_expansion` | max degree | max degree | min degree    |

## Relation to upstream AutoGraph

This repository started from [BorgwardtLab/AutoGraph](https://github.com/BorgwardtLab/AutoGraph) and was scrubbed to the code needed for the paper:

- Only QM9 and Planar remain. Other datasets (Guacamol, MOSES, SBM, protein, point-cloud, NetworkX community graphs) were removed.
- The `mamba_ssm` and `orca` dependencies were dropped.
- Unused model and evaluation code was pruned.
- The four linearization strategies, LU computation, permutation and cross-strategy evaluation, and the analysis pipeline that produces the paper's tables and figures were added.

If you want to see how linearization bias is injected, `autograph/data/sent_utils.pyx` is a helpful starting point. It is the Cython SENT sampler and exposes the `start_bias`, `jump_bias`, and `neighbor_bias` knobs that control which node is chosen at each walk step. Even without prior Cython experience the file is fairly readable. `autograph/linearization.py` wires combinations of those knobs into the four named strategies.

We use gradient accumulation (`trainer.accumulate_grad_batches=8` in the default configs) so the effective batch size matches the one reported in the upstream AutoGraph paper while each step still fits on a single consumer GPU.

## Setup

Requirements:

- Python 3.13 or newer.
- A C compiler on `PATH`. `sent_utils` is Cython and is JIT-compiled via `pyximport` on first import; on Linux `gcc` or `clang` is enough.
- A CUDA GPU for training and generation.

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The dependency floors in `pyproject.toml` were tested with PyTorch 2.10, Lightning 2.6, and Transformers 4.57 on CUDA 12.8.

## Quick start

Train on QM9 with stochastic linearization:

```bash
.venv/bin/python train.py experiment=qm9 datamodule.linearization_strategy=random_order seed=0
```

Evaluate a checkpoint:

```bash
.venv/bin/python evaluate.py experiment=qm9 model.pretrained_path=/path/to/last.ckpt
```

QM9 and Planar are downloaded to `./datasets/` on first run.

## Reproducing paper results

Three phases: (A) training, (B) evaluation and generation, (C) aggregation into tables and figures. A and B need a GPU; C is CPU only. The SLURM launchers we used on our cluster are not included. The commands below are the underlying Python invocations.

### A. Training

QM9 (100k steps) and Planar (200k steps), 4 strategies × 3 seeds each:

```bash
for strategy in random_order max_degree_first min_degree_first anchor_expansion; do
  for seed in 0 1 2; do
    .venv/bin/python train.py experiment=qm9    datamodule.linearization_strategy=$strategy seed=$seed
    .venv/bin/python train.py experiment=planar datamodule.linearization_strategy=$strategy seed=$seed
  done
done
```

QM9 subset sweep (4 strategies × 3 sizes × 3 seeds, 20k steps each):

```bash
for strategy in random_order max_degree_first min_degree_first anchor_expansion; do
  for n in 128 1000 10000; do
    for seed in 0 1 2; do
      .venv/bin/python train.py experiment=qm9 \
        datamodule.linearization_strategy=$strategy \
        datamodule.subset_size=$n \
        trainer.max_steps=20000 \
        seed=$seed
    done
  done
done
```

Checkpoints are written under `logs/train/<dataset>/<strategy>/<model>/<seed>/runs/.../checkpoints/`.

### B. Evaluation

Each script takes `--checkpoint` pointing at a trained `.ckpt`. Re-run per (strategy, seed). See `--help` for all flags.

| Step | Script                                                                              | Purpose                                                        | Paper artefact     |
|:-----|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|:-------------------|
| B1   | `scripts/analysis/eval_invariance.py --all-eval-strategies --num_graphs -1`         | Per-graph NLL under random permutations on the full test set  | Table 2, cross-eval |
| B2   | same with `--dataset Planar`                                                        | Planar permutation eval                                        | Figure 1           |
| B3   | `scripts/analysis/eval_invariance.py --num_permutations 8`                          | Population-level LU at K=8                                     | Table 2 (LU, NLL)  |
| B4   | `scripts/analysis/eval_invariance.py --num_permutations 32`                         | Population-level LU at K=32                                    | Figure C heatmap   |
| B5   | `scripts/analysis/generate_molecules.py --num_samples 10000`                        | Sample 10k molecules per model                                 | Table 1a           |
| B6   | `scripts/analysis/eval_generated.py --num_permutations 32`                          | Score generated molecules under K=32 permutations              | Table 1b           |

### C. Aggregation

Once B is done:

```bash
bash scripts/analysis/run_aggregation.sh
```

Outputs land in `results/tables/` and `results/figures/`.

## Project structure

```
autograph/                    Core library
  linearization.py            The four strategies
  ece.py                      Expected Calibration Error
  mol.py                      Molecule / RDKit utilities
  lr_schedulers.py            LR schedules
  data/
    sent_utils.pyx            Cython SENT sampler with bias knobs
    sent_utils_wrapper.py     Python wrapper around the Cython sampler
    datasets.py               LightningDataModule for QM9 / Planar
    tokenizer.py              SENT token / graph conversion
    mol_dataset.py, spectre_dataset.py, batch_converter.py
  evaluation/
    metrics.py                VUN, FCD, PGD, molecule metrics
    visualization.py
  models/seq_models.py        Transformer sequence model (Llama family)
configs/                      Hydra configs
scripts/analysis/             Evaluation, aggregation, figures, tables
  eval_invariance.py
  generate_molecules.py
  eval_generated.py
  table*_*.py, fig*_*.py      Paper-numbered aggregators
  run_aggregation.sh
tests/
train.py
evaluate.py
```

## Citation

```bibtex
@inproceedings{fredsgaard2026same,
  title         = {Same Graph, Different Likelihoods: Calibration of Autoregressive Graph Generators via Permutation-Equivalent Encodings},
  author        = {Fredsgaard, Laurits and Thomas, Aaron and Andersen, Michael Riis and Schmidt, Mikkel N. and Sugiyama, Mahito},
  booktitle     = {Towards Trustworthy Predictions Workshop at AISTATS},
  year          = {2026},
  eprint        = {2604.05613},
  archivePrefix = {arXiv},
}
```

## License

BSD-3-Clause. This project is a derivative of [BorgwardtLab/AutoGraph](https://github.com/BorgwardtLab/AutoGraph), which is also BSD-3-Clause. See `LICENSE`.
