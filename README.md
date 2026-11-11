# Perturb-seq + Deep Learning: Predicting CRISPRi Guide Identity from Expression

**Author:** Syed Nurul Hasan, Ph.D.
**Contact:** nayanchoton1789@gmail.com | [ORCID: 0000-0002-4564-872X](https://orcid.org/0000-0002-4564-872X)

---

## Why this project

Two related CV gaps, closed together: **Perturb-seq** (CRISPR perturbation
with single-cell readout) and **deep learning (PyTorch/TensorFlow)** applied
to single-cell data. This repo trains a PyTorch classifier to predict which
CRISPRi guide a cell received from its transcriptional response alone — the
same modeling task underlying real Perturb-seq QC (confirming a guide
produced a detectable on-target effect) and perturbation-signature discovery.

## Pipeline

```
Adamson et al. 2016 Perturb-seq (K562, CRISPRi, UPR pathway targets)
        │
        ▼
1. preprocess.py  (scanpy)
   ├── Drop unassigned-guide cells
   ├── QC filter, normalize, log1p
   ├── Top 2000 HVGs, scale
   └── Export dense matrix + integer labels
        │
        ▼
2. train_classifier.py  (PyTorch)
   ├── MLP (2000 -> 256 -> 64 -> 8), dropout, class-weighted cross-entropy
   ├── 80/20 stratified train/test split
   └── Training curve + confusion matrix + per-class precision/recall/F1
```

## Dataset

One sample (`GSM2406675`, run `10X001`) from Adamson CH, Norman TM, Mount A,
et al. "A Multiplexed Single-Cell CRISPR Screening Platform Enables
Systematic Dissection of the Unfolded Protein Response." *Cell*.
2016;167(7):1867-1882.e21. — a CRISPRi Perturb-seq screen of UPR-pathway
transcription factors in K562 cells. Fetched as a harmonized `.h5ad` from
the [scPerturb](https://scperturb.org) project's Zenodo archive
(MD5-verified in `scripts/download_data.sh`), rather than reprocessed from
raw FASTQ — same "downstream of a public processed artifact" pattern as the
Cell Ranger disclosure in `cite-seq-wnn-trajectory`.

5,768 cells x 35,635 genes; after dropping 6 cells with no confidently
assigned guide, 8 perturbation groups remain: 7 targeted genes (**SPI1,
EP300, SNAI1, ZNF326, BHLHE40, CREB1, DDIT3**) plus one large group labeled
`62(mod)_pBA581` — by far the most abundant single guide identity (1,769/5,762
cells), consistent with a non-targeting or reference-guide population in
this design, though I haven't independently confirmed that assignment
against the original paper's supplementary tables, so I'm stating it as a
reasonable inference rather than a confirmed fact.

## Results (this run)

- Train/test split: 4,601 / 1,151 cells (stratified by guide identity).
- Majority-class baseline (always predict `62(mod)_pBA581`): **30.8%**.
- Model test accuracy after 40 epochs: **44.0%** — a clear lift over the
  majority-class baseline, i.e. the network is picking up real
  perturbation-specific transcriptional signal, not just learning class
  frequencies (class-weighted loss specifically guards against the latter).
- The training curve shows classic small-N overfitting — train loss drops to
  ~0.02 while test loss climbs back up after ~epoch 5 — expected with 2,000
  features and a few hundred cells per class; an early-stopping checkpoint
  around epoch 5 (test_acc ≈ 0.46) would likely generalize slightly better
  than the final epoch. Reported as observed rather than tuned away.
- Per-class performance is uneven and biologically legible: **SPI1**
  knockdown is the easiest to detect (0.84 precision / 0.91 recall) —
  consistent with SPI1 being a dominant myeloid master transcription factor
  whose loss produces a large, distinctive expression shift — while
  **CREB1** and **DDIT3** knockdowns are barely separable from background
  (~0.15 F1), consistent with more subtle or redundant regulatory roles.
  Full breakdown: [`results/metrics.json`](results/metrics.json),
  [`results/training_and_confusion.png`](results/training_and_confusion.png).

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_data.sh
python3 scripts/preprocess.py
python3 scripts/train_classifier.py
```

## Key Features

- Real, published Perturb-seq data (MD5-verified download)
- A genuine multi-class deep learning classification task on single-cell
  expression, not a toy dataset
- Class-weighted loss to handle real perturbation-group imbalance
- Baseline comparison (majority-class) so the accuracy number means something

## References

1. Adamson CH, Norman TM, Mount A, et al. A Multiplexed Single-Cell CRISPR
   Screening Platform Enables Systematic Dissection of the Unfolded Protein
   Response. *Cell*. 2016;167(7):1867-1882.e21.
2. Peidli S, Green TD, Shen C, et al. scPerturb: harmonized single-cell
   perturbation data. *Nature Methods*. 2024;21:531-540.

## License

MIT — see [LICENSE](LICENSE).

## Running on HPC (SLURM) and AWS

Includes `run_analysis.sh` (Docker or Apptainer), a SLURM job script, and a one-shot EC2 launcher. See [HPC_AWS.md](HPC_AWS.md) for steps and what was verified.
