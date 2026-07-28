#!/usr/bin/env python3
"""QC + normalization + HVG selection on the Adamson et al. 2016 Perturb-seq
sample, then export a dense HVG expression matrix + integer perturbation
labels for the PyTorch classifier in train_classifier.py."""
import json

import numpy as np
import scanpy as sc

IN_PATH = "data/adamson2016_10X001.h5ad"
OUT_DIR = "results"

adata = sc.read_h5ad(IN_PATH)
print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Drop cells with no confidently-assigned guide -- '*' is this harmonized
# metadata's explicit placeholder, and a handful of cells are genuinely NaN.
adata = adata[~adata.obs["perturbation"].isna()].copy()
adata = adata[adata.obs["perturbation"] != "*"].copy()
print(f"After dropping unassigned-guide cells: {adata.shape[0]} cells")

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
print(f"After basic QC filtering: {adata.shape[0]} cells x {adata.shape[1]} genes")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
print(f"After HVG selection: {adata.shape[1]} genes")

sc.pp.scale(adata, max_value=10)

X = adata.X.astype(np.float32)
# .astype(str) on a pandas Categorical can leave NaN entries as float NaN
# rather than the string "nan" -- convert element-by-element instead.
labels = np.array([str(v) for v in adata.obs["perturbation"].values])

classes = sorted(set(labels))
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[label] for label in labels], dtype=np.int64)

print("\nClass distribution:")
for c in classes:
    print(f"  {c}: {int((y == class_to_idx[c]).sum())}")

np.savez(f"{OUT_DIR}/features.npz", X=X, y=y)
with open(f"{OUT_DIR}/classes.json", "w") as f:
    json.dump(classes, f, indent=2)

print(f"\nSaved {OUT_DIR}/features.npz (X: {X.shape}, y: {y.shape}) and {OUT_DIR}/classes.json")
