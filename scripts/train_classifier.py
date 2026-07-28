#!/usr/bin/env python3
"""PyTorch MLP classifying which CRISPRi guide a cell received, from its
HVG expression profile alone -- applying deep learning to Perturb-seq data
to recover perturbation identity from the transcriptional response it
causes (the modeling task behind on-target/off-target QC and perturbation
signature discovery in real Perturb-seq analysis)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)

RESULTS_DIR = "results"

data = np.load(f"{RESULTS_DIR}/features.npz")
X, y = data["X"], data["y"]
with open(f"{RESULTS_DIR}/classes.json") as f:
    classes = json.load(f)

n_classes = len(classes)
print(f"X: {X.shape}, y: {y.shape}, {n_classes} classes: {classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

class_counts = np.bincount(y_train, minlength=n_classes)
class_weights = torch.tensor(class_counts.sum() / (n_classes * class_counts), dtype=torch.float32)
print("Class weights (inverse frequency):", dict(zip(classes, class_weights.tolist())))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)


class PerturbationClassifier(nn.Module):
    def __init__(self, n_genes, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


model = PerturbationClassifier(X.shape[1], n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

n_epochs = 40
history = {"train_loss": [], "test_loss": [], "test_acc": []}

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_ds)

    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        test_loss = criterion(test_out, y_test_t).item()
        test_pred = test_out.argmax(dim=1)
        test_acc = (test_pred == y_test_t).float().mean().item()

    history["train_loss"].append(epoch_loss)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"epoch {epoch + 1:3d}/{n_epochs}  train_loss={epoch_loss:.4f}  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

model.eval()
with torch.no_grad():
    final_pred = model(X_test_t).argmax(dim=1).numpy()

report = classification_report(y_test, final_pred, target_names=classes, output_dict=True, zero_division=0)
print("\nPer-class performance:")
print(classification_report(y_test, final_pred, target_names=classes, zero_division=0))

n_control = class_counts.max()
control_class = classes[int(np.argmax(class_counts))]
baseline_acc = n_control / len(y_train)
print(f"Majority-class baseline (always predict '{control_class}'): {baseline_acc:.4f}")
print(f"Model test accuracy: {history['test_acc'][-1]:.4f}")

with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
    json.dump({
        "n_epochs": n_epochs,
        "final_test_acc": history["test_acc"][-1],
        "majority_class_baseline_acc": baseline_acc,
        "majority_class": control_class,
        "classification_report": report,
    }, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["test_loss"], label="test")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("cross-entropy loss")
axes[0].set_title("Training curve")
axes[0].legend()

cm = confusion_matrix(y_test, final_pred, labels=range(n_classes))
im = axes[1].imshow(cm, cmap="viridis")
axes[1].set_xticks(range(n_classes))
axes[1].set_yticks(range(n_classes))
axes[1].set_xticklabels(classes, rotation=90, fontsize=7)
axes[1].set_yticklabels(classes, fontsize=7)
axes[1].set_xlabel("predicted")
axes[1].set_ylabel("true")
axes[1].set_title(f"Confusion matrix (test acc={history['test_acc'][-1]:.2f})")
fig.colorbar(im, ax=axes[1], fraction=0.046)

fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/training_and_confusion.png", dpi=150)
print(f"\nSaved {RESULTS_DIR}/metrics.json and {RESULTS_DIR}/training_and_confusion.png")
