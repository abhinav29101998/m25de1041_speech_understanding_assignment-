import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech
from train import DisentangledModel, LibriSpeakerDataset, CONFIG, train


# Simple baseline model (no disentanglement)
class BaselineModel(nn.Module):
    def __init__(self, input_dim=40, num_spk=251):
        super().__init__()
        self.rnn = nn.GRU(input_dim, 256, num_layers=2,
                          batch_first=True, bidirectional=True)
        self.proj = nn.Linear(512, 128)
        self.clf = nn.Linear(128, num_spk)

    def forward(self, x):
        out, _ = self.rnn(x)
        z = F.normalize(self.proj(out.mean(1)), dim=-1)
        return self.clf(z), z, None, None


def train_baseline(dataset, config):
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    model = BaselineModel(num_spk=dataset.num_speakers)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()

        for feats, spk_ids, _ in loader:
            logits, *_ = model(feats)
            loss = F.cross_entropy(logits, spk_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# Accuracy
def accuracy(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for feats, spk_ids, _ in loader:
            logits, *_ = model(feats)
            pred = logits.argmax(-1)

            correct += (pred == spk_ids).sum().item()
            total += spk_ids.size(0)

    return correct / total


# Equal Error Rate
def eer(model, dataset):
    loader = DataLoader(dataset, batch_size=64)

    embeddings = []
    labels = []

    with torch.no_grad():
        for feats, spk_ids, _ in loader:
            _, z, *_ = model(feats)
            if z is None:
                continue
            embeddings.append(z.numpy())
            labels.append(spk_ids.numpy())

    if not embeddings:
        return float("nan")

    Z = np.concatenate(embeddings)
    L = np.concatenate(labels)

    pairs = np.random.choice(len(Z), (3000, 2), replace=False)

    scores = [np.dot(Z[i], Z[j]) for i, j in pairs]
    same = [int(L[i] == L[j]) for i, j in pairs]

    fpr, tpr, _ = roc_curve(same, scores)
    fnr = 1 - tpr

    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2 * 100


# Measure correlation between zs and ze
def disent_score(model, loader):
    model.eval()
    zs_list, ze_list = [], []

    with torch.no_grad():
        for feats, _, _ in loader:
            _, zs, ze, _ = model(feats)
            if ze is None:
                return float("nan")
            zs_list.append(zs.numpy())
            ze_list.append(ze.numpy())

    zs = np.concatenate(zs_list)
    ze = np.concatenate(ze_list)

    corr = np.corrcoef(zs.T, ze.T)
    dim = zs.shape[1]

    return float(np.mean(np.abs(corr[:dim, dim:])))


# Print results
def print_table(results):
    print("\nResults:")
    print(f"{'Model':<20} {'Acc':>8} {'EER':>8} {'Disent':>10}")

    for name, val in results.items():
        acc = val.get("accuracy", 0) * 100
        eer_val = val.get("eer", 0)
        dis = val.get("disentanglement", 0)

        print(f"{name:<20} {acc:>7.2f} {eer_val:>8.2f} {dis:>10.4f}")


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    samples = get_librispeech(local_root=local, max_samples=500)

    dataset = LibriSpeakerDataset(samples, max_frames=CONFIG["max_frames"])
    loader = DataLoader(dataset, batch_size=64)

    results = {}

    print("Training baseline...")
    baseline = train_baseline(dataset, CONFIG)
    results["Baseline"] = {
        "accuracy": accuracy(baseline, loader),
        "eer": eer(baseline, dataset)
    }

    print("Training disentangled model...")
    model, _, _ = train(CONFIG, samples)
    results["Disentangled"] = {
        "accuracy": accuracy(model, loader),
        "eer": eer(model, dataset),
        "disentanglement": disent_score(model, loader)
    }

    print_table(results)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved: results.json")