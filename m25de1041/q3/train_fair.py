import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech, compute_mel_spectrogram, pad_or_trim


# Dataset with text + gender
class FairASRDataset(Dataset):
    GENDER_MAP = {"male": 0, "female": 1, "unknown": 2}

    def __init__(self, samples, max_frames=200, max_len=50):
        chars = set()
        for s in samples:
            chars.update(s.get("text", "").upper())

        self.vocab = {c: i+1 for i, c in enumerate(sorted(chars))}
        self.vocab["<pad>"] = 0

        self.items = []

        for s in samples:
            g = s.get("gender", "unknown").lower()
            g = g if g in self.GENDER_MAP else "unknown"

            mel = compute_mel_spectrogram(s["audio"], s["sr"])
            mel = pad_or_trim(mel, max_frames)

            text = [self.vocab.get(c, 0) for c in s.get("text", "").upper()[:max_len]]
            text += [0] * (max_len - len(text))

            self.items.append((
                torch.tensor(mel),
                torch.tensor(text),
                torch.tensor(self.GENDER_MAP[g])
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# Simple ASR model
class ASRModel(nn.Module):
    def __init__(self, input_dim=40, hidden=256, vocab_size=100):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


# Standard loss
def standard_loss(logits, targets):
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=0
    )


# Fairness-aware loss
def fair_loss(logits, targets, groups, lam=0.5):
    B, T, V = logits.shape

    losses = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=0,
        reduction="none"
    ).reshape(B, -1).mean(-1)

    std = losses.mean()

    group_losses = []
    for g in range(3):
        mask = (groups == g)
        if mask.sum() > 0:
            group_losses.append(losses[mask].mean())

    if len(group_losses) < 2:
        return std, std, 0.0

    gap = max(group_losses) - min(group_losses)
    return std + lam * gap, std, gap


# Train model
def train_one(dataset, use_fairness=False, epochs=10):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ASRModel(vocab_size=len(dataset.vocab))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()

        for feats, tgts, groups in loader:
            logits = model(feats)

            if use_fairness:
                loss, _, _ = fair_loss(logits, tgts, groups)
            else:
                loss = standard_loss(logits, tgts)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1} done")

    return model


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    samples = get_librispeech(local_root=local, max_samples=400)
    dataset = FairASRDataset(samples)

    print("Training normal model")
    train_one(dataset, use_fairness=False)

    print("Training fair model")
    train_one(dataset, use_fairness=True)