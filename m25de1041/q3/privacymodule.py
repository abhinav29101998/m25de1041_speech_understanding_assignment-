import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech, compute_mel_spectrogram, pad_or_trim


# Dataset with gender labels
class PrivacyDataset(Dataset):
    def __init__(self, samples, max_frames=200):
        self.items = []

        for s in samples:
            g = s.get("gender", "unknown").lower()
            if g not in ("male", "female"):
                continue

            mel = compute_mel_spectrogram(s["audio"], s["sr"])
            mel = pad_or_trim(mel, max_frames)

            label = 0 if g == "male" else 1
            self.items.append((torch.tensor(mel), torch.tensor(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# Model
class PrivacyTransformer(nn.Module):
    def __init__(self, dim=40, T=200):
        super().__init__()

        self.content_enc = nn.GRU(dim, 128, batch_first=True)
        self.attr_enc = nn.Linear(dim, 16)

        self.decoder = nn.Sequential(
            nn.Linear(128 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, T * dim)
        )

        self.gender_clf = nn.Linear(16, 2)
        self.proto = nn.Embedding(2, 16)

        self.T = T
        self.dim = dim

    def encode(self, x):
        _, h = self.content_enc(x)
        content = h[-1]
        attr = self.attr_enc(x.mean(1))
        return content, attr

    def forward(self, x):
        c, a = self.encode(x)
        recon = self.decoder(torch.cat([c, a], dim=-1))
        recon = recon.view(-1, self.T, self.dim)
        g_logits = self.gender_clf(a)
        return recon, c, a, g_logits

    def transform(self, x, target_gender):
        c, _ = self.encode(x)
        a = self.proto(target_gender)
        out = self.decoder(torch.cat([c, a], dim=-1))
        return out.view(-1, self.T, self.dim)


# Train model
def train_privacy(samples, epochs=10):
    dataset = PrivacyDataset(samples)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = PrivacyTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()

        for feats, labels in loader:
            recon, _, attrs, logits = model(feats)

            loss_recon = F.mse_loss(recon, feats)
            loss_attr = F.cross_entropy(logits, labels)

            loss = loss_recon + 0.5 * loss_attr

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1} done")

    torch.save(model.state_dict(), "data/privacy_model.pt")
    return model, dataset


# Demo transformation
def demo(model, dataset):
    for feat, label in dataset:
        if label.item() == 0:
            x = feat.unsqueeze(0)
            break

    target = torch.tensor([1])

    with torch.no_grad():
        transformed = model.transform(x, target)

    diff = (x - transformed).abs().mean().item()
    print("Feature change:", diff)


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    samples = get_librispeech(local_root=local, max_samples=300)

    model, dataset = train_privacy(samples)
    demo(model, dataset)