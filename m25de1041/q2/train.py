import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech, compute_mel_spectrogram, pad_or_trim


# Dataset for speaker + environment labels
class LibriSpeakerDataset(Dataset):
    def __init__(self, samples, max_frames=200, num_envs=3, seed=42):
        random.seed(seed)

        speakers = sorted(set(s["speaker_id"] for s in samples))
        self.spk2idx = {s: i for i, s in enumerate(speakers)}
        self.num_speakers = len(speakers)
        self.max_frames = max_frames

        self.items = []
        for s in samples:
            self.items.append({
                "audio": s["audio"],
                "sr": s["sr"],
                "speaker_id": self.spk2idx[s["speaker_id"]],
                "env_id": random.randint(0, num_envs - 1)
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mel = compute_mel_spectrogram(item["audio"], item["sr"])
        mel = pad_or_trim(mel, self.max_frames)

        return (
            torch.tensor(mel),
            torch.tensor(item["speaker_id"], dtype=torch.long),
            torch.tensor(item["env_id"], dtype=torch.long)
        )


# Speaker encoder
class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden=256, embed_dim=128):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, num_layers=2,
                          batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, embed_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        pooled = out.mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)


# Environment encoder
class EnvEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden=64, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim)
        )

    def forward(self, x):
        return self.net(x.mean(dim=1))


# Decoder for reconstruction
class Decoder(nn.Module):
    def __init__(self, spk_dim=128, env_dim=32, out_dim=40, seq_len=200):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(spk_dim + env_dim, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * out_dim)
        )
        self.out_dim = out_dim

    def forward(self, zs, ze):
        combined = torch.cat([zs, ze], dim=-1)
        return self.net(combined).view(-1, self.seq_len, self.out_dim)


# Full model
class DisentangledModel(nn.Module):
    def __init__(self, input_dim=40, num_spk=251, seq_len=200):
        super().__init__()
        self.spk_enc = SpeakerEncoder(input_dim)
        self.env_enc = EnvEncoder(input_dim)
        self.decoder = Decoder(seq_len=seq_len)
        self.clf = nn.Linear(128, num_spk)

    def forward(self, x):
        zs = self.spk_enc(x)
        ze = self.env_enc(x)
        recon = self.decoder(zs, ze)
        logits = self.clf(zs)
        return logits, zs, ze, recon


# Keep speaker and environment embeddings independent
def disentanglement_loss(zs, ze):
    zs = zs - zs.mean(0, keepdim=True)
    ze = ze - ze.mean(0, keepdim=True)
    corr = (zs.T @ ze) / zs.shape[0]
    return (corr ** 2).sum()


# Contrastive loss for same-speaker consistency
def contrastive_invariance_loss(zs, spk_ids, temp=0.07):
    sim = F.cosine_similarity(zs.unsqueeze(1), zs.unsqueeze(0), dim=-1) / temp

    labels = (spk_ids.unsqueeze(1) == spk_ids.unsqueeze(0)).float()
    mask = 1 - torch.eye(zs.size(0), device=zs.device)
    labels = labels * mask

    exp_sim = torch.exp(sim) * mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-10)

    loss = -(labels * log_prob).sum(1) / (labels.sum(1) + 1e-10)
    return loss.mean()


# Training loop
def train(config, samples):
    os.makedirs("results", exist_ok=True)

    dataset = LibriSpeakerDataset(samples,
                                 max_frames=config["max_frames"],
                                 num_envs=config["num_envs"])

    loader = DataLoader(dataset, batch_size=config["batch_size"],
                        shuffle=True, drop_last=True)

    model = DisentangledModel(num_spk=dataset.num_speakers,
                              seq_len=config["max_frames"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()

        for feats, spk_ids, env_ids in loader:
            logits, zs, ze, recon = model(feats)

            loss_cls = F.cross_entropy(logits, spk_ids)
            loss_recon = F.mse_loss(recon, feats)
            loss_dis = disentanglement_loss(zs, ze)
            loss_cont = contrastive_invariance_loss(zs, spk_ids)

            total_loss = (
                config["w_cls"] * loss_cls +
                config["w_recon"] * loss_recon +
                config["w_dis"] * loss_dis +
                config["w_cont"] * loss_cont
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config['epochs']} done")

    torch.save(model.state_dict(), "results/model.pt")
    print("Saved: results/model.pt")

    return model, dataset


CONFIG = {
    "max_frames": 200,
    "num_envs": 3,
    "batch_size": 32,
    "epochs": 15,
    "lr": 1e-3,
    "w_cls": 1.0,
    "w_recon": 0.3,
    "w_dis": 0.05,
    "w_cont": 0.2
}


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    samples = get_librispeech(local_root=local, max_samples=500)
    print("Loaded samples:", len(samples))

    train(CONFIG, samples)