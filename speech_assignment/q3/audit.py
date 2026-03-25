import sys
import os
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech


# Load samples and normalize gender labels
def load_with_gender(local, max_samples=1000):
    samples = get_librispeech(local_root=local, max_samples=max_samples)
    for s in samples:
        g = s.get("gender", "unknown").lower()
        if g == "male":
            s["gender"] = "male"
        elif g == "female":
            s["gender"] = "female"
        else:
            s["gender"] = "unknown"
    return samples


# Count samples per gender
def representation(samples):
    counts = defaultdict(int)
    for s in samples:
        counts[s["gender"]] += 1

    total = len(samples)
    print("\nGender distribution:")
    for g, c in sorted(counts.items()):
        print(f"{g:<10}: {c} ({c/total*100:.1f}%)")

    return dict(counts)


# Average duration per gender
def duration_by_gender(samples):
    durs = defaultdict(list)

    for s in samples:
        dur = len(s["audio"]) / s["sr"]
        durs[s["gender"]].append(dur)

    stats = {}
    for g, vals in durs.items():
        stats[g] = {
            "mean": np.mean(vals),
            "std": np.std(vals)
        }

    return stats


# Proxy for audio difficulty (energy variation)
def quality_proxy(samples):
    scores = defaultdict(list)

    for s in samples:
        audio = s["audio"].astype(np.float32)
        fl = int(s["sr"] * 0.025)

        rms = [
            np.sqrt(np.mean(audio[i:i+fl]**2) + 1e-10)
            for i in range(0, len(audio)-fl, fl)
        ]

        cov = np.std(rms) / (np.mean(rms) + 1e-10)
        scores[s["gender"]].append(cov)

    return {g: np.mean(v) for g, v in scores.items()}


# Count unique speakers
def speaker_stats(samples):
    spk = defaultdict(set)

    for s in samples:
        spk[s["gender"]].add(s["speaker_id"])

    return {g: len(ids) for g, ids in spk.items()}


# Gap between groups
def fairness_gap(values):
    vals = list(values.values())
    return max(vals) - min(vals) if len(vals) > 1 else 0.0


# Simple bias score
def doc_debt(rep, quality):
    counts = list(rep.values())
    imbalance = max(counts) / max(min(counts), 1) - 1.0
    gap = fairness_gap(quality)

    return min(imbalance * 2 + gap * 5, 10.0)


# Plot summary
def plot_audit(rep, dur_stats, quality, spk_stats, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    genders = sorted(rep.keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0,0].bar(genders, [rep[g] for g in genders])
    axes[0,0].set_title("Count")

    axes[0,1].bar(genders, [spk_stats.get(g, 0) for g in genders])
    axes[0,1].set_title("Speakers")

    axes[1,0].bar(genders, [dur_stats[g]["mean"] for g in genders])
    axes[1,0].set_title("Duration")

    axes[1,1].bar(genders, [quality[g] for g in genders])
    axes[1,1].set_title("Difficulty")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("Saved:", save_path)


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    samples = load_with_gender(local, max_samples=500)

    rep = representation(samples)
    dur = duration_by_gender(samples)
    quality = quality_proxy(samples)
    spk = speaker_stats(samples)

    gap = fairness_gap(quality)
    debt = doc_debt(rep, quality)

    print("\nFairness gap:", gap)
    print("Doc debt:", debt)

    plot_audit(rep, dur, quality, spk, "data/audit.png")

    with open("data/audit_results.json", "w") as f:
        json.dump({
            "representation": rep,
            "fairness_gap": gap,
            "doc_debt": debt
        }, f, indent=2)