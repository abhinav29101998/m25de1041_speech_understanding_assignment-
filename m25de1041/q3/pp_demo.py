import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech, compute_mel_spectrogram, pad_or_trim
from privacymodule import PrivacyTransformer, train_privacy

MAX_FRAMES = 200


def run_demo(local_root=None):
    os.makedirs("data/examples", exist_ok=True)

    print("Privacy Preservation Demo (LibriSpeech)")
    samples = get_librispeech(local_root=local_root, max_samples=200)

    male_samples = [s for s in samples if s.get("gender", "").lower() == "male"]
    female_samples = [s for s in samples if s.get("gender", "").lower() == "female"]

    print(f"Male samples   : {len(male_samples)}")
    print(f"Female samples : {len(female_samples)}")

    if not male_samples:
        print("No male samples found. Using fallback samples.")
        male_samples = samples[:5]

    model_path = "data/privacy_module.pt"
    model = PrivacyTransformer(dim=40, T=MAX_FRAMES)

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print("Training model...")
        result = train_privacy(samples, epochs=15, max_frames=MAX_FRAMES)
        if result is None:
            print("Training failed.")
            return
        model, _ = result

    model.eval()

    print("\nTransforming samples...\n")
    print(f"{'Speaker':<10} {'Transcript':<35} {'CosSim':>10} {'Shift':>10}")
    print("-" * 70)

    results = []

    for i, s in enumerate(male_samples[:3]):
        mel = compute_mel_spectrogram(s["audio"], s["sr"])
        mel = pad_or_trim(mel, MAX_FRAMES)

        x = torch.tensor(mel).unsqueeze(0)
        tgt = torch.tensor([1])  # target = female

        with torch.no_grad():
            transformed = model.transform(x, tgt)
            c_orig, _ = model.encode(x)
            c_trans, _ = model.encode(transformed)

        cos_sim = torch.nn.functional.cosine_similarity(c_orig, c_trans).item()
        feat_shift = (x - transformed).abs().mean().item()

        transcript = s.get("text", "N/A")[:33]

        print(f"{s['speaker_id']:<10} {transcript:<35} {cos_sim:>10.4f} {feat_shift:>10.4f}")

        results.append({
            "speaker_id": s["speaker_id"],
            "transcript": s.get("text", ""),
            "mel_orig": mel,
            "mel_trans": transformed[0].numpy(),
            "cos_sim": cos_sim,
            "feat_shift": feat_shift,
        })

        _save_pair_plot(
            mel,
            transformed[0].numpy(),
            s["speaker_id"],
            transcript,
            f"data/examples/pair_{i+1}_spk{s['speaker_id']}.png"
        )

    mean_sim = np.mean([r["cos_sim"] for r in results])
    mean_shift = np.mean([r["feat_shift"] for r in results])

    print("\nSummary:")
    print(f"Mean cosine similarity : {mean_sim:.4f}")
    print(f"Mean feature shift     : {mean_shift:.4f}")

    if mean_sim > 0.85:
        print("High content preservation")
    elif mean_sim > 0.60:
        print("Moderate preservation")
    else:
        print("Low preservation")

    _save_comparison_figure(results)

    print("\nOutputs saved in data/examples/")


def _save_pair_plot(mel_orig, mel_trans, spk_id, transcript, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    plots = [
        (mel_orig, f"Original (Spk {spk_id})"),
        (mel_trans, "Transformed"),
        (np.abs(mel_orig - mel_trans), "Difference")
    ]

    for ax, (feat, title) in zip(axes, plots):
        im = ax.imshow(feat.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'"{transcript[:50]}"')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_comparison_figure(results):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))

    if n == 1:
        axes = [axes]

    for row, r in enumerate(results):
        plots = [
            (r["mel_orig"], f"Original (Spk {r['speaker_id']})"),
            (r["mel_trans"], f"Transformed (sim={r['cos_sim']:.3f})"),
            (np.abs(r["mel_orig"] - r["mel_trans"]), "Difference")
        ]

        for col, (feat, title) in enumerate(plots):
            ax = axes[row][col]
            im = ax.imshow(feat.T, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Mel")
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("data/examples/pp_comparison.png", dpi=150)
    plt.close()
    print("Saved comparison figure")


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(local_root=local)