import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech


# Compute real cepstrum
def cepstrum(frame, NFFT=512):
    spectrum = np.fft.rfft(frame, n=NFFT)
    log_spec = np.log(np.abs(spectrum) + 1e-10)
    return np.fft.irfft(log_spec)


# Decide voiced/unvoiced using cepstral peak
def is_voiced(cep, sr, low_hz=60, high_hz=400):
    low = int(sr / high_hz)
    high = min(int(sr / low_hz), len(cep) // 2)

    if low >= high:
        return False, 0.0

    region = np.abs(cep[low:high])
    strength = region.max() / (np.mean(np.abs(cep)) + 1e-10)

    return strength > 3.5, float(strength)


# Detect voiced/unvoiced segments
def detect_boundaries(audio, sr, frame_ms=25, hop_ms=10, NFFT=512):
    audio = audio.astype(np.float64)

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    labels = []
    strengths = []

    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start:start + frame_len] * np.hamming(frame_len)

        cep = cepstrum(frame, NFFT)
        voiced, strength = is_voiced(cep, sr)

        labels.append(voiced)
        strengths.append(strength)

    labels = np.array(labels)

    # Smooth decisions
    smooth = np.convolve(labels.astype(float), np.ones(7)/7, mode="same") > 0.5

    segments = []
    current = smooth[0]
    t0 = 0.0

    for i, val in enumerate(smooth):
        if val != current:
            segments.append((t0, i * hop_ms / 1000,
                             "voiced" if current else "unvoiced"))
            t0 = i * hop_ms / 1000
            current = val

    segments.append((t0, len(smooth) * hop_ms / 1000,
                     "voiced" if current else "unvoiced"))

    return segments, smooth, np.array(strengths)


# Plot waveform + decisions
def visualize(audio, sr, segments, smooth, strengths, hop_ms, save_path):
    t_audio = np.arange(len(audio)) / sr
    t_frames = np.arange(len(smooth)) * hop_ms / 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_audio, audio, lw=0.5)
    axes[0].set_title("Waveform")

    # Highlight segments
    for ax in axes[:2]:
        for s, e, lbl in segments:
            color = "green" if lbl == "voiced" else "red"
            ax.axvspan(s, e, alpha=0.15, color=color)

    axes[1].plot(t_frames, strengths)
    axes[1].axhline(3.5, linestyle="--")
    axes[1].set_title("Voiced strength")

    axes[2].fill_between(t_frames, smooth.astype(int), alpha=0.7)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Unvoiced", "Voiced"])
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("Saved:", save_path)


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    samples = get_librispeech(local_root=local, max_samples=3)
    os.makedirs("data", exist_ok=True)

    for i, s in enumerate(samples):
        print(f"\nSample {i+1} | Speaker {s['speaker_id']}")

        segments, smooth, strengths = detect_boundaries(s["audio"], s["sr"])

        print(f"{'Start':>8} {'End':>8} {'Label':>10}")
        for seg in segments:
            print(f"{seg[0]:>8.3f} {seg[1]:>8.3f} {seg[2]:>10}")

        visualize(
            s["audio"],
            s["sr"],
            segments,
            smooth,
            strengths,
            hop_ms=10,
            save_path=f"data/voiced_unvoiced_{i}.png"
        )