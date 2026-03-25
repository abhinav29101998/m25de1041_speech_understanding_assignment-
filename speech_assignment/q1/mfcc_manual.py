import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Import dataset loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech


# Pre-emphasis to boost high frequencies
def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


# Split signal into overlapping frames
def frame_signal(signal, sr, frame_ms=25, hop_ms=10):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    num_frames = 1 + (len(signal) - frame_len) // hop_len

    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, frame_len),
        strides=(signal.strides[0] * hop_len, signal.strides[0])
    ).copy()

    return frames, frame_len


# Apply window (default: Hamming)
def apply_window(frames, window_type="hamming"):
    N = frames.shape[1]

    if window_type == "hamming":
        w = np.hamming(N)
    elif window_type == "hanning":
        w = np.hanning(N)
    else:
        w = np.ones(N)

    return frames * w


# Compute power spectrum
def compute_power_spectrum(frames, NFFT=512):
    return (1.0 / NFFT) * np.abs(np.fft.rfft(frames, n=NFFT)) ** 2


# Convert between Hz and Mel scale
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


# Create Mel filter bank
def mel_filterbank(sr, NFFT=512, num_filters=26, low_hz=0, high_hz=None):
    if high_hz is None:
        high_hz = sr / 2

    mel_pts = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), num_filters + 2)
    hz_pts = mel_to_hz(mel_pts)

    bins = np.floor((NFFT + 1) * hz_pts / sr).astype(int)
    filters = np.zeros((num_filters, NFFT // 2 + 1))

    for m in range(1, num_filters + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]

        filters[m - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        filters[m - 1, center:right] = (right - np.arange(center, right)) / max(right - center, 1)

    return filters


# Log compression
def log_compress(energy):
    return np.log(energy + 1e-10)


# DCT to get MFCC coefficients
def apply_dct(log_energy, num_ceps=13):
    N = log_energy.shape[1]
    n = np.arange(N)

    coeffs = []
    for k in range(num_ceps):
        c = np.sum(log_energy * np.cos(np.pi * k / N * (n + 0.5)), axis=1)
        coeffs.append(c)

    return np.stack(coeffs, axis=1)


# Complete MFCC pipeline
def extract_mfcc(audio, sr, num_ceps=13, num_filters=26,
                 NFFT=512, frame_ms=25, hop_ms=10, window_type="hamming"):

    audio = audio.astype(np.float64)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    signal = pre_emphasis(audio)
    frames, _ = frame_signal(signal, sr, frame_ms, hop_ms)

    windowed = apply_window(frames, window_type)
    power = compute_power_spectrum(windowed, NFFT)

    fbank = mel_filterbank(sr, NFFT, num_filters)
    log_energy = log_compress(power @ fbank.T)

    return apply_dct(log_energy, num_ceps)


# Plot MFCCs for multiple samples
def plot_mfcc_grid(mfcc_list, titles, save_path="data/mfcc_plot.png"):
    n = len(mfcc_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, mfcc, title in zip(axes, mfcc_list, titles):
        im = ax.imshow(mfcc.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Frame")
        ax.set_ylabel("MFCC")

        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("Saved:", save_path)


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    os.makedirs("data", exist_ok=True)

    samples = get_librispeech(local_root=local, max_samples=5)

    mfcc_list = []
    titles = []

    for s in samples:
        mfcc = extract_mfcc(s["audio"], s["sr"])
        mfcc_list.append(mfcc)

        duration = len(s["audio"]) / s["sr"]
        titles.append(f"Spk {s['speaker_id']} ({duration:.1f}s)")

        print(f"Speaker {s['speaker_id']} | {duration:.2f}s | {mfcc.shape}")

    plot_mfcc_grid(mfcc_list, titles)

    np.save("data/example_mfcc.npy", mfcc_list[0])
    print("Saved: data/example_mfcc.npy")