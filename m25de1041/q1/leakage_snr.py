import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to import custom loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech


# Create window based on type
def make_window(name, N):
    if name == "rectangular":
        return np.ones(N)
    elif name == "hamming":
        return np.hamming(N)
    elif name == "hanning":
        return np.hanning(N)


# Calculate spectral leakage = sidelobe energy / total energy
def spectral_leakage(chunk, window_name, NFFT=512):
    N = len(chunk)
    w = make_window(window_name, N)

    # Apply window and compute power spectrum
    spectrum = np.abs(np.fft.rfft(chunk * w, n=NFFT)) ** 2

    # Find main lobe region around peak
    peak = np.argmax(spectrum)
    width = max(3, NFFT // 64)

    left = max(0, peak - width)
    right = min(len(spectrum), peak + width + 1)

    main_energy = spectrum[left:right].sum()

    # Everything else is considered leakage
    mask = np.ones(len(spectrum), dtype=bool)
    mask[left:right] = False
    side_energy = spectrum[mask].sum()

    leakage_ratio = side_energy / (main_energy + side_energy + 1e-10)
    return leakage_ratio, spectrum


# Estimate SNR using top spectrum values as signal and median as noise
def compute_snr(chunk, window_name, NFFT=512):
    N = len(chunk)
    w = make_window(window_name, N)

    spectrum = np.abs(np.fft.rfft(chunk * w, n=NFFT)) ** 2

    # Signal power: top 10% of spectral bins
    sorted_power = np.sort(spectrum)[::-1]
    signal_power = np.mean(sorted_power[:max(1, len(spectrum)//10)])

    # Noise floor: median of spectrum
    noise_power = np.median(spectrum) + 1e-10

    snr = 10 * np.log10(signal_power / noise_power)
    return snr, spectrum


# Run analysis across multiple audio samples
def analyze(samples, frame_ms=25, NFFT=512):
    windows = ["rectangular", "hamming", "hanning"]
    results = {w: {"leakage": [], "snr": []} for w in windows}

    for s in samples:
        audio = s["audio"].astype(np.float64)
        sr = s["sr"]

        # Take a short frame from middle of audio
        frame_len = int(sr * frame_ms / 1000)
        mid = len(audio) // 2

        chunk = audio[mid:mid + frame_len]
        if len(chunk) < frame_len:
            chunk = np.pad(chunk, (0, frame_len - len(chunk)))

        # Compute metrics for each window
        for w in windows:
            lk, _ = spectral_leakage(chunk, w, NFFT)
            sn, _ = compute_snr(chunk, w, NFFT)

            results[w]["leakage"].append(lk)
            results[w]["snr"].append(sn)

    # Average results over all samples
    final = {}
    for w in windows:
        final[w] = {
            "leakage": np.mean(results[w]["leakage"]),
            "snr": np.mean(results[w]["snr"])
        }

    return final


# Plot spectra and print summary table
def plot_and_print(results, sample, NFFT=512):
    windows = list(results.keys())

    sr = sample["sr"]
    audio = sample["audio"].astype(np.float64)

    # Use one sample chunk for visualization
    frame_len = int(sr * 25 / 1000)
    mid = len(audio) // 2

    chunk = audio[mid:mid + frame_len]
    if len(chunk) < frame_len:
        chunk = np.pad(chunk, (0, frame_len - len(chunk)))

    freqs = np.fft.rfftfreq(NFFT, 1 / sr)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for i, w in enumerate(windows):
        _, sp_lk = spectral_leakage(chunk, w, NFFT)
        _, sp_sn = compute_snr(chunk, w, NFFT)

        lk = results[w]["leakage"]
        sn = results[w]["snr"]

        # Plot leakage spectrum
        axes[0, i].plot(freqs, 10 * np.log10(sp_lk + 1e-10))
        axes[0, i].set_title(f"{w} | leakage={lk:.5f}")
        axes[0, i].set_xlabel("Hz")
        axes[0, i].set_ylabel("dB")

        # Plot SNR spectrum
        axes[1, i].plot(freqs, 10 * np.log10(sp_sn + 1e-10))
        axes[1, i].set_title(f"SNR={sn:.2f} dB")
        axes[1, i].set_xlabel("Hz")
        axes[1, i].set_ylabel("dB")

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/leakage_snr_comparison.png", dpi=150)
    plt.close()

    print("Saved plot to data/leakage_snr_comparison.png\n")

    # Print comparison table
    print("=" * 50)
    print(f"{'Window':<12} {'Leakage':>12} {'SNR (dB)':>12}")
    print("-" * 50)

    for w, v in results.items():
        print(f"{w:<12} {v['leakage']:>12.6f} {v['snr']:>12.2f}")

    print("=" * 50)
    print("Lower leakage is better, higher SNR is better.")


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    samples = get_librispeech(local_root=local, max_samples=20)
    results = analyze(samples)

    plot_and_print(results, samples[0])