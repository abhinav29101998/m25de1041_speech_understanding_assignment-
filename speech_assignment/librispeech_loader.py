import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path


# Load using HuggingFace (streaming)
def load_librispeech_hf(split="train.clean.100", max_samples=None):
    from datasets import load_dataset

    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split=split,
        streaming=True,
        trust_remote_code=True
    )

    samples = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        audio = np.array(item["audio"]["array"], dtype=np.float32)

        samples.append({
            "audio": audio,
            "sr": item["audio"]["sampling_rate"],
            "text": item.get("text", ""),
            "speaker_id": str(item.get("speaker_id", i)),
            "gender": "unknown"
        })

    print("Loaded:", len(samples))
    return samples


# Load from local dataset
def load_librispeech_local(root, max_samples=None, seed=42):
    root = Path(root)

    speaker_meta = {}
    speakers_txt = root.parent / "SPEAKERS.TXT"

    if speakers_txt.exists():
        for line in speakers_txt.read_text().splitlines():
            if line.startswith(";") or not line.strip():
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                spk = parts[0]
                gender = parts[1].upper()
                speaker_meta[spk] = "male" if gender == "M" else "female"

    items = []

    for trans_file in root.rglob("*.trans.txt"):
        transcripts = {}

        for line in trans_file.read_text().splitlines():
            if line.strip():
                utt, *words = line.split()
                transcripts[utt] = " ".join(words)

        for utt, text in transcripts.items():
            path = trans_file.parent / f"{utt}.flac"
            if not path.exists():
                continue

            spk = utt.split("-")[0]

            items.append({
                "path": str(path),
                "text": text,
                "speaker_id": spk,
                "gender": speaker_meta.get(spk, "unknown")
            })

    random.seed(seed)
    random.shuffle(items)

    if max_samples:
        items = items[:max_samples]

    samples = []

    for it in items:
        try:
            audio, sr = sf.read(it["path"])

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            samples.append({
                **it,
                "audio": audio.astype(np.float32),
                "sr": sr
            })

        except:
            continue

    print("Loaded:", len(samples))
    return samples


# Main loader
def get_librispeech(local_root=None, max_samples=None):
    if local_root and os.path.isdir(local_root):
        return load_librispeech_local(local_root, max_samples)
    else:
        print("Using HuggingFace loader")
        return load_librispeech_hf(max_samples=max_samples)


# Helper functions
def get_unique_speakers(samples):
    return sorted(set(s["speaker_id"] for s in samples))


def group_by_speaker(samples):
    groups = {}
    for s in samples:
        groups.setdefault(s["speaker_id"], []).append(s)
    return groups


# Compute mel spectrogram (used by other modules)
def compute_mel_spectrogram(audio, sr, n_mels=40):
    from mfcc_manual import (
        pre_emphasis, frame_signal, apply_window,
        compute_power_spectrum, mel_filterbank, log_compress
    )

    sig = pre_emphasis(audio.astype(np.float64))
    frames, _ = frame_signal(sig, sr)

    windowed = apply_window(frames, "hamming")
    power = compute_power_spectrum(windowed)

    fbank = mel_filterbank(sr, num_filters=n_mels)
    mel = power @ fbank.T

    return log_compress(mel).astype(np.float32)


# Pad or trim to fixed length
def pad_or_trim(feat, max_frames=300):
    T = feat.shape[0]

    if T >= max_frames:
        return feat[:max_frames]

    pad = np.zeros((max_frames - T, feat.shape[1]), dtype=feat.dtype)
    return np.vstack([feat, pad])


if __name__ == "__main__":
    import sys

    local = sys.argv[1] if len(sys.argv) > 1 else None
    samples = get_librispeech(local_root=local, max_samples=10)

    print("Sample keys:", samples[0].keys())
    print("Audio shape:", samples[0]["audio"].shape)
    print("Speaker:", samples[0]["speaker_id"])