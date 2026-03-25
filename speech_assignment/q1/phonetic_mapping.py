import sys
import os
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from librispeech_loader import get_librispeech
from voiced_unvoiced import detect_boundaries


# Load pretrained Wav2Vec2 model
def load_model():
    print("Loading model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    return processor, model


# Get boundaries from model predictions
def model_boundaries(audio, sr, processor, model, frame_dur=0.02):
    if sr != 16000:
        try:
            import torchaudio
            wav = torch.tensor(audio).unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 16000)(wav).squeeze(0).numpy()
            sr = 16000
        except:
            pass

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    pred_ids = logits.argmax(-1).numpy()

    # Boundary when predicted label changes
    boundaries = [
        i * frame_dur
        for i in range(1, len(pred_ids))
        if pred_ids[i] != pred_ids[i - 1]
    ]

    phones = processor.batch_decode(logits.argmax(-1, keepdim=True).T)[0]

    return np.array(boundaries), phones, pred_ids, frame_dur


# Boundaries from manual cepstrum method
def manual_boundaries(audio, sr):
    segments, _, _ = detect_boundaries(audio, sr)
    return np.array([s for s, e, lbl in segments[1:]])


# RMSE between manual and model boundaries
def compute_rmse(manual_b, model_b):
    if len(manual_b) == 0 or len(model_b) == 0:
        return float("inf")

    errors = []
    for mb in manual_b:
        closest = model_b[np.argmin(np.abs(model_b - mb))]
        errors.append((mb - closest) ** 2)

    return float(np.sqrt(np.mean(errors)))


# Map segments to dominant phoneme
def phoneme_map(pred_ids, model_b, frame_dur, processor):
    times = [0.0] + list(model_b)
    result = []

    for i in range(len(times)):
        start = times[i]
        end = times[i + 1] if i + 1 < len(times) else len(pred_ids) * frame_dur

        fs = int(start / frame_dur)
        fe = int(end / frame_dur)

        segment = pred_ids[fs:fe]

        if len(segment):
            dominant = np.bincount(segment).argmax()
            phone = processor.tokenizer.convert_ids_to_tokens([int(dominant)])[0]
            result.append((start, end, phone))

    return result


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None

    os.makedirs("data", exist_ok=True)
    samples = get_librispeech(local_root=local, max_samples=5)

    processor, model = load_model()

    all_rmse = []

    print(f"\n{'Utterance':<30} {'Manual':>8} {'Model':>8} {'RMSE(ms)':>10}")
    print("-" * 60)

    for s in samples:
        audio, sr = s["audio"], s["sr"]

        man_b = manual_boundaries(audio, sr)
        mod_b, _, pred_ids, fdur = model_boundaries(audio, sr, processor, model)

        rmse = compute_rmse(man_b, mod_b)
        all_rmse.append(rmse)

        name = s['speaker_id'] + '-' + s.get('utt_id', '?')

        print(f"{name:<30} {len(man_b):>8} {len(mod_b):>8} {rmse*1000:>10.2f}")

    valid_rmse = [r for r in all_rmse if r != float("inf")]
    mean_rmse = np.mean(valid_rmse)

    print("-" * 60)
    print(f"{'Mean RMSE':<48} {mean_rmse*1000:>10.2f} ms")

    # Show phoneme map for first sample
    s = samples[0]
    mod_b, _, pred_ids, fdur = model_boundaries(s["audio"], s["sr"], processor, model)

    mapping = phoneme_map(pred_ids, mod_b, fdur, processor)

    print("\nPhoneme segments (first sample):")
    print(f"{'Start':>7} {'End':>7} {'Phone':>8}")

    for seg in mapping[:15]:
        print(f"{seg[0]:>7.3f} {seg[1]:>7.3f} {seg[2]:>8}")

    with open("data/rmse_result.txt", "w") as f:
        f.write(f"Mean RMSE: {mean_rmse:.4f} sec\n")

    print("\nSaved: data/rmse_result.txt")