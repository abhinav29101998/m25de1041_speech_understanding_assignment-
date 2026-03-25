# Speech Processing Assignment
## Dataset: LibriSpeech train-clean-100

---

## Dataset Setup

**Option A — HuggingFace (automatic download, ~6 GB):**
```bash
pip install datasets soundfile
# Scripts auto-download on first run — no extra step needed
```

**Option B — Manual download:**
```bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
# Then pass the path as argument to each script:
#   python mfcc_manual.py /path/to/train-clean-100
```

---

## Install dependencies
```bash
pip install -r requirements.txt
```

---

## Q1: MFCC & Boundary Detection

```bash
cd q1

# Step 1: Extract MFCCs manually (no librosa.feature.mfcc)
python mfcc_manual.py [/path/to/train-clean-100]
#  data/mfcc_plot.png  (grid of 5 real LibriSpeech utterances)

# Step 2: Spectral leakage & SNR on 20 real utterances
python leakage_snr.py [/path/to/train-clean-100]
#  data/leakage_snr_comparison.png + printed table

# Step 3: Voiced/Unvoiced segmentation (3 utterances)
python voiced_unvoiced.py [/path/to/train-clean-100]
#  data/voiced_unvoiced_spk<id>.png for each utterance

# Step 4: Wav2Vec2 forced alignment + RMSE (needs internet)
python phonetic_mapping.py [/path/to/train-clean-100]
#  data/rmse_result.txt  (mean RMSE in ms across utterances)
```

---

## Q2: Disentangled Speaker Recognition

```bash
cd q2

# Full evaluation: baseline vs. disentangled vs. ours
python eval.py [/path/to/train-clean-100]
#  results/comparison.png  results/results.json

# Train only:
python train.py [/path/to/train-clean-100]
#  results/model.pt   results/history.json
```

**Checkpoint mapping:**
- `results/model.pt` : Disentangled + Contrastive model (best, w_cont=0.2)
- To load: `ckpt = torch.load("results/model.pt"); model.load_state_dict(ckpt["model"])`

---

## Q3: Ethical Audit & Privacy

```bash
cd q3

# Step 1: Audit gender bias using SPEAKERS.TXT labels
python audit.py [/path/to/train-clean-100]
#  data/audit_plots.pdf   data/audit_results.json

# Step 2: Privacy-preserving voice transformation (male→female)
python privacymodule.py [/path/to/train-clean-100]
#  data/privacy_module.pt   data/privacy_transform_demo.png

# Step 3: ASR with fairness loss (compares male/female/unknown WER gap)
python train_fair.py [/path/to/train-clean-100]
#  data/fairness_training.png   data/fairness_results.json
```

---

## LibriSpeech Dataset Facts

| Property         | Value |
|------------------|-------|
| Dataset          | train-clean-100 |
| Total hours      | ~100 hours |
| Utterances       | ~28,539 |
| Unique speakers  | 251 (125 female, 126 male) |
| Sample rate      | 16 kHz |
| Format           | FLAC |
| Gender labels    | SPEAKERS.TXT (M/F per speaker) |
| Transcripts      | .trans.txt per chapter |

---

## Concepts at a Glance

| Concept          | Simple Analogy |
|------------------|----------------|
| MFCC             | Compress audio into 13 numbers per 25ms frame |
| Windowing        | Smooth chunk edges before FFT to avoid fake frequencies |
| Spectral Leakage | Fake frequencies from abrupt cuts in the signal |
| Cepstrum         | FFT of the FFT separates pitch from vocal tract |
| Voiced/Unvoiced  | Vowels (periodic) vs consonants like 's','f' (noisy) |
| Disentanglement  | Separate WHO is speaking from WHERE they are |
| Fairness Loss    | Penalize performance gap between genders |
| Privacy Module   | Change how someone sounds without changing what they say |
