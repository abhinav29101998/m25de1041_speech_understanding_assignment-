# Speech Processing Assignment
**Roll Number:** M25DE1041  
**Dataset:** LibriSpeech train-clean-100

---

## Project Structure

```
M25DE1041/
├── README.md
├── requirements.txt
├── librispeech_loader.py        ← shared dataset loader (used by all questions)
│
├── q1/
│   ├── mfcc_manual.py           ← manual MFCC/cepstrum engine
│   ├── leakage_snr.py           ← spectral leakage & SNR analysis
│   ├── voiced_unvoiced.py       ← voiced/unvoiced boundary detection
│   ├── phonetic_mapping.py      ← Wav2Vec2 forced alignment + RMSE
│   └── data_manifest.txt        ← list of audio files used
│
├── q2/
│   ├── review.md                ← technical critical review of paper
│   ├── train.py                 ← disentangled speaker recognition training
│   └── eval.py                  ← evaluation vs baseline + results
│
└── q3/
    ├── audit.py                 ← bias audit of LibriSpeech
    ├── privacymodule.py         ← privacy-preserving voice transformer
    ├── pp_demo.py               ← standalone demo: male → female transformation
    └── train_fair.py            ← ASR training with fairness loss
```

---

## Setup

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Dataset (choose one option)

**Option A — HuggingFace auto-download (~6 GB, no extra steps):**
```bash
# Scripts download automatically on first run
# Just run any script without arguments:
python q1/mfcc_manual.py
```

**Option B — Manual local download:**
```bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
# Then pass the path as an argument:
python q1/mfcc_manual.py /path/to/train-clean-100
```

---

## How to Run

### Question 1: MFCC & Boundary Detection

```bash
cd q1

# 1. Manual MFCC extraction (no librosa.feature.mfcc)
python mfcc_manual.py [/path/to/train-clean-100]
# Output: data/mfcc_plot.png

# 2. Spectral leakage & SNR for 3 window types
python leakage_snr.py [/path/to/train-clean-100]
# Output: data/leakage_snr_comparison.png + printed table

# 3. Voiced/Unvoiced boundary detection
python voiced_unvoiced.py [/path/to/train-clean-100]
# Output: data/voiced_unvoiced_spk<id>.png

# 4. Phonetic mapping + RMSE (downloads Wav2Vec2 on first run)
python phonetic_mapping.py [/path/to/train-clean-100]
# Output: data/rmse_result.txt
```

---

### Question 2: Disentangled Speaker Recognition

```bash
cd q2

# Train all 3 models and compare (baseline / disentangled / ours)
python eval.py [/path/to/train-clean-100]
# Output: results/comparison.png, results/results.json

# Train only
python train.py [/path/to/train-clean-100]
# Output: results/model.pt, results/history.json
```

**Checkpoint:** `results/model.pt` → Disentangled + Contrastive model  
**Load with:**
```python
ckpt = torch.load("results/model.pt")
model.load_state_dict(ckpt["model"])
```

---

### Question 3: Ethical Audit & Privacy

```bash
cd q3

# 1. Bias audit using real gender labels from SPEAKERS.TXT
python audit.py [/path/to/train-clean-100]
# Output: data/audit_plots.pdf, data/audit_results.json

# 2. Train privacy-preserving transformer
python privacymodule.py [/path/to/train-clean-100]
# Output: data/privacy_module.pt, data/privacy_transform_demo.png

# 3. Standalone privacy demo (male → female transformation)
python pp_demo.py [/path/to/train-clean-100]
# Output: data/examples/pp_comparison.png, data/examples/pair_*.png

# 4. ASR training with fairness loss
python train_fair.py [/path/to/train-clean-100]
# Output: data/fairness_training.png, data/fairness_results.json
```

---

## Dataset Details

| Property        | Value                              |
|-----------------|------------------------------------|
| Name            | LibriSpeech train-clean-100        |
| Total hours     | ~100 hours                         |
| Utterances      | ~28,539                            |
| Unique speakers | 251 (125 female, 126 male)         |
| Sample rate     | 16,000 Hz                          |
| Format          | FLAC                               |
| Gender labels   | SPEAKERS.TXT (M/F per speaker ID)  |
| Transcripts     | `<spk>-<chapter>.trans.txt`        |
| License         | CC BY 4.0                          |

---

## Tools Used

| Tool            | Version   | Purpose                              |
|-----------------|-----------|--------------------------------------|
| Python          | 3.9+      | All scripts                          |
| PyTorch         | 2.0+      | Model training (Q2, Q3)              |
| HuggingFace     | 4.35+     | Wav2Vec2 (Q1), datasets loader       |
| NumPy           | 1.24+     | Signal processing (Q1)               |
| soundfile       | 0.12+     | FLAC audio loading                   |
| matplotlib      | 3.7+      | All plots                            |
| scikit-learn    | 1.3+      | EER computation (Q2)                 |
