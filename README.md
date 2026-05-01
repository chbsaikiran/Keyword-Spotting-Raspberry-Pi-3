# Keyword Spotting on Raspberry Pi 3

A decoder-only transformer trained to recognize 35 keywords from the Google Speech Commands v2 dataset, quantized to INT8 TFLite and deployed on a Raspberry Pi 3.

## Overview

The pipeline has three stages:

1. **Train** — train a small transformer on the host machine (GPU recommended)
2. **Quantize** — apply post-training quantization (AWQ, SmoothQuant, or both) and export directly to TFLite INT8
3. **Infer** — run the INT8 TFLite model on the Raspberry Pi 3 from a microphone or audio file

### Model Architecture

- Decoder-only transformer with a learnable CLS token appended at the **end** of the sequence
- In causal attention the last position attends to all prior frames, making it a natural classification pooling point
- Input: 80-bin log-mel spectrogram at 16 kHz, 10 ms hop → `[101, 80]` frames
- 4 decoder blocks, D_MODEL=256, 4 heads, D_FF=512 — **~2.1 M parameters**
- Trained on 35 keywords: `backward bed bird cat dog down eight five follow forward four go happy house learn left marvin nine no off on one right seven sheila six stop three tree two up visual wow yes zero`

### Quantization Methods

| Script | Method | Output model |
|---|---|---|
| `quantize_awq.py` | AWQ only | `keyword_spotting_awq.tflite` |
| `quantize_smoothquant.py` | SmoothQuant only | `keyword_spotting_smoothquant.tflite` |
| `quantize.py` | SmoothQuant + AWQ combined | `keyword_spotting_combined.tflite` |

All three use `litert-torch` (PT2E static INT8 quantization with per-channel weights) to convert the transformed PyTorch model directly to TFLite — no ONNX step.

---

## Requirements

### Training machine (Linux / macOS / Windows, GPU recommended)

```bash
pip install -r requirements_training.txt
```

### Raspberry Pi 3

```bash
pip install -r requirements_rpi.txt
```

---

## Usage

### 1. Train

```bash
python train.py
```

- Downloads the Speech Commands v2 dataset (~2.4 GB) on first run into `./data/`
- Trains for 40 epochs with AdamW + OneCycleLR + label smoothing
- Saves the best checkpoint (by validation accuracy) to `checkpoints/best_model.pt`

### 2. Quantize

Run **one** of the following on the training machine:

```bash
# AWQ only (recommended — best accuracy for transformers)
python quantize_awq.py

# SmoothQuant only
python quantize_smoothquant.py

# SmoothQuant + AWQ combined
python quantize.py
```

Each script:
1. Loads `checkpoints/best_model.pt`
2. Calibrates activation statistics on the validation set
3. Applies weight transformations (lossless, folded into LayerNorm)
4. Exports directly to TFLite INT8 via PT2E quantization (no ONNX step)
5. Prints FP32 vs INT8 TFLite accuracy comparison
6. Saves `checkpoints/preprocess_config.json`

### 3. Evaluate (on training machine)

```bash
python evaluate.py --model checkpoints/keyword_spotting_awq.tflite
python evaluate.py --model checkpoints/keyword_spotting_smoothquant.tflite
python evaluate.py --model checkpoints/keyword_spotting_combined.tflite
```

### 4. Deploy to Raspberry Pi 3

Copy the two output files to the Pi:

```bash
scp checkpoints/keyword_spotting_awq.tflite pi@<pi-ip>:~/keyword_spotting/
scp checkpoints/preprocess_config.json      pi@<pi-ip>:~/keyword_spotting/
```

### 5. Run inference on Raspberry Pi 3

```bash
# Real-time microphone (1-second sliding windows)
python inference_rpi_awq.py --mode realtime

# Single audio file
python inference_rpi_awq.py --mode file --file clip.wav

# Latency benchmark (100 silent clips)
python inference_rpi_awq.py --mode benchmark
```

For SmoothQuant or combined models replace `inference_rpi_awq.py` with `inference_rpi_smoothquant.py` or `inference_rpi.py` respectively.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `keyword_spotting_awq.tflite` | Path to TFLite model |
| `--config` | `preprocess_config.json` | Path to preprocessing config |
| `--mode` | `realtime` | `realtime` / `file` / `benchmark` |
| `--file` | — | Audio file path (mode=file only) |
| `--threshold` | `0.70` | Confidence threshold for detection |

---

## Project Structure

```
├── train.py                        # Model definition + training
├── quantize_awq.py                 # AWQ-only quantization → TFLite
├── quantize_smoothquant.py         # SmoothQuant-only quantization → TFLite
├── quantize.py                     # SmoothQuant + AWQ combined → TFLite
├── evaluate.py                     # Evaluate TFLite model on validation set
├── inference_rpi_awq.py            # Pi inference script for AWQ model
├── inference_rpi_smoothquant.py    # Pi inference script for SmoothQuant model
├── inference_rpi.py                # Pi inference script for combined model
├── requirements_training.txt       # Dependencies for training machine
├── requirements_rpi.txt            # Dependencies for Raspberry Pi
└── checkpoints/                    # Saved models (generated)
    ├── best_model.pt
    ├── keyword_spotting_awq.tflite
    ├── keyword_spotting_smoothquant.tflite
    ├── keyword_spotting_combined.tflite
    └── preprocess_config.json
```

---

## Audio Preprocessing

All inference scripts reproduce the exact same preprocessing as training:

- Resample to 16 kHz, pad/trim to 1 second
- 80-bin mel spectrogram: FFT size 512, window 25 ms, hop 10 ms
- Log mel + per-sample mean/variance normalization

If `librosa` is available it is used; otherwise a pure NumPy/SciPy fallback is used automatically.
