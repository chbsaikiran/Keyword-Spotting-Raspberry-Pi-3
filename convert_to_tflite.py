"""
Convert ONNX FP32 models to TFLite INT8 for Raspberry Pi 3 (armv7l).

Run this on your training machine AFTER running the quantize scripts.

Requirements (training machine):
    pip install onnx-tf tensorflow onnxsim

Input files (produced by quantize_smoothquant.py / quantize_awq.py / quantize.py):
    checkpoints/keyword_spotting_smoothquant_fp32.onnx
    checkpoints/keyword_spotting_awq_fp32.onnx
    checkpoints/keyword_spotting_smooth.onnx

Output files (copy all three + preprocess_config.json to Raspberry Pi):
    checkpoints/keyword_spotting_smoothquant.tflite
    checkpoints/keyword_spotting_awq.tflite
    checkpoints/keyword_spotting_combined.tflite
"""

import os
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader

import onnx
import onnxsim
from onnx_tf.backend import prepare
import tensorflow as tf

from train import (
    SpeechCommandsDataset,
    collate_fn,
    DATA_ROOT,
    N_MELS,
    MAX_FRAMES,
)

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_DIR = "./checkpoints"
N_CALIB  = 200

CONVERSIONS = [
    {
        "name":        "SmoothQuant",
        "onnx_path":   f"{CKPT_DIR}/keyword_spotting_smoothquant_fp32.onnx",
        "tflite_path": f"{CKPT_DIR}/keyword_spotting_smoothquant.tflite",
    },
    {
        "name":        "AWQ",
        "onnx_path":   f"{CKPT_DIR}/keyword_spotting_awq_fp32.onnx",
        "tflite_path": f"{CKPT_DIR}/keyword_spotting_awq.tflite",
    },
    {
        "name":        "Combined (SQ+AWQ)",
        "onnx_path":   f"{CKPT_DIR}/keyword_spotting_smooth.onnx",
        "tflite_path": f"{CKPT_DIR}/keyword_spotting_combined.tflite",
    },
]


# ── Calibration data ──────────────────────────────────────────────────────────

def load_calibration_samples(n: int = N_CALIB) -> list:
    """Return a list of float32 numpy arrays [1, MAX_FRAMES, N_MELS]."""
    ds = SpeechCommandsDataset(DATA_ROOT, "validation", augment=False)
    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    num_workers=0, collate_fn=collate_fn)
    samples = []
    for i, (mel, _) in enumerate(dl):
        if i >= n:
            break
        samples.append(mel.numpy().astype(np.float32))
    print(f"  Loaded {len(samples)} calibration samples")
    return samples


# ── Conversion ────────────────────────────────────────────────────────────────

def _simplify_onnx(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Run onnx-simplifier to:
      - Fold constants (precomputes the static causal mask)
      - Expand complex ops (LayerNormalization → primitives)
      Both steps improve onnx-tf compatibility.
    """
    simplified, ok = onnxsim.simplify(model)
    if ok:
        return simplified
    print("    onnxsim simplification had warnings — using original model")
    return model


def _onnx_to_saved_model(onnx_path: str, saved_model_dir: str):
    """ONNX FP32 → TensorFlow SavedModel via onnx-tf."""
    model = onnx.load(onnx_path)
    model = _simplify_onnx(model)
    tf_rep = prepare(model)
    tf_rep.export_graph(saved_model_dir)


def _saved_model_to_tflite(
    saved_model_dir: str,
    tflite_path: str,
    calib_samples: list,
):
    """
    TF SavedModel → TFLite with full INT8 quantization.

    Strategy
    --------
    - Weights: INT8 per-channel
    - Activations: INT8 (calibrated via representative dataset)
    - Input / output tensors: kept as float32 so inference scripts need
      no extra scale/zero-point handling
    - Fallback: TFLITE_BUILTINS (float32) for any op not in INT8 kernel set,
      so the model always converts even if a few ops stay in float32
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for sample in calib_samples:
            yield [sample]   # shape [1, MAX_FRAMES, N_MELS]

    converter.representative_dataset    = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,   # float32 fallback
    ]
    # Float32 I/O — simpler inference, no quantization bookkeeping on RPi
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(tflite_path)), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / 1e6
    print(f"  Saved: {tflite_path}  ({size_mb:.2f} MB)")


def convert(onnx_path: str, tflite_path: str, calib_samples: list):
    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = os.path.join(tmp, "saved_model")
        print("  ONNX → TF SavedModel ...", end=" ", flush=True)
        _onnx_to_saved_model(onnx_path, saved_model_dir)
        print("done")
        print("  TF SavedModel → TFLite INT8 ...", end=" ", flush=True)
        _saved_model_to_tflite(saved_model_dir, tflite_path, calib_samples)
        print("done")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_tflite(tflite_path: str, calib_samples: list, keywords: list):
    """Quick sanity check: run a few samples through the TFLite model."""
    from train import SpeechCommandsDataset, collate_fn, DATA_ROOT, KEYWORDS
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    ds = SpeechCommandsDataset(DATA_ROOT, "validation", augment=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=0, collate_fn=collate_fn)
    correct = total = 0
    for i, (mel, label) in enumerate(dl):
        if i >= 300:
            break
        interpreter.set_tensor(inp_idx, mel.numpy().astype(np.float32))
        interpreter.invoke()
        logits = interpreter.get_tensor(out_idx)[0]
        pred    = int(logits.argmax())
        correct += int(pred == label.item())
        total   += 1

    acc = correct / total
    print(f"  Validation accuracy (300 samples): {acc:.4f}")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading calibration data from validation set...")
    calib_samples = load_calibration_samples(N_CALIB)

    results = []
    for entry in CONVERSIONS:
        name       = entry["name"]
        onnx_path  = entry["onnx_path"]
        tflite_path = entry["tflite_path"]

        if not os.path.isfile(onnx_path):
            print(f"\n[{name}] SKIP — {onnx_path} not found.")
            print(f"  Run the corresponding quantize script first.")
            continue

        print(f"\n[{name}]  {onnx_path}")
        try:
            convert(onnx_path, tflite_path, calib_samples)
            results.append(tflite_path)
        except Exception as exc:
            print(f"  ERROR during conversion: {exc}")
            raise

    if results:
        print("\nValidating converted models...")
        from train import KEYWORDS
        for path in results:
            print(f"  {path}")
            validate_tflite(path, calib_samples, KEYWORDS)

    print("\n" + "=" * 55)
    print("Copy these files to your Raspberry Pi 3:")
    for path in results:
        print(f"  {path}")
    print(f"  checkpoints/preprocess_config.json")
    print("=" * 55)
    print("\nOn the Pi, run:")
    print("  python inference_rpi_smoothquant.py --mode benchmark")
    print("  python inference_rpi_awq.py         --mode benchmark")


if __name__ == "__main__":
    main()
