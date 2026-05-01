"""
SmoothQuant-only post-training quantization pipeline.

Steps:
  1. Calibrate per-channel activation max values on the validation set
  2. Compute per-channel smooth scales  s = act_max^alpha / w_max^(1-alpha)
  3. Absorb scale into (LayerNorm, Linear) pairs — mathematically lossless
  4. Export transformed model to TFLite INT8 via PT2E quantization (litert-torch)
  5. Accuracy comparison FP32 vs INT8 TFLite

Output files (copy both to Raspberry Pi):
    checkpoints/keyword_spotting_smoothquant.tflite
    checkpoints/preprocess_config.json
"""

import copy
import json
import os

import torch
import torch.nn as nn
import litert_torch as ltt
import tensorflow as tf
from ai_edge_quantizer import quantizer as aq
from torch.utils.data import DataLoader

from train import (
    CKPT_PATH,
    DATA_ROOT,
    HOP_LENGTH,
    KEYWORDS,
    MAX_FRAMES,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WIN_LENGTH,
    KeywordSpottingTransformer,
    SpeechCommandsDataset,
    collate_fn,
)

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_DIR       = "./checkpoints"
TFLITE_PATH    = f"{CKPT_DIR}/keyword_spotting_smoothquant.tflite"
CONFIG_PATH    = f"{CKPT_DIR}/preprocess_config.json"

CALIB_BATCHES  = 64
CALIB_BATCH_SZ = 32
SQ_ALPHA       = 0.5   # 0 = push all difficulty to weights, 1 = to activations


# ── Activation calibration ────────────────────────────────────────────────────

class ActivationCalibrator:
    """
    Attach forward hooks to every nn.Linear and track the per-input-channel
    maximum absolute activation across the entire calibration set.
    """

    def __init__(self, model: nn.Module):
        self.model  = model
        self.scales: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def _register(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().float()          # [..., in_features]
            ch_max = x.reshape(-1, x.shape[-1]).abs().max(dim=0).values
            if name in self.scales:
                self.scales[name] = torch.maximum(self.scales[name], ch_max)
            else:
                self.scales[name] = ch_max.clone()
        return hook

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def run(self, dataloader: DataLoader, n_batches: int) -> dict[str, torch.Tensor]:
        self._register()
        self.model.eval()
        for i, (mel, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            self.model(mel)
        self._remove()
        return self.scales


# ── SmoothQuant transformation ────────────────────────────────────────────────

def _smooth_pair(
    norm: nn.LayerNorm,
    linear: nn.Linear,
    act_scale: torch.Tensor,
    alpha: float,
):
    """
    In-place SmoothQuant for one (LayerNorm → Linear) pair.

    Derivation
    ----------
    We want to equate:  linear(norm(x))  ==  linear_new(norm_new(x))

    Introduce per-channel scale  s  (shape: [in_features]):
        norm_new  : gamma_new = gamma / s   →  norm_new(x) = norm(x) / s
        linear_new: W_new[:,j] = W[:,j] * s[j]

    Then:  linear_new(norm_new(x)) = (W * s) · (norm(x) / s) = W · norm(x) ✓

    Scale formula (SmoothQuant paper, Eq. 1):
        s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)

    When alpha=0.5 this is the geometric mean — splits difficulty equally.
    """
    w_col_max = linear.weight.abs().max(dim=0).values.clamp(min=1e-8)
    a_max     = act_scale.clamp(min=1e-8)
    s         = (a_max ** alpha) / (w_col_max ** (1.0 - alpha))
    s         = s.clamp(min=1e-8)

    # Absorb 1/s into LayerNorm
    norm.weight.data.div_(s)
    if norm.bias is not None:
        norm.bias.data.div_(s)

    # Absorb s into Linear (per input-channel column)
    linear.weight.data.mul_(s.unsqueeze(0))


def apply_smooth_quant(
    model: KeywordSpottingTransformer,
    act_scales: dict[str, torch.Tensor],
    alpha: float,
) -> KeywordSpottingTransformer:
    """
    Return a deep-copied model with SmoothQuant applied to every
    (norm1 → attn.qkv) and (norm2 → ff.net[0]) pair in each decoder block.
    """
    model = copy.deepcopy(model)
    for i, block in enumerate(model.blocks):
        _smooth_pair(
            block.norm1, block.attn.qkv,
            act_scales.get(
                f"blocks.{i}.attn.qkv",
                torch.ones(block.attn.qkv.weight.shape[1]),
            ),
            alpha,
        )
        _smooth_pair(
            block.norm2, block.ff.net[0],
            act_scales.get(
                f"blocks.{i}.ff.net.0",
                torch.ones(block.ff.net[0].weight.shape[1]),
            ),
            alpha,
        )
    return model


# ── TFLite INT8 export ────────────────────────────────────────────────────────

def export_tflite(model: nn.Module, tflite_path: str):
    """SmoothQuant-transformed PyTorch model → TFLite INT8 via litert-torch + ai_edge_quantizer."""
    model = model.eval()
    sample = (torch.zeros(1, MAX_FRAMES, N_MELS),)

    fp32_bytes = ltt.convert(model, sample).model_content()

    qt = aq.Quantizer(bytearray(fp32_bytes))
    qt.add_dynamic_config(
        regex=".*",
        operation_name=aq._TFLOpName.FULLY_CONNECTED,
        num_bits=8,
        granularity=aq.qtyping.QuantGranularity.CHANNELWISE,
    )
    int8_bytes = qt.quantize().quantized_model

    os.makedirs(os.path.dirname(os.path.abspath(tflite_path)), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(int8_bytes)

    size_mb = os.path.getsize(tflite_path) / 1e6
    print(f"  Saved: {tflite_path}  ({size_mb:.2f} MB)")


# ── Evaluation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_pytorch(model: nn.Module, dataloader: DataLoader, n: int = 30) -> float:
    model.eval()
    correct = total = 0
    for i, (mel, labels) in enumerate(dataloader):
        if i >= n:
            break
        preds    = model(mel).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def eval_tflite(tflite_path: str, dataloader: DataLoader, n: int = 30) -> float:
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    correct = total = 0
    for i, (mel, labels) in enumerate(dataloader):
        if i >= n:
            break
        for j in range(mel.shape[0]):
            interp.set_tensor(inp_idx, mel[j : j + 1].numpy())
            interp.invoke()
            logits = interp.get_tensor(out_idx)[0]
            correct += int(logits.argmax() == labels[j].item())
            total += 1
    return correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("Loading checkpoint...")
    ckpt  = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = KeywordSpottingTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded  (training val_acc: {ckpt['val_acc']:.4f})")

    # Build dataloaders
    ds = SpeechCommandsDataset(DATA_ROOT, "validation", augment=False)
    calib_dl = DataLoader(ds, CALIB_BATCH_SZ, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
    val_dl   = DataLoader(ds, CALIB_BATCH_SZ, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

    # ── Step 1: Calibrate activations ─────────────────────────────────────────
    print(f"\n[1/3] Calibrating activations ({CALIB_BATCHES} batches)...")
    calibrator = ActivationCalibrator(model)
    act_scales = calibrator.run(calib_dl, CALIB_BATCHES)
    print(f"  Scales collected for {len(act_scales)} Linear layers")

    # ── Step 2: Apply SmoothQuant ─────────────────────────────────────────────
    print(f"[2/3] Applying SmoothQuant  (alpha={SQ_ALPHA})...")
    model_sq = apply_smooth_quant(model, act_scales, SQ_ALPHA)

    # Sanity check: transformed model must match original accuracy
    acc_orig = eval_pytorch(model,    val_dl, n=20)
    acc_sq   = eval_pytorch(model_sq, val_dl, n=20)
    print(f"  FP32 accuracy  original={acc_orig:.4f}  after SmoothQuant={acc_sq:.4f}")
    assert abs(acc_orig - acc_sq) < 0.005, (
        "SmoothQuant broke mathematical equivalence — check _smooth_pair()"
    )

    # ── Step 3: Export to TFLite INT8 ────────────────────────────────────────
    print("[3/3] Exporting to TFLite INT8 (PT2E + litert-torch)...")
    export_tflite(model_sq, TFLITE_PATH)

    # ── Accuracy comparison ───────────────────────────────────────────────────
    print("\nValidating (30 batches each)...")
    acc_fp32   = eval_pytorch(model_sq, val_dl, n=30)
    acc_tflite = eval_tflite(TFLITE_PATH, val_dl, n=30)
    print(f"  FP32  (SmoothQuant) : {acc_fp32:.4f}")
    print(f"  INT8  TFLite        : {acc_tflite:.4f}")
    print(f"  Accuracy drop       : {acc_fp32 - acc_tflite:.4f}")

    # ── Save preprocessing config ─────────────────────────────────────────────
    with open(CONFIG_PATH, "w") as f:
        json.dump(
            {
                "sample_rate": SAMPLE_RATE, "n_fft": N_FFT,
                "win_length": WIN_LENGTH,   "hop_length": HOP_LENGTH,
                "n_mels": N_MELS,           "max_frames": MAX_FRAMES,
                "keywords": KEYWORDS,
            },
            f, indent=2,
        )

    print(f"\nFiles ready for Raspberry Pi 3:")
    print(f"  {TFLITE_PATH}")
    print(f"  {CONFIG_PATH}")
    print(
        "\nRun on Pi:\n"
        "  python inference_rpi_smoothquant.py --mode realtime"
    )


if __name__ == "__main__":
    main()
