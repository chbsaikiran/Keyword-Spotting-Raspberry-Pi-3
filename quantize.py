"""
Post-training quantization pipeline:
  1. SmoothQuant  — equalize activation/weight ranges via per-channel scaling
  2. AWQ          — activation-aware weight quantization (protect salient channels)
  3. ONNX export  — opset 17
  4. ONNX Runtime static INT8 quantization (QDQ format, per-channel weights)

Run this on your training machine (GPU or fast CPU), then copy
  checkpoints/keyword_spotting_int8.onnx
  checkpoints/preprocess_config.json
to the Raspberry Pi.
"""

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

from train import (
    KeywordSpottingTransformer,
    SpeechCommandsDataset,
    collate_fn,
    CKPT_PATH,
    DATA_ROOT,
    N_MELS,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    MAX_FRAMES,
    SAMPLE_RATE,
    KEYWORDS,
)

# ── Paths & constants ─────────────────────────────────────────────────────────

CKPT_DIR        = "./checkpoints"
ONNX_RAW_PATH   = f"{CKPT_DIR}/keyword_spotting_raw.onnx"
ONNX_SMOOTH_PATH = f"{CKPT_DIR}/keyword_spotting_smooth.onnx"
ONNX_QUANT_PATH = f"{CKPT_DIR}/keyword_spotting_int8.onnx"
CONFIG_PATH     = f"{CKPT_DIR}/preprocess_config.json"

CALIB_BATCHES   = 64    # calibration batches for stats collection
CALIB_BATCH_SZ  = 32
SQ_ALPHA        = 0.5   # SmoothQuant alpha: 0 = all weight, 1 = all activation
AWQ_N_GRID      = 20    # grid search steps for AWQ alpha
AWQ_N_BITS      = 8


# ── SmoothQuant ───────────────────────────────────────────────────────────────

class ActivationCalibrator:
    """Collect per-channel max-absolute activation stats via forward hooks."""

    def __init__(self, model: nn.Module):
        self.model  = model
        self.scales: dict[str, torch.Tensor] = {}
        self._hooks = []

    def register(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def hook(module, inputs, _output):
            x = inputs[0].detach().float()          # [..., in_features]
            channel_max = x.reshape(-1, x.shape[-1]).abs().max(dim=0).values
            if name in self.scales:
                self.scales[name] = torch.maximum(self.scales[name], channel_max)
            else:
                self.scales[name] = channel_max.clone()
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def run(self, dataloader: DataLoader, n_batches: int) -> dict[str, torch.Tensor]:
        self.register()
        self.model.eval()
        for i, (mel, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            self.model(mel)
        self.remove()
        return self.scales


def _apply_smooth_pair(
    norm: nn.LayerNorm,
    linear: nn.Linear,
    act_scale: torch.Tensor,
    alpha: float,
):
    """
    SmoothQuant in-place transformation for one (LayerNorm, Linear) pair.

    Derives per-channel scale s such that:
        norm_new(x) = norm(x) / s          [activations divided by s]
        linear_new.W[:,j] = W[:,j] * s[j]  [weights multiplied by s]
    The net result linear_new(norm_new(x)) == linear(norm(x)).
    """
    w_col_max = linear.weight.abs().max(dim=0).values.clamp(min=1e-8)
    act_max   = act_scale.clamp(min=1e-8)
    smooth    = (act_max ** alpha) / (w_col_max ** (1.0 - alpha))
    smooth    = smooth.clamp(min=1e-8)

    # Absorb into LayerNorm: gamma /= s  → output ≡ old_output / s
    norm.weight.data.div_(smooth)
    if norm.bias is not None:
        norm.bias.data.div_(smooth)

    # Absorb into Linear: W[:, j] *= s[j]
    linear.weight.data.mul_(smooth.unsqueeze(0))


def apply_smooth_quant(
    model: KeywordSpottingTransformer,
    act_scales: dict[str, torch.Tensor],
    alpha: float = SQ_ALPHA,
) -> KeywordSpottingTransformer:
    model = copy.deepcopy(model)
    for i, block in enumerate(model.blocks):
        _apply_smooth_pair(
            block.norm1, block.attn.qkv,
            act_scales.get(f"blocks.{i}.attn.qkv",
                           torch.ones(block.attn.qkv.weight.shape[1])),
            alpha,
        )
        _apply_smooth_pair(
            block.norm2, block.ff.net[0],
            act_scales.get(f"blocks.{i}.ff.net.0",
                           torch.ones(block.ff.net[0].weight.shape[1])),
            alpha,
        )
    return model


# ── AWQ ───────────────────────────────────────────────────────────────────────

def _pseudo_quantize(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Simulate per-output-channel symmetric INT-n quantization."""
    q_max = 2 ** (n_bits - 1) - 1
    scale = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / q_max
    return (w / scale).round().clamp(-q_max - 1, q_max) * scale


def _awq_best_scale(
    w: torch.Tensor,
    act_max: torch.Tensor,
    n_bits: int = AWQ_N_BITS,
    n_grid: int = AWQ_N_GRID,
) -> torch.Tensor:
    """
    Grid-search alpha ∈ [0,1] that minimises per-output-channel quantisation
    error when the input channel j is scaled by act_max[j]^alpha.

    Returns the best per-input-channel scale (normalised so mean == 1).
    """
    best_alpha = 0.5
    best_err   = float("inf")
    act_max    = act_max.clamp(min=1e-8)

    for step in range(n_grid + 1):
        alpha = step / n_grid
        scale = (act_max ** alpha).clamp(min=1e-8)
        scale = scale / scale.mean()              # normalise

        w_scaled = w / scale.unsqueeze(0)         # [out, in] / [in]
        w_quant  = _pseudo_quantize(w_scaled, n_bits)
        w_back   = w_quant * scale.unsqueeze(0)

        err = (w - w_back).pow(2).mean().item()
        if err < best_err:
            best_err   = err
            best_alpha = alpha

    scale = (act_max ** best_alpha).clamp(min=1e-8)
    return scale / scale.mean()


def compute_awq_scales(
    model: KeywordSpottingTransformer,
    act_scales: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    awq_scales: dict[str, torch.Tensor] = {}
    for i, block in enumerate(model.blocks):
        for key, linear in [
            (f"blocks.{i}.attn.qkv",   block.attn.qkv),
            (f"blocks.{i}.ff.net.0",   block.ff.net[0]),
        ]:
            if key not in act_scales:
                continue
            awq_scales[key] = _awq_best_scale(
                linear.weight.data.clone(),
                act_scales[key],
            )
    return awq_scales


def apply_awq_scales(
    model: KeywordSpottingTransformer,
    awq_scales: dict[str, torch.Tensor],
) -> KeywordSpottingTransformer:
    """
    Fold AWQ input-channel scales into (LayerNorm, Linear) pairs:
        linear.weight[:, j] /= scale[j]   → weight range reduced
        norm.weight[j]       *= scale[j]   → absorb inverse scale
    Net output is identical to un-scaled model.
    """
    model = copy.deepcopy(model)
    for i, block in enumerate(model.blocks):
        for key, norm, linear in [
            (f"blocks.{i}.attn.qkv", block.norm1, block.attn.qkv),
            (f"blocks.{i}.ff.net.0", block.norm2, block.ff.net[0]),
        ]:
            if key not in awq_scales:
                continue
            scale = awq_scales[key]
            linear.weight.data.div_(scale.unsqueeze(0))
            norm.weight.data.mul_(scale)
            if norm.bias is not None:
                norm.bias.data.mul_(scale)
    return model


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, path: str):
    model.eval()
    dummy = torch.zeros(1, MAX_FRAMES, N_MELS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(path))
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Exported: {path}  ({size_mb:.2f} MB)")


# ── ONNX Runtime static INT8 quantization ────────────────────────────────────

class SpeechCalibReader(CalibrationDataReader):
    def __init__(self, dataloader: DataLoader, n_batches: int):
        self._iter     = iter(dataloader)
        self._n        = n_batches
        self._count    = 0

    def get_next(self):
        if self._count >= self._n:
            return None
        try:
            mel, _ = next(self._iter)
            self._count += 1
            return {"mel": mel.numpy()}
        except StopIteration:
            return None

    def rewind(self):
        self._count = 0


def quantize_onnx_int8(raw_path: str, quant_path: str, dataloader: DataLoader):
    reader = SpeechCalibReader(dataloader, CALIB_BATCHES)
    quantize_static(
        model_input=raw_path,
        model_output=quant_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        # reduce_range avoids INT8 overflow on ARM NEON accumulators (RPi3)
        reduce_range=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "EnableSubgraph": False,
        },
    )
    size_mb = os.path.getsize(quant_path) / 1e6
    print(f"  Quantized: {quant_path}  ({size_mb:.2f} MB)")


# ── Validation ────────────────────────────────────────────────────────────────

def eval_pytorch(model: nn.Module, dataloader: DataLoader, n_batches: int = 30) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (mel, labels) in enumerate(dataloader):
            if i >= n_batches:
                break
            preds    = model(mel).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def eval_onnx(path: str, dataloader: DataLoader, n_batches: int = 30) -> float:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess    = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    correct = total = 0
    for i, (mel, labels) in enumerate(dataloader):
        if i >= n_batches:
            break
        logits   = sess.run(None, {"mel": mel.numpy()})[0]
        preds    = logits.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total   += labels.shape[0]
    return correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load checkpoint
    print("Loading checkpoint...")
    ckpt   = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model  = KeywordSpottingTransformer(**config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded  (val_acc at training: {ckpt['val_acc']:.4f})")

    # Build calibration dataloader (use validation split)
    calib_ds = SpeechCommandsDataset(DATA_ROOT, "validation", augment=False)
    calib_dl = DataLoader(
        calib_ds, batch_size=CALIB_BATCH_SZ, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )

    # ── Step 1: SmoothQuant calibration ──────────────────────────────────────
    print("\n[1/6] SmoothQuant calibration...")
    calibrator = ActivationCalibrator(model)
    sq_scales  = calibrator.run(calib_dl, CALIB_BATCHES)
    print(f"  Collected scales for {len(sq_scales)} Linear layers")

    # ── Step 2: Apply SmoothQuant ─────────────────────────────────────────────
    print("[2/6] Applying SmoothQuant (alpha={})...".format(SQ_ALPHA))
    model_sq = apply_smooth_quant(model, sq_scales, alpha=SQ_ALPHA)

    # ── Step 3: AWQ calibration on smooth model ───────────────────────────────
    print("[3/6] AWQ calibration on smooth model...")
    calibrator_awq = ActivationCalibrator(model_sq)
    awq_act_scales = calibrator_awq.run(calib_dl, CALIB_BATCHES)

    print("[4/6] Computing AWQ scales (grid={})...".format(AWQ_N_GRID))
    awq_scales = compute_awq_scales(model_sq, awq_act_scales)

    # ── Step 4: Apply AWQ ─────────────────────────────────────────────────────
    print("[4/6] Applying AWQ weight transformations...")
    model_final = apply_awq_scales(model_sq, awq_scales)

    # Quick sanity check: the transformed model should match the original
    val_dl = DataLoader(calib_ds, batch_size=CALIB_BATCH_SZ, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    acc_orig  = eval_pytorch(model,       val_dl, n_batches=20)
    acc_final = eval_pytorch(model_final, val_dl, n_batches=20)
    print(f"  FP32 accuracy  original={acc_orig:.4f}  after SQ+AWQ={acc_final:.4f}")
    assert abs(acc_orig - acc_final) < 0.005, (
        "SmoothQuant/AWQ broke equivalence — check transformations"
    )

    # ── Step 5: Export to ONNX ────────────────────────────────────────────────
    print("\n[5/6] Exporting to ONNX (opset 17)...")
    export_onnx(model_final, ONNX_SMOOTH_PATH)

    # ── Step 6: Static INT8 quantization ─────────────────────────────────────
    print("[6/6] Running ONNX Runtime static INT8 quantization...")
    quant_dl = DataLoader(
        calib_ds, batch_size=CALIB_BATCH_SZ, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    quantize_onnx_int8(ONNX_SMOOTH_PATH, ONNX_QUANT_PATH, quant_dl)

    # ── Accuracy comparison ───────────────────────────────────────────────────
    print("\nValidating final models (30 batches each)...")
    acc_fp32 = eval_onnx(ONNX_SMOOTH_PATH, val_dl, n_batches=30)
    acc_int8 = eval_onnx(ONNX_QUANT_PATH,  val_dl, n_batches=30)
    print(f"  FP32 ONNX : {acc_fp32:.4f}")
    print(f"  INT8 ONNX : {acc_int8:.4f}")
    print(f"  Accuracy drop: {acc_fp32 - acc_int8:.4f}")

    # ── Save preprocessing config ─────────────────────────────────────────────
    with open(CONFIG_PATH, "w") as f:
        json.dump(
            {
                "sample_rate": SAMPLE_RATE,
                "n_fft": N_FFT,
                "win_length": WIN_LENGTH,
                "hop_length": HOP_LENGTH,
                "n_mels": N_MELS,
                "max_frames": MAX_FRAMES,
                "keywords": KEYWORDS,
            },
            f,
            indent=2,
        )

    print(f"\nFiles ready for Raspberry Pi 3:")
    print(f"  {ONNX_QUANT_PATH}")
    print(f"  {CONFIG_PATH}")
    print(
        "\nCopy both files to the Pi, then:\n"
        "  python inference_rpi.py --model keyword_spotting_int8.onnx "
        "--config preprocess_config.json --mode realtime"
    )


if __name__ == "__main__":
    main()
