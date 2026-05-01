"""
Post-training quantization pipeline:
  1. SmoothQuant  — equalize activation/weight ranges via per-channel scaling
  2. AWQ          — activation-aware weight quantization (protect salient channels)
  3. TFLite INT8 export via PT2E quantization (litert-torch)

Run this on your training machine (GPU or fast CPU), then copy
  checkpoints/keyword_spotting_combined.tflite
  checkpoints/preprocess_config.json
to the Raspberry Pi.
"""

import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import litert_torch as ltt
import tensorflow as tf
from ai_edge_quantizer import quantizer as aq

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
TFLITE_PATH     = f"{CKPT_DIR}/keyword_spotting_combined.tflite"
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
        def hook(_module, inputs, _output):
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


# ── TFLite INT8 export ────────────────────────────────────────────────────────

def export_tflite(model: nn.Module, tflite_path: str):
    """SQ+AWQ-transformed PyTorch model → TFLite INT8 via litert-torch + ai_edge_quantizer."""
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

    # ── Step 5: Export to TFLite INT8 ────────────────────────────────────────
    print("\n[5/5] Exporting to TFLite INT8 (PT2E + litert-torch)...")
    export_tflite(model_final, TFLITE_PATH)

    # ── Accuracy comparison ───────────────────────────────────────────────────
    print("\nValidating final model (30 batches each)...")
    acc_fp32   = eval_pytorch(model_final, val_dl, n_batches=30)
    acc_tflite = eval_tflite(TFLITE_PATH, val_dl, n=30)
    print(f"  FP32  (SQ+AWQ) : {acc_fp32:.4f}")
    print(f"  INT8  TFLite   : {acc_tflite:.4f}")
    print(f"  Accuracy drop  : {acc_fp32 - acc_tflite:.4f}")

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
    print(f"  {TFLITE_PATH}")
    print(f"  {CONFIG_PATH}")
    print(
        "\nCopy both files to the Pi, then:\n"
        "  python inference_rpi.py --mode realtime"
    )


if __name__ == "__main__":
    main()
