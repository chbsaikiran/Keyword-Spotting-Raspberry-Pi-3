"""
AWQ (Activation-aware Weight Quantization) — standalone quantization pipeline.

Steps:
  1. Calibrate per-channel activation importance on the validation set
  2. Grid-search the optimal per-channel scale alpha for each Linear layer
       scale_j = act_max_j ^ alpha   minimises simulated INT8 weight error
  3. Fold scales into (LayerNorm, Linear) pairs — mathematically lossless
  4. Export transformed model to ONNX (opset 17)
  5. ONNX Runtime static INT8 quantization (QDQ, per-channel, reduce_range)
  6. Accuracy comparison FP32 vs INT8

Key idea: AWQ protects "salient" weight channels (those whose input activations
are large and thus have outsized impact on the output) by searching for the
per-channel scale that minimises the quantisation error specifically for those
channels.  All scaling is folded into adjacent LayerNorm parameters so the
ONNX graph is unchanged in structure.

Output files (copy both to Raspberry Pi):
    checkpoints/keyword_spotting_awq_int8.onnx
    checkpoints/preprocess_config.json
"""

import copy
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
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
ONNX_FP32_PATH = f"{CKPT_DIR}/keyword_spotting_awq_fp32.onnx"
ONNX_INT8_PATH = f"{CKPT_DIR}/keyword_spotting_awq_int8.onnx"
CONFIG_PATH    = f"{CKPT_DIR}/preprocess_config.json"

CALIB_BATCHES  = 64
CALIB_BATCH_SZ = 32
AWQ_N_BITS     = 8    # target quantisation bit-width
AWQ_N_GRID     = 20   # grid-search resolution for alpha ∈ [0, 1]


# ── Activation calibration ────────────────────────────────────────────────────

class ActivationCalibrator:
    """
    Attach forward hooks to every nn.Linear and record the per-input-channel
    maximum absolute activation across the entire calibration set.

    These maxima are used as a proxy for "activation importance": channels
    with large activations have large influence on the output and are therefore
    the most damaging to quantise naively.
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
            x = inputs[0].detach().float()
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


# ── AWQ scale search ──────────────────────────────────────────────────────────

def _pseudo_quantize(w: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Simulate symmetric per-output-channel INT-n quantisation.

    For each output-channel row  w[i, :]:
        scale_i = max(|w[i,:]|) / (2^(n_bits-1) - 1)
        w_q     = round(w / scale_i) * scale_i
    """
    q_max  = 2 ** (n_bits - 1) - 1
    scale  = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / q_max
    w_int  = (w / scale).round().clamp(-q_max - 1, q_max)
    return w_int * scale


def _find_awq_scale(
    w: torch.Tensor,
    act_max: torch.Tensor,
    n_bits: int = AWQ_N_BITS,
    n_grid: int = AWQ_N_GRID,
) -> torch.Tensor:
    """
    Grid-search for the per-input-channel AWQ scale.

    For each alpha in {0/n_grid, 1/n_grid, …, n_grid/n_grid}:

        scale_j = act_max_j ^ alpha          (importance-based per-channel scale)
        scale   = scale / mean(scale)        (normalise so overall magnitude is preserved)

        W_scaled[:, j] = W[:, j] / scale_j  (scale down the weight channel)
        error = || W - Q(W_scaled) * scale || (how much does quantisation hurt?)

    The alpha that gives the lowest error is chosen.

    Intuition
    ---------
    - alpha=0  →  no scaling  →  standard uniform quantisation
    - alpha=1  →  scale = act_max  →  aggressively equalises weight column norms
    - The optimum is usually around 0.5 and varies per layer

    Returns
    -------
    scale : [in_features] float tensor (normalised, clipped ≥ 1e-8)
    """
    act_max = act_max.clamp(min=1e-8)
    best_alpha = 0.5
    best_err   = float("inf")

    for step in range(n_grid + 1):
        alpha = step / n_grid
        scale = (act_max ** alpha).clamp(min=1e-8)
        scale = scale / scale.mean()                   # normalise

        w_scaled = w / scale.unsqueeze(0)              # [out, in]
        w_back   = _pseudo_quantize(w_scaled, n_bits) * scale.unsqueeze(0)
        err      = (w - w_back).pow(2).mean().item()

        if err < best_err:
            best_err   = err
            best_alpha = alpha

    scale = (act_max ** best_alpha).clamp(min=1e-8)
    return scale / scale.mean()


def compute_awq_scales(
    model: KeywordSpottingTransformer,
    act_scales: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Compute the optimal AWQ per-input-channel scale for the two Linear layers
    in each decoder block that are most sensitive to quantisation:
        blocks.i.attn.qkv   — receives LayerNorm output before attention
        blocks.i.ff.net.0   — receives LayerNorm output before FFN
    """
    awq_scales: dict[str, torch.Tensor] = {}
    for i, block in enumerate(model.blocks):
        for key, linear in [
            (f"blocks.{i}.attn.qkv", block.attn.qkv),
            (f"blocks.{i}.ff.net.0", block.ff.net[0]),
        ]:
            if key not in act_scales:
                continue
            print(f"  AWQ grid-search: {key} ...", end=" ", flush=True)
            awq_scales[key] = _find_awq_scale(
                linear.weight.data.clone(),
                act_scales[key],
            )
            print("done")
    return awq_scales


# ── Apply AWQ scales ──────────────────────────────────────────────────────────

def apply_awq_scales(
    model: KeywordSpottingTransformer,
    awq_scales: dict[str, torch.Tensor],
) -> KeywordSpottingTransformer:
    """
    Fold AWQ input-channel scales into adjacent (LayerNorm, Linear) pairs.

    Transformation for scale  s  (shape: [in_features]):
        linear.weight[:, j] /= s[j]   →  weight column ranges reduced
        norm.weight[j]       *= s[j]   →  absorbs the inverse scale
        norm.bias[j]         *= s[j]   →  (if present)

    Equivalence proof:
        output = linear(norm(x))
               = W · (gamma ⊙ x_norm + beta)
        After folding s:
               = (W / s) · ((gamma * s) ⊙ x_norm + (beta * s))
               = W · (gamma ⊙ x_norm + beta)   ✓
    """
    model = copy.deepcopy(model)
    for i, block in enumerate(model.blocks):
        for key, norm, linear in [
            (f"blocks.{i}.attn.qkv", block.norm1, block.attn.qkv),
            (f"blocks.{i}.ff.net.0", block.norm2, block.ff.net[0]),
        ]:
            if key not in awq_scales:
                continue
            s = awq_scales[key]
            linear.weight.data.div_(s.unsqueeze(0))
            norm.weight.data.mul_(s)
            if norm.bias is not None:
                norm.bias.data.mul_(s)
    return model


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, MAX_FRAMES, N_MELS)
    torch.onnx.export(
        model, dummy, path,
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(path))
    print(f"  Exported: {path}  ({os.path.getsize(path)/1e6:.2f} MB)")


# ── ONNX Runtime static INT8 quantization ────────────────────────────────────

class SpeechCalibReader(CalibrationDataReader):
    def __init__(self, dataloader: DataLoader, n_batches: int):
        self._iter  = iter(dataloader)
        self._max   = n_batches
        self._count = 0

    def get_next(self):
        if self._count >= self._max:
            return None
        try:
            mel, _ = next(self._iter)
            self._count += 1
            return {"mel": mel.numpy()}
        except StopIteration:
            return None


def quantize_onnx(fp32_path: str, int8_path: str, dataloader: DataLoader):
    """
    Static INT8 quantization via ONNX Runtime.

    After AWQ scaling, weight column norms are more uniform, so per-channel
    INT8 quantisation (the ONNX Runtime default) has less rounding error than
    it would on the raw un-scaled model.

    reduce_range=True avoids ARM NEON 8-bit multiply accumulation overflow
    on Raspberry Pi 3.
    """
    reader = SpeechCalibReader(dataloader, CALIB_BATCHES)
    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "EnableSubgraph": False,
        },
    )
    print(f"  Quantized: {int8_path}  ({os.path.getsize(int8_path)/1e6:.2f} MB)")


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


def eval_onnx(path: str, dataloader: DataLoader, n: int = 30) -> float:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess    = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    correct = total = 0
    for i, (mel, labels) in enumerate(dataloader):
        if i >= n:
            break
        logits   = sess.run(None, {"mel": mel.numpy()})[0]
        preds    = logits.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total   += labels.shape[0]
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
    print(f"\n[1/4] Calibrating activation importance ({CALIB_BATCHES} batches)...")
    calibrator = ActivationCalibrator(model)
    act_scales = calibrator.run(calib_dl, CALIB_BATCHES)
    print(f"  Collected scales for {len(act_scales)} Linear layers")

    # ── Step 2: AWQ scale search ──────────────────────────────────────────────
    print(f"[2/4] Searching AWQ scales (grid={AWQ_N_GRID}, bits={AWQ_N_BITS})...")
    awq_scales = compute_awq_scales(model, act_scales)

    # ── Step 3: Apply AWQ ─────────────────────────────────────────────────────
    print("[3/4] Applying AWQ weight transformations...")
    model_awq = apply_awq_scales(model, awq_scales)

    # Sanity check: transformed model must match original accuracy
    acc_orig = eval_pytorch(model,     val_dl, n=20)
    acc_awq  = eval_pytorch(model_awq, val_dl, n=20)
    print(f"  FP32 accuracy  original={acc_orig:.4f}  after AWQ={acc_awq:.4f}")
    assert abs(acc_orig - acc_awq) < 0.005, (
        "AWQ broke mathematical equivalence — check apply_awq_scales()"
    )

    # ── Step 4: Export to ONNX ────────────────────────────────────────────────
    print("[4/4] Exporting to ONNX (opset 17)...")
    export_onnx(model_awq, ONNX_FP32_PATH)

    # ── Step 5: Static INT8 quantization ─────────────────────────────────────
    print("Running ONNX Runtime static INT8 quantization...")
    quant_dl = DataLoader(ds, CALIB_BATCH_SZ, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)
    quantize_onnx(ONNX_FP32_PATH, ONNX_INT8_PATH, quant_dl)

    # ── Accuracy comparison ───────────────────────────────────────────────────
    print("\nValidating (30 batches each)...")
    acc_fp32 = eval_onnx(ONNX_FP32_PATH, val_dl, n=30)
    acc_int8 = eval_onnx(ONNX_INT8_PATH, val_dl, n=30)
    print(f"  FP32 ONNX (AWQ) : {acc_fp32:.4f}")
    print(f"  INT8 ONNX (AWQ) : {acc_int8:.4f}")
    print(f"  Accuracy drop   : {acc_fp32 - acc_int8:.4f}")

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
    print(f"  {ONNX_INT8_PATH}")
    print(f"  {CONFIG_PATH}")
    print(
        "\nRun on Pi:\n"
        "  python inference_rpi_awq.py --mode realtime"
    )


if __name__ == "__main__":
    main()
