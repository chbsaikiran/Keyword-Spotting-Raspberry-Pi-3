"""
Evaluate a quantized ONNX model on the Speech Commands v2 validation set.

Usage:
    python evaluate.py --model checkpoints/keyword_spotting_awq_int8.onnx
    python evaluate.py --model checkpoints/keyword_spotting_smoothquant_int8.onnx
    python evaluate.py --model checkpoints/keyword_spotting_int8.onnx
"""

import argparse

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import (
    DATA_ROOT,
    SpeechCommandsDataset,
    collate_fn,
)


def evaluate(model_path: str, batch_size: int = 64, num_workers: int = 4) -> float:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads     = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    val_ds = SpeechCommandsDataset(DATA_ROOT, "validation", augment=False)
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    correct = total = 0
    for mel, labels in tqdm(val_dl, desc="Evaluating"):
        logits = sess.run(None, {"mel": mel.numpy()})[0]
        preds   = logits.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total   += labels.shape[0]

    return correct / total


def main():
    p = argparse.ArgumentParser(description="Evaluate quantized ONNX model on validation set")
    p.add_argument("--model",       required=True, help="Path to ONNX model")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    print(f"Model : {args.model}")
    acc = evaluate(args.model, args.batch_size, args.num_workers)
    print(f"Val accuracy: {acc:.4f}  ({acc:.2%})")


if __name__ == "__main__":
    main()
