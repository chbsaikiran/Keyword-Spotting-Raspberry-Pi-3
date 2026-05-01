"""
Evaluate a quantized TFLite model on the Speech Commands v2 validation set.

Usage:
    python evaluate.py --model checkpoints/keyword_spotting_awq.tflite
    python evaluate.py --model checkpoints/keyword_spotting_smoothquant.tflite
    python evaluate.py --model checkpoints/keyword_spotting_combined.tflite
"""

import argparse

import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import (
    DATA_ROOT,
    SpeechCommandsDataset,
    collate_fn,
)


def evaluate(model_path: str, batch_size: int = 32, num_workers: int = 4) -> float:
    interp = tf.lite.Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    val_ds = SpeechCommandsDataset(DATA_ROOT, "validation", augment=False)
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    correct = total = 0
    for mel, labels in tqdm(val_dl, desc="Evaluating"):
        for i in range(mel.shape[0]):
            interp.set_tensor(inp_idx, mel[i : i + 1].numpy())
            interp.invoke()
            logits = interp.get_tensor(out_idx)[0]
            correct += int(logits.argmax() == labels[i].item())
            total += 1

    return correct / total


def main():
    p = argparse.ArgumentParser(description="Evaluate quantized TFLite model on validation set")
    p.add_argument("--model",       required=True, help="Path to .tflite model")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    print(f"Model : {args.model}")
    inp = tf.lite.Interpreter(model_path=args.model).get_input_details()
    print(f"Input : {inp[0]['shape']}  dtype={inp[0]['dtype'].__name__}")

    acc = evaluate(args.model, args.batch_size, args.num_workers)
    print(f"Val accuracy: {acc:.4f}  ({acc:.2%})")


if __name__ == "__main__":
    main()
