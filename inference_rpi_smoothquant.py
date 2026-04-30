"""
Keyword spotting inference for Raspberry Pi 3 — SmoothQuant INT8 model.

Copy these files from your training machine:
    keyword_spotting_smoothquant_int8.onnx
    preprocess_config.json

Install on the Pi:
    pip install -r requirements_rpi.txt

Usage:
    python inference_rpi_smoothquant.py --mode realtime
    python inference_rpi_smoothquant.py --mode file --file clip.wav
    python inference_rpi_smoothquant.py --mode benchmark

SmoothQuant note:
    The weights in this model have been pre-scaled so that activation and
    weight quantisation ranges are balanced across channels.  This typically
    gives lower INT8 accuracy drop than unmodified static quantisation,
    especially for the LayerNorm → Linear transitions in the transformer.
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False


# ── Default paths ─────────────────────────────────────────────────────────────

DEFAULT_MODEL  = "keyword_spotting_smoothquant_int8.onnx"
DEFAULT_CONFIG = "preprocess_config.json"


# ── Mel spectrogram ───────────────────────────────────────────────────────────

def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    lo_mel  = 2595.0 * np.log10(1.0 + 0.0         / 700.0)
    hi_mel  = 2595.0 * np.log10(1.0 + (sr / 2.0)  / 700.0)
    pts_mel = np.linspace(lo_mel, hi_mel, n_mels + 2)
    pts_hz  = 700.0 * (10.0 ** (pts_mel / 2595.0) - 1.0)
    bins    = np.floor((n_fft + 1) * pts_hz / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            if mid != lo:
                fb[m - 1, k] = float(k - lo) / float(mid - lo)
        for k in range(mid, hi):
            if hi != mid:
                fb[m - 1, k] = float(hi - k) / float(hi - mid)
    return fb


def _stft_numpy(wave: np.ndarray, n_fft: int, hop: int, win_len: int) -> np.ndarray:
    from scipy.signal import get_window
    window = get_window("hann", win_len).astype(np.float32)
    pad    = n_fft - win_len
    window = np.pad(window, (pad // 2, pad - pad // 2))
    n_frames = max(1, (len(wave) - n_fft) // hop + 1)
    power    = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame       = wave[i * hop: i * hop + n_fft] * window
        s           = np.fft.rfft(frame, n=n_fft)
        power[:, i] = s.real ** 2 + s.imag ** 2
    return power


def compute_log_mel(wave: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Compute normalised log-mel spectrogram matching the training transform.
    Returns [max_frames, n_mels] float32.
    """
    sr, n_fft = cfg["sample_rate"], cfg["n_fft"]
    hop, win  = cfg["hop_length"], cfg["win_length"]
    n_mels    = cfg["n_mels"]
    max_f     = cfg["max_frames"]

    if _HAS_LIBROSA:
        mel     = librosa.feature.melspectrogram(
            y=wave, sr=sr, n_fft=n_fft, hop_length=hop,
            win_length=win, n_mels=n_mels,
        )
        log_mel = np.log(mel + 1e-6).T
    else:
        power   = _stft_numpy(wave, n_fft, hop, win)
        fb      = _mel_filterbank(sr, n_fft, n_mels)
        log_mel = np.log(fb @ power + 1e-6).T

    T = log_mel.shape[0]
    if T < max_f:
        log_mel = np.vstack(
            [log_mel, np.zeros((max_f - T, n_mels), dtype=np.float32)]
        )
    else:
        log_mel = log_mel[:max_f]

    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
    return log_mel.astype(np.float32)


# ── Audio file loading ────────────────────────────────────────────────────────

def _resample(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wave
    n    = int(len(wave) * target_sr / orig_sr)
    xold = np.linspace(0.0, len(wave) - 1, len(wave))
    xnew = np.linspace(0.0, len(wave) - 1, n)
    return np.interp(xnew, xold, wave).astype(np.float32)


def load_audio(path: str, target_sr: int) -> np.ndarray:
    if _HAS_SF:
        wave, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
    else:
        from scipy.io import wavfile
        sr, wave = wavfile.read(path)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave = wave.astype(np.float32)
        if wave.max() > 1.5:
            wave /= 32768.0

    wave = _resample(wave, sr, target_sr)
    n = target_sr
    if len(wave) < n:
        wave = np.pad(wave, (0, n - len(wave)))
    return wave[:n]


# ── ONNX inference engine ─────────────────────────────────────────────────────

class SmoothQuantSpotter:
    """
    Keyword spotter using a SmoothQuant INT8 ONNX model.

    SmoothQuant pre-scaled the weights so activation and weight ranges are
    balanced.  The INT8 quantisation error is therefore more uniformly
    distributed across channels, giving better accuracy than naive INT8.
    """

    def __init__(self, model_path: str, config_path: str):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.keywords = self.cfg["keywords"]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads     = 4   # RPi3 is quad-core
        opts.inter_op_num_threads     = 1
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern       = True
        opts.enable_cpu_mem_arena     = True

        self.sess = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        print(f"[SmoothQuant] Loaded: {model_path}")
        print(f"  Classes ({len(self.keywords)}): {', '.join(self.keywords)}")

    def predict(self, wave: np.ndarray) -> tuple[str, float, float]:
        """
        Parameters
        ----------
        wave : mono float32 at cfg["sample_rate"]

        Returns
        -------
        (keyword, confidence, latency_ms)
        """
        t0      = time.perf_counter()
        log_mel = compute_log_mel(wave, self.cfg)
        inp     = log_mel[np.newaxis]                       # [1, T, n_mels]
        logits  = self.sess.run(None, {"mel": inp})[0][0]  # [n_classes]
        e       = np.exp(logits - logits.max())
        probs   = e / e.sum()
        idx     = int(probs.argmax())
        lat     = (time.perf_counter() - t0) * 1_000.0
        return self.keywords[idx], float(probs[idx]), lat

    def predict_file(self, path: str) -> tuple[str, float, float]:
        wave = load_audio(path, self.cfg["sample_rate"])
        return self.predict(wave)


# ── Inference modes ───────────────────────────────────────────────────────────

def run_realtime(spotter: SmoothQuantSpotter, threshold: float):
    if not _HAS_SD:
        print("sounddevice not installed.  Run:  pip install sounddevice")
        return
    sr = spotter.cfg["sample_rate"]
    print(f"\n[SmoothQuant] Listening… (Ctrl+C to stop, threshold={threshold:.0%})")
    print("-" * 60)
    try:
        while True:
            audio = sd.rec(sr, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            wave = audio.flatten()
            kw, conf, lat = spotter.predict(wave)
            bar = "█" * int(conf * 20)
            tag = ">>> DETECTED" if conf >= threshold else "    silence "
            print(f"{tag}  {kw:<12s}  {conf:5.1%}  |{bar:<20s}|  {lat:5.1f} ms")
    except KeyboardInterrupt:
        print("\nStopped.")


def run_file(spotter: SmoothQuantSpotter, path: str):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    kw, conf, lat = spotter.predict_file(path)
    print(f"[SmoothQuant]")
    print(f"  File      : {path}")
    print(f"  Predicted : {kw}  ({conf:.2%})  in {lat:.1f} ms")


def run_benchmark(spotter: SmoothQuantSpotter, n_runs: int = 100):
    sr      = spotter.cfg["sample_rate"]
    silence = np.zeros(sr, dtype=np.float32)
    for _ in range(10):                        # warmup
        spotter.predict(silence)
    lats = [spotter.predict(silence)[2] for _ in range(n_runs)]
    a    = np.array(lats)
    print(f"\n[SmoothQuant] Benchmark on {n_runs} silent clips:")
    print(f"  Mean   : {a.mean():.1f} ms")
    print(f"  Median : {np.median(a):.1f} ms")
    print(f"  P95    : {np.percentile(a, 95):.1f} ms")
    print(f"  P99    : {np.percentile(a, 99):.1f} ms")
    print(f"  Min    : {a.min():.1f} ms  |  Max: {a.max():.1f} ms")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Keyword spotting — SmoothQuant INT8 — Raspberry Pi 3"
    )
    p.add_argument("--model",     default=DEFAULT_MODEL)
    p.add_argument("--config",    default=DEFAULT_CONFIG)
    p.add_argument("--mode",      default="realtime",
                   choices=["realtime", "file", "benchmark"])
    p.add_argument("--file",      default=None)
    p.add_argument("--threshold", type=float, default=0.70)
    args = p.parse_args()

    for fpath in (args.model, args.config):
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Not found: {fpath}\n"
                "Copy the model and config from the training machine."
            )

    spotter = SmoothQuantSpotter(args.model, args.config)

    if args.mode == "benchmark":
        run_benchmark(spotter)
    elif args.mode == "file":
        if args.file is None:
            print("Provide --file <path>")
        else:
            run_file(spotter, args.file)
    else:
        run_realtime(spotter, args.threshold)


if __name__ == "__main__":
    main()
