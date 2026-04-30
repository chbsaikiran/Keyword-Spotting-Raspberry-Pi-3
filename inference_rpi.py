"""
Keyword spotting inference for Raspberry Pi 3.
Dependencies: onnxruntime, numpy, librosa, sounddevice, soundfile

Copy these files from your training machine:
    keyword_spotting_int8.onnx
    preprocess_config.json

Install on the Pi:
    pip install -r requirements_rpi.txt

Usage:
    python inference_rpi.py --mode realtime        # microphone
    python inference_rpi.py --mode file --file clip.wav
    python inference_rpi.py --mode benchmark
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


# ── Mel spectrogram (numpy / librosa) ────────────────────────────────────────

def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a [n_mels, n_fft//2+1] triangular mel filterbank."""
    lo_mel  = 2595.0 * np.log10(1.0 + 0.0   / 700.0)
    hi_mel  = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
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


def _stft_numpy(
    waveform: np.ndarray, n_fft: int, hop: int, win_len: int
) -> np.ndarray:
    """Power spectrum via a simple numpy STFT (fallback when librosa absent)."""
    from scipy.signal import get_window
    window = get_window("hann", win_len).astype(np.float32)
    pad    = n_fft - win_len
    window = np.pad(window, (pad // 2, pad - pad // 2))

    n_frames = max(1, (len(waveform) - n_fft) // hop + 1)
    power    = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame      = waveform[i * hop: i * hop + n_fft] * window
        spectrum   = np.fft.rfft(frame, n=n_fft)
        power[:, i] = (spectrum.real ** 2 + spectrum.imag ** 2)
    return power


def compute_log_mel(waveform: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Compute log-mel spectrogram matching the training transform.

    Parameters
    ----------
    waveform : 1-D float32 array at cfg["sample_rate"]
    cfg      : preprocess_config.json as a dict

    Returns
    -------
    [max_frames, n_mels] float32 array (normalised)
    """
    sr, n_fft, hop, win = (
        cfg["sample_rate"], cfg["n_fft"],
        cfg["hop_length"],  cfg["win_length"],
    )
    n_mels     = cfg["n_mels"]
    max_frames = cfg["max_frames"]

    if _HAS_LIBROSA:
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=n_fft,
            hop_length=hop, win_length=win, n_mels=n_mels,
        )
        log_mel = np.log(mel + 1e-6).T          # [T, n_mels]
    else:
        power   = _stft_numpy(waveform, n_fft, hop, win)
        fb      = _mel_filterbank(sr, n_fft, n_mels)
        mel     = fb @ power                     # [n_mels, T]
        log_mel = np.log(mel + 1e-6).T           # [T, n_mels]

    # Pad or trim to max_frames
    T = log_mel.shape[0]
    if T < max_frames:
        log_mel = np.vstack(
            [log_mel, np.zeros((max_frames - T, n_mels), dtype=np.float32)]
        )
    else:
        log_mel = log_mel[:max_frames]

    # Per-sample normalisation (matches training)
    mean    = log_mel.mean()
    std     = log_mel.std() + 1e-5
    log_mel = (log_mel - mean) / std

    return log_mel.astype(np.float32)


# ── Utility: load audio from file ────────────────────────────────────────────

def _resample_linear(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wave
    ratio      = target_sr / orig_sr
    target_len = int(len(wave) * ratio)
    x_old = np.linspace(0.0, len(wave) - 1, len(wave))
    x_new = np.linspace(0.0, len(wave) - 1, target_len)
    return np.interp(x_new, x_old, wave).astype(np.float32)


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load any wav/flac file, return mono float32 at target_sr."""
    if _HAS_SF:
        wave, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
    else:
        from scipy.io import wavfile
        sr, wave = wavfile.read(path)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave = wave.astype(np.float32)
        if wave.max() > 1.5:       # raw PCM int16 range
            wave /= 32768.0

    wave = _resample_linear(wave, sr, target_sr)

    # Pad / trim to 1 second
    n = target_sr
    if len(wave) < n:
        wave = np.pad(wave, (0, n - len(wave)))
    else:
        wave = wave[:n]
    return wave


# ── ONNX keyword spotter ──────────────────────────────────────────────────────

class KeywordSpotter:
    def __init__(self, model_path: str, config_path: str):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.keywords = self.cfg["keywords"]

        opts = ort.SessionOptions()
        # RPi3 has 4 cores — give all to intra-op parallelism
        opts.intra_op_num_threads          = 4
        opts.inter_op_num_threads          = 1
        opts.execution_mode                = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level      = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern            = True
        opts.enable_cpu_mem_arena          = True

        self.sess = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        print(f"Loaded: {model_path}")
        print(f"Classes ({len(self.keywords)}): {', '.join(self.keywords)}")

    def predict(self, waveform: np.ndarray) -> tuple[str, float, float]:
        """
        Parameters
        ----------
        waveform : mono float32 array at cfg["sample_rate"]

        Returns
        -------
        (keyword, confidence, latency_ms)
        """
        t0      = time.perf_counter()
        log_mel = compute_log_mel(waveform, self.cfg)        # [T, n_mels]
        inp     = log_mel[np.newaxis]                        # [1, T, n_mels]
        logits  = self.sess.run(None, {"mel": inp})[0][0]   # [n_classes]

        # Numerically stable softmax
        e        = np.exp(logits - logits.max())
        probs    = e / e.sum()
        idx      = int(probs.argmax())
        lat_ms   = (time.perf_counter() - t0) * 1_000.0

        return self.keywords[idx], float(probs[idx]), lat_ms

    def predict_file(self, path: str) -> tuple[str, float, float]:
        wave = load_audio(path, self.cfg["sample_rate"])
        return self.predict(wave)


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_realtime(spotter: KeywordSpotter, threshold: float):
    if not _HAS_SD:
        print("sounddevice not installed — cannot run real-time mode.")
        print("Install it with:  pip install sounddevice")
        return

    sr       = spotter.cfg["sample_rate"]
    duration = 1.0          # 1-second windows
    n_samples = int(sr * duration)

    print(f"\nListening… (Ctrl+C to stop, threshold={threshold:.0%})")
    print("-" * 55)
    try:
        while True:
            audio = sd.rec(n_samples, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            wave = audio.flatten()

            keyword, conf, lat = spotter.predict(wave)
            bar  = "█" * int(conf * 20)
            tag  = ">>> DETECTED" if conf >= threshold else "    silence "
            print(f"{tag}  {keyword:<12s}  {conf:5.1%}  |{bar:<20s}|  {lat:5.1f} ms")

    except KeyboardInterrupt:
        print("\nStopped.")


def run_file(spotter: KeywordSpotter, path: str):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    keyword, conf, lat = spotter.predict_file(path)
    print(f"File     : {path}")
    print(f"Predicted: {keyword}  ({conf:.2%})  in {lat:.1f} ms")


def run_benchmark(spotter: KeywordSpotter, n_runs: int = 100):
    sr      = spotter.cfg["sample_rate"]
    silence = np.zeros(sr, dtype=np.float32)

    # Warmup (JIT compilation, cache warm)
    for _ in range(10):
        spotter.predict(silence)

    lats = []
    for _ in range(n_runs):
        _, _, lat = spotter.predict(silence)
        lats.append(lat)

    lats = np.array(lats)
    print(f"\nBenchmark on {n_runs} silent clips:")
    print(f"  Mean   : {lats.mean():.1f} ms")
    print(f"  Median : {np.median(lats):.1f} ms")
    print(f"  P95    : {np.percentile(lats, 95):.1f} ms")
    print(f"  P99    : {np.percentile(lats, 99):.1f} ms")
    print(f"  Min    : {lats.min():.1f} ms")
    print(f"  Max    : {lats.max():.1f} ms")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Keyword spotting — Raspberry Pi 3")
    p.add_argument("--model",     default="keyword_spotting_int8.onnx",
                   help="Path to INT8 ONNX model")
    p.add_argument("--config",    default="preprocess_config.json",
                   help="Path to preprocessing config JSON")
    p.add_argument("--mode",      default="realtime",
                   choices=["realtime", "file", "benchmark"],
                   help="Inference mode")
    p.add_argument("--file",      default=None,
                   help="Audio file path (mode=file only)")
    p.add_argument("--threshold", type=float, default=0.70,
                   help="Detection confidence threshold (default 0.70)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Copy keyword_spotting_int8.onnx from the training machine."
        )
    if not os.path.isfile(args.config):
        raise FileNotFoundError(
            f"Config not found: {args.config}\n"
            "Copy preprocess_config.json from the training machine."
        )

    spotter = KeywordSpotter(args.model, args.config)

    if args.mode == "benchmark":
        run_benchmark(spotter)
    elif args.mode == "file":
        if args.file is None:
            print("Specify --file <path> when using --mode file")
        else:
            run_file(spotter, args.file)
    else:
        run_realtime(spotter, args.threshold)


if __name__ == "__main__":
    main()
