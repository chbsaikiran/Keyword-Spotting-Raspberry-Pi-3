"""
Keyword spotting inference for Raspberry Pi 3 — AWQ TFLite model.

Copy these files from your training machine:
    keyword_spotting_awq.tflite
    preprocess_config.json

Install on the Pi:
    pip install -r requirements_rpi.txt

Usage:
    python inference_rpi_awq.py --mode realtime
    python inference_rpi_awq.py --mode file --file clip.wav
    python inference_rpi_awq.py --mode benchmark
"""

import argparse
import json
import os
import re
import time

import numpy as np
import tflite_runtime.interpreter as tflite

try:
    import sounddevice as sd
    _HAS_SD = True
except (ImportError, OSError):
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

DEFAULT_MODEL  = "checkpoints/keyword_spotting_awq.tflite"
DEFAULT_CONFIG = "checkpoints/preprocess_config.json"


# ── Mel spectrogram ───────────────────────────────────────────────────────────

def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    lo_mel  = 2595.0 * np.log10(1.0 + 0.0        / 700.0)
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
    sr, n_fft = cfg["sample_rate"], cfg["n_fft"]
    hop, win  = cfg["hop_length"], cfg["win_length"]
    n_mels, max_f = cfg["n_mels"], cfg["max_frames"]

    if _HAS_LIBROSA:
        mel     = librosa.feature.melspectrogram(
            y=wave, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win, n_mels=n_mels,
        )
        log_mel = np.log(mel + 1e-6).T
    else:
        power   = _stft_numpy(wave, n_fft, hop, win)
        fb      = _mel_filterbank(sr, n_fft, n_mels)
        log_mel = np.log(fb @ power + 1e-6).T

    T = log_mel.shape[0]
    if T < max_f:
        log_mel = np.vstack([log_mel, np.zeros((max_f - T, n_mels), dtype=np.float32)])
    else:
        log_mel = log_mel[:max_f]

    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
    return log_mel.astype(np.float32)


# ── Audio loading ─────────────────────────────────────────────────────────────

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


# ── TFLite inference engine ───────────────────────────────────────────────────

class AWQSpotter:
    """Keyword spotter using the AWQ TFLite INT8 model."""

    def __init__(self, model_path: str, config_path: str):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.keywords = self.cfg["keywords"]

        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=4,
        )
        self.interpreter.allocate_tensors()

        n_mels, max_frames = self.cfg["n_mels"], self.cfg["max_frames"]

        input_details  = self.interpreter.get_input_details()
        self._out_idx  = self.interpreter.get_output_details()[0]["index"]

        # First declared input is always the mel spectrogram.
        self._inp_idx      = input_details[0]["index"]
        self._extra_inputs = []  # [(tensor_index, numpy_data), ...]

        # Any additional declared inputs (shape != mel) are extra parameters.
        mel_shape = [1, max_frames, n_mels]
        for det in input_details[1:]:
            shape = [int(x) for x in det["shape"]]
            self._extra_inputs.append((det["index"],
                                       self._make_cls_data(shape, det["dtype"])))

        # Build a lookup from index → tensor detail for the probe below.
        td_by_idx: dict = {}
        try:
            for td in self.interpreter.get_tensor_details():
                td_by_idx[td["index"]] = td
        except AttributeError:
            pass

        # ── Probe invoke: discover weight tensors that the C++ runtime        ──
        # ── requires but tflite_runtime doesn't surface via                  ──
        # ── get_input_details().  For each one we force-allocate it with     ──
        # ── resize_tensor_input(), then set its saved INT8 data from config. ──
        dummy = np.zeros([1, max_frames, n_mels], dtype=np.float32)
        for _attempt in range(20):
            self.interpreter.set_tensor(self._inp_idx, dummy)
            for idx, data in self._extra_inputs:
                self.interpreter.set_tensor(idx, data)
            try:
                self.interpreter.invoke()
                break
            except RuntimeError as exc:
                m = re.search(r'Input tensor (\d+) lacks data', str(exc))
                if not m:
                    raise
                missing = int(m.group(1))
                td = td_by_idx.get(missing)
                if td is None:
                    raise RuntimeError(
                        f"Input tensor {missing} lacks data and is not in "
                        f"get_tensor_details(). Re-run quantize_awq.py."
                    ) from exc
                shape = [int(x) for x in td["shape"]]
                dtype = td["dtype"]
                weight = self._load_hidden_weight(missing, shape, dtype)

                # Force-allocate by trying each input position (position 0 is
                # the mel input; hidden weights are at positions 1, 2, ...).
                allocated = False
                for pos in range(1, 10):
                    try:
                        self.interpreter.resize_tensor_input(pos, shape)
                        self.interpreter.allocate_tensors()
                        self.interpreter.set_tensor(missing, weight)
                        allocated = True
                        self._extra_inputs.append((missing, weight))
                        print(f"  [AWQ] Allocated hidden weight: idx={missing} "
                              f"shape={shape} dtype={dtype.__name__} "
                              f"at input pos {pos}")
                        break
                    except Exception:
                        continue
                if not allocated:
                    raise RuntimeError(
                        f"Input tensor {missing} (shape={shape}) cannot be "
                        f"force-allocated.\nRe-run quantize_awq.py on the "
                        f"training machine, copy checkpoints/ to the Pi, "
                        f"then retry."
                    ) from exc

        print(f"[AWQ] Loaded: {model_path}")
        print(f"  Input shape  : {mel_shape}")
        print(f"  Extra inputs : {len(self._extra_inputs)}")
        print(f"  Classes ({len(self.keywords)}): {', '.join(self.keywords)}")

    def _make_cls_data(self, shape: list, dtype) -> np.ndarray:
        """Return the cls_token array (from config) or zeros, cast to dtype."""
        n = int(np.prod(shape)) if shape else 1
        if "cls_token" in self.cfg:
            flat = np.array(self.cfg["cls_token"], dtype=np.float32).flatten()
            if flat.size == n:
                return flat.reshape(shape).astype(dtype)
        return np.zeros(shape, dtype=dtype)

    def _load_hidden_weight(self, tensor_idx: int, shape: list, dtype) -> np.ndarray:
        """Return saved INT8 weight data from config, or zeros as fallback."""
        hw = self.cfg.get("hidden_weights", {}).get(str(tensor_idx))
        if hw is not None:
            return np.array(hw["data"], dtype=dtype).reshape(shape)
        print(f"  [AWQ] Warning: no saved data for tensor {tensor_idx} in config "
              f"— using zeros (re-run quantize_awq.py for correct weights).")
        return np.zeros(shape, dtype=dtype)

    def predict(self, wave: np.ndarray) -> tuple:
        t0      = time.perf_counter()
        log_mel = compute_log_mel(wave, self.cfg)
        inp     = log_mel[np.newaxis].astype(np.float32)

        self.interpreter.set_tensor(self._inp_idx, inp)
        for idx, data in self._extra_inputs:
            self.interpreter.set_tensor(idx, data)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self._out_idx)[0]

        e     = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx   = int(probs.argmax())
        lat   = (time.perf_counter() - t0) * 1_000.0
        return self.keywords[idx], float(probs[idx]), lat

    def predict_file(self, path: str) -> tuple:
        wave = load_audio(path, self.cfg["sample_rate"])
        return self.predict(wave)


# ── Inference modes ───────────────────────────────────────────────────────────

def run_realtime(spotter: AWQSpotter, threshold: float):
    if not _HAS_SD:
        print("sounddevice not installed.  Run:  pip install sounddevice")
        return
    sr = spotter.cfg["sample_rate"]
    print(f"\n[AWQ] Listening… (Ctrl+C to stop, threshold={threshold:.0%})")
    print("-" * 60)
    try:
        while True:
            audio = sd.rec(sr, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            kw, conf, lat = spotter.predict(audio.flatten())
            bar = "█" * int(conf * 20)
            tag = ">>> DETECTED" if conf >= threshold else "    silence "
            print(f"{tag}  {kw:<12s}  {conf:5.1%}  |{bar:<20s}|  {lat:5.1f} ms")
    except KeyboardInterrupt:
        print("\nStopped.")


def run_file(spotter: AWQSpotter, path: str):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    kw, conf, lat = spotter.predict_file(path)
    print(f"[AWQ]  {path}")
    print(f"  Predicted : {kw}  ({conf:.2%})  in {lat:.1f} ms")


def run_benchmark(spotter: AWQSpotter, n_runs: int = 100):
    silence = np.zeros(spotter.cfg["sample_rate"], dtype=np.float32)
    for _ in range(10):
        spotter.predict(silence)
    lats = np.array([spotter.predict(silence)[2] for _ in range(n_runs)])
    print(f"\n[AWQ] Benchmark ({n_runs} runs):")
    print(f"  Mean   : {lats.mean():.1f} ms")
    print(f"  Median : {np.median(lats):.1f} ms")
    print(f"  P95    : {np.percentile(lats, 95):.1f} ms")
    print(f"  P99    : {np.percentile(lats, 99):.1f} ms")
    print(f"  Min    : {lats.min():.1f} ms  |  Max: {lats.max():.1f} ms")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Keyword spotting — AWQ TFLite — RPi3")
    p.add_argument("--model",     default=DEFAULT_MODEL)
    p.add_argument("--config",    default=DEFAULT_CONFIG)
    p.add_argument("--mode",      default="realtime",
                   choices=["realtime", "file", "benchmark"])
    p.add_argument("--file",      default=None)
    p.add_argument("--threshold", type=float, default=0.70)
    args = p.parse_args()

    for f in (args.model, args.config):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Not found: {f}\nCopy from training machine.")

    spotter = AWQSpotter(args.model, args.config)

    if args.mode == "benchmark":
        run_benchmark(spotter)
    elif args.mode == "file":
        run_file(spotter, args.file) if args.file else print("Provide --file <path>")
    else:
        run_realtime(spotter, args.threshold)


if __name__ == "__main__":
    main()
