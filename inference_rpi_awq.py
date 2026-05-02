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
import struct
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


def _find_inputs_vec_offset(buf: bytes) -> int:
    """
    Navigate the TFLite flatbuffer hierarchy to find subgraph[0].inputs vector.

    TFLite flatbuffer layout (FlatBuffers format):
      buf[0:4]  → uint32 absolute offset to root Model table
      Model.subgraphs  = field index 2 in Model's vtable
      SubGraph.inputs  = field index 1 in SubGraph's vtable

    Returns the absolute offset of the inputs vector's count field, or -1.
    """
    try:
        # Root table (Model) absolute position
        root = struct.unpack_from("<I", buf, 0)[0]

        # Each FlatBuffers table begins with a signed int32 (soffset_t) whose
        # value, added to the table position, gives the vtable position.
        vtbl = root + struct.unpack_from("<i", buf, root)[0]

        # vtable layout: [uint16 vtbl_bytes][uint16 obj_bytes][field0 uint16]...
        # Model.subgraphs is field index 2 → vtable entry at vtbl + 4 + 2*2
        sg_voff = struct.unpack_from("<H", buf, vtbl + 4 + 2 * 2)[0]
        if sg_voff == 0:
            return -1  # field absent

        # The field slot holds a uoffset_t (uint32) pointing forward to the vector
        sg_ptr = root + sg_voff
        sg_vec = sg_ptr + struct.unpack_from("<I", buf, sg_ptr)[0]

        # Vector of SubGraph tables: [uint32 count][uoffset_0][uoffset_1]...
        # Each uoffset is relative to its own 4-byte slot.
        if struct.unpack_from("<I", buf, sg_vec)[0] == 0:
            return -1  # no subgraphs
        sg0_slot = sg_vec + 4
        sg0 = sg0_slot + struct.unpack_from("<I", buf, sg0_slot)[0]

        # SubGraph[0] vtable
        sg_vtbl = sg0 + struct.unpack_from("<i", buf, sg0)[0]

        # SubGraph.inputs is field index 1 → vtable entry at sg_vtbl + 4 + 1*2
        inp_voff = struct.unpack_from("<H", buf, sg_vtbl + 4 + 1 * 2)[0]
        if inp_voff == 0:
            return -1  # field absent

        # inputs is a vector of int32 scalars (not a vector of tables)
        inp_ptr = sg0 + inp_voff
        inp_vec = inp_ptr + struct.unpack_from("<I", buf, inp_ptr)[0]

        # Sanity check: count should be a small positive number
        count = struct.unpack_from("<I", buf, inp_vec)[0]
        if count == 0 or count > 200:
            return -1

        return inp_vec
    except Exception:
        return -1


def _patch_hidden_inputs(model_bytes: bytes, hidden_indices: list,
                         anchor_idx: int = -1) -> bytes:
    """
    Remove hidden weight tensors from the TFLite subgraph inputs list in-place.

    PT2E lifts nn.Parameter values into the subgraph inputs list.  Newer
    tf.lite auto-initialises them from the flatbuffer buffer data; older
    tflite_runtime on the Pi does not and raises "Input tensor N lacks data".

    Primary path: navigate the flatbuffer object hierarchy to find the exact
    offset of the inputs vector — immune to false-positive matches on weight
    data or tensor shape vectors (which share the same [count][int32...] format).

    Fallback: heuristic byte scan with anchor_idx to reduce false positives
    (used only if flatbuffer navigation fails for any reason).
    """
    buf = bytearray(model_bytes)
    hidden_set = set(hidden_indices)

    # ── Primary: proper flatbuffer navigation ─────────────────────────────────
    vec_pos = _find_inputs_vec_offset(bytes(buf))
    if vec_pos >= 0:
        count = struct.unpack_from("<I", buf, vec_pos)[0]
        entries = [
            struct.unpack_from("<i", buf, vec_pos + 4 + j * 4)[0]
            for j in range(count)
        ]
        new_entries = [e for e in entries if e not in hidden_set]
        if len(new_entries) < count:
            struct.pack_into("<I", buf, vec_pos, len(new_entries))
            for j, e in enumerate(new_entries):
                struct.pack_into("<i", buf, vec_pos + 4 + j * 4, e)
            # Zero the vacated slots so buffer size is unchanged
            for j in range(len(new_entries), count):
                struct.pack_into("<i", buf, vec_pos + 4 + j * 4, 0)
        return bytes(buf)

    # ── Fallback: heuristic scan ───────────────────────────────────────────────
    for hidden_idx in hidden_indices:
        patched = False
        for count in range(2, 32):
            count_bytes = struct.pack("<I", count)
            clen        = len(count_bytes)
            window      = clen + count * 4
            i = 0
            while i <= len(buf) - window:
                if buf[i:i + clen] == count_bytes:
                    segment = buf[i:i + window]
                    entries = [
                        struct.unpack_from("<i", segment, clen + j * 4)[0]
                        for j in range(count)
                    ]
                    anchor_ok = (anchor_idx < 0) or (anchor_idx in entries)
                    if hidden_idx in entries and anchor_ok:
                        new_entries = [e for e in entries if e != hidden_idx]
                        new_count   = len(new_entries)
                        new_seg     = bytearray(struct.pack("<I", new_count))
                        for e in new_entries:
                            new_seg += struct.pack("<i", e)
                        new_seg += b'\x00' * (window - len(new_seg))
                        buf[i:i + window] = new_seg
                        patched = True
                        break
                i += 4
            if patched:
                break
    return bytes(buf)


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

        n_mels, max_frames = self.cfg["n_mels"], self.cfg["max_frames"]
        mel_shape = [1, max_frames, n_mels]

        with open(model_path, "rb") as f:
            model_bytes = f.read()

        hidden_found: list = []
        for attempt in range(12):  # up to 12 hidden weight tensors
            interp = tflite.Interpreter(model_content=model_bytes, num_threads=4)
            interp.allocate_tensors()

            input_details = interp.get_input_details()
            out_idx       = interp.get_output_details()[0]["index"]
            inp_idx       = input_details[0]["index"]
            extra_inputs  = []
            for det in input_details[1:]:
                shape = [int(x) for x in det["shape"]]
                extra_inputs.append((det["index"],
                                     self._make_cls_data(shape, det["dtype"])))

            dummy = np.zeros([1, max_frames, n_mels], dtype=np.float32)
            interp.set_tensor(inp_idx, dummy)
            for idx, data in extra_inputs:
                interp.set_tensor(idx, data)

            try:
                interp.invoke()
                # Probe succeeded — commit state and break
                self.interpreter   = interp
                self._inp_idx      = inp_idx
                self._out_idx      = out_idx
                self._extra_inputs = extra_inputs
                break
            except RuntimeError as exc:
                m = re.search(r'Input tensor (\d+) lacks data', str(exc))
                if not m:
                    raise
                bad = int(m.group(1))
                if bad in hidden_found:
                    raise RuntimeError(
                        f"Patch loop stalled on tensor {bad}. "
                        f"Cannot auto-fix this model."
                    ) from exc
                hidden_found.append(bad)
                print(f"[AWQ] Auto-patching hidden weight tensor {bad} "
                      f"(attempt {attempt + 1})…")
                model_bytes = _patch_hidden_inputs(model_bytes, [bad],
                                                   anchor_idx=inp_idx)
        else:
            raise RuntimeError(
                "Could not patch all hidden weight tensors after 12 attempts."
            )

        if hidden_found:
            print(f"[AWQ] Patched {len(hidden_found)} hidden tensor(s): "
                  f"{hidden_found}")

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
