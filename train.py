"""
Decoder-only Transformer for Keyword Spotting
Dataset: Google Speech Commands v2 (35 keywords)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 16_000
N_MELS       = 80
N_FFT        = 512
WIN_LENGTH   = 400     # 25 ms
HOP_LENGTH   = 160     # 10 ms
MAX_FRAMES   = 101     # frames for 1-second clip

D_MODEL      = 256
N_HEADS      = 4
N_LAYERS     = 4
D_FF         = 512
DROPOUT      = 0.1

BATCH_SIZE   = 64
LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 40
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT    = "./data"
CKPT_DIR     = "./checkpoints"
CKPT_PATH    = f"{CKPT_DIR}/best_model.pt"

KEYWORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
]
LABEL2IDX  = {w: i for i, w in enumerate(KEYWORDS)}
NUM_CLASSES = len(KEYWORDS)  # 35


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpeechCommandsDataset(Dataset):
    def __init__(self, root: str, subset: str, augment: bool = False):
        self._ds = SPEECHCOMMANDS(root, url="speech_commands_v0.02",
                                  download=True, subset=subset)
        self.augment = augment
        self.mel_transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        self.freq_mask = FrequencyMasking(freq_mask_param=10)
        self.time_mask = TimeMasking(time_mask_param=20)
        self._indices = [
            i for i in range(len(self._ds)) if self._ds[i][2] in LABEL2IDX
        ]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        waveform, sr, label, *_ = self._ds[self._indices[idx]]

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        target_len = SAMPLE_RATE
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[..., :target_len]

        mel = self.mel_transform(waveform)     # [1, n_mels, T]
        mel = (mel + 1e-6).log()
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)

        if self.augment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        mel = mel.squeeze(0).T                 # [T, n_mels]
        return mel, LABEL2IDX[label]


def collate_fn(batch):
    mels, labels = zip(*batch)
    return torch.stack(mels), torch.tensor(labels, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = math.sqrt(self.head_dim)
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)     # [3, B, H, T, D]
        q, k, v = qkv.unbind(0)               # each [B, H, T, D]

        attn = (q @ k.transpose(-2, -1)) / self.scale   # [B, H, T, T]
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + causal
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # net.0 and net.3 are the two Linear layers targeted by SmoothQuant/AWQ
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class KeywordSpottingTransformer(nn.Module):
    """
    Decoder-only transformer for keyword spotting.
    Input:  [B, T, n_mels]  log-mel spectrogram frames
    Output: [B, num_classes] logits

    A learnable [CLS] token is appended at the END of the sequence.
    In causal attention the last position attends to all previous frames,
    making it a natural pooling point for classification.
    """

    def __init__(
        self,
        n_mels:      int = N_MELS,
        d_model:     int = D_MODEL,
        n_heads:     int = N_HEADS,
        n_layers:    int = N_LAYERS,
        d_ff:        int = D_FF,
        dropout:     float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        max_len:     int = MAX_FRAMES,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_emb    = nn.Embedding(max_len + 1, d_model)  # +1 for CLS
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks     = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x   = self.input_proj(x)                          # [B, T, D]
        cls = self.cls_token.expand(B, -1, -1)            # [B, 1, D]
        x   = torch.cat([x, cls], dim=1)                  # [B, T+1, D]
        pos = torch.arange(T + 1, device=x.device)
        x   = x + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x   = self.norm(x)
        return self.head(x[:, -1, :])                     # CLS token output


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading datasets (downloads ~2.4 GB on first run)...")
    train_ds = SpeechCommandsDataset(DATA_ROOT, "training",   augment=True)
    val_ds   = SpeechCommandsDataset(DATA_ROOT, "validation", augment=False)
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_dl   = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model = KeywordSpottingTransformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  |  Device: {DEVICE}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_dl), epochs=EPOCHS,
    )

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_correct = total = 0
        for mel, labels in train_dl:
            mel, labels = mel.to(DEVICE), labels.to(DEVICE)
            logits = model(mel)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss    += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += labels.size(0)

        train_acc = total_correct / total

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for mel, labels in val_dl:
                mel, labels = mel.to(DEVICE), labels.to(DEVICE)
                preds        = model(mel).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"loss {total_loss/total:.4f} | "
            f"train {train_acc:.4f} | val {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "config": dict(
                        n_mels=N_MELS, d_model=D_MODEL, n_heads=N_HEADS,
                        n_layers=N_LAYERS, d_ff=D_FF, dropout=0.0,
                        num_classes=NUM_CLASSES, max_len=MAX_FRAMES,
                    ),
                },
                CKPT_PATH,
            )
            print(f"  ✓ Saved best model  val_acc={val_acc:.4f}")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
