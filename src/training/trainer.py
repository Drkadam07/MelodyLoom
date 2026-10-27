from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from src.config import ModelConfig, TrainingConfig
from src.models.vocal2music_transformer import Vocal2MusicTransformer


class Vocal2MusicDataset(Dataset):
    """
    Loads pre-aligned multimodal tensors (lyrics, melody, style, and target music tokens).
    Includes a synthetic fallback mode so you can test the training pipeline immediately.
    """

    def __init__(self, data_records: List[Dict[str, torch.Tensor]]):
        self.records = data_records

    @classmethod
    def create_synthetic_dataset(
        cls, num_samples: int, m_cfg: ModelConfig
    ) -> "Vocal2MusicDataset":
        records = []
        for _ in range(num_samples):
            records.append(
                {
                    "lyric_tokens": torch.randint(1, m_cfg.lyrics_vocab_size, (64,)),
                    "melody_pitches": torch.randint(48, 84, (128,)),
                    "melody_durations": torch.randint(1, 16, (128,)),
                    "genre_id": torch.tensor(0, dtype=torch.long),
                    "mood_id": torch.tensor(2, dtype=torch.long),
                    "key_id": torch.tensor(0, dtype=torch.long),
                    "norm_tempo": torch.tensor(96.0 / 180.0, dtype=torch.float32),
                    "music_tokens": torch.randint(1, m_cfg.music_vocab_size, (256,)),
                }
            )
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.records[idx]


class MultimodalTrainer:
    """
    Production trainer with AMP (Automatic Mixed Precision), gradient accumulation,
    cosine annealing with warmup, and checkpoint management.
    """

    def __init__(
        self,
        model: Vocal2MusicTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        t_cfg: TrainingConfig,
        pad_id: int = 0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = t_cfg

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id, label_smoothing=t_cfg.label_smoothing
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=t_cfg.learning_rate,
            weight_decay=t_cfg.weight_decay,
            betas=(0.9, 0.98),
        )
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            music_in = batch["music_tokens"][:, :-1]
            music_tgt = batch["music_tokens"][:, 1:]

            lyrics_pad_mask = batch["lyric_tokens"] == 0
            melody_pad_mask = batch["melody_pitches"] == 0
            music_pad_mask = music_in == 0

            with autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(
                    music_tokens=music_in,
                    lyric_tokens=batch["lyric_tokens"],
                    melody_pitches=batch["melody_pitches"],
                    melody_durations=batch["melody_durations"],
                    genre_id=batch["genre_id"],
                    mood_id=batch["mood_id"],
                    key_id=batch["key_id"],
                    norm_tempo=batch["norm_tempo"],
                    lyrics_pad_mask=lyrics_pad_mask,
                    melody_pad_mask=melody_pad_mask,
                    music_pad_mask=music_pad_mask,
                )
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), music_tgt.reshape(-1)
                )
                scaled_loss = loss / self.cfg.grad_accum_steps

            self.scaler.scale(scaled_loss).backward()

            if (step + 1) % self.cfg.grad_accum_steps == 0 or (step + 1) == len(
                self.train_loader
            ):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))
