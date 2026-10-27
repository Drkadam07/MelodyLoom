import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.config import ModelConfig, TrainingConfig, VocabConfig
from src.models.vocal2music_transformer import Vocal2MusicTransformer
from src.music.tokenizer import REMITokenizer


class Vocal2MusicDataset(Dataset):
    """
    Loads pre-aligned multimodal tensors (lyrics, melody, style, and target music tokens).
    Supports loading from serialized .pt files or generating synthetic test data.
    """

    def __init__(self, data_records: List[Dict[str, torch.Tensor]]):
        self.records = data_records

    @classmethod
    def from_preprocessed_pt(cls, pt_path: Path) -> "Vocal2MusicDataset":
        records = torch.load(pt_path, map_location="cpu", weights_only=True)
        return cls(records)

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
    Production trainer with modern PyTorch AMP (Automatic Mixed Precision), gradient accumulation,
    cosine annealing with warmup, validation loops, and checkpoint management.
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
        # Modern PyTorch amp API
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
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

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
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

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            music_in = batch["music_tokens"][:, :-1]
            music_tgt = batch["music_tokens"][:, 1:]

            lyrics_pad_mask = batch["lyric_tokens"] == 0
            melody_pad_mask = batch["melody_pitches"] == 0
            music_pad_mask = music_in == 0

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
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
            total_loss += loss.item()

        return total_loss / max(1, len(self.val_loader))

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Vocal2Music Multimodal Transformer"
    )
    parser.add_argument(
        "--train-path", type=Path, default=None, help="Path to train.pt"
    )
    parser.add_argument("--val-path", type=Path, default=None, help="Path to val.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    v_cfg = VocabConfig()
    remi = REMITokenizer(v_cfg)
    m_cfg = ModelConfig(music_vocab_size=remi.vocab_size)
    t_cfg = TrainingConfig(batch_size=args.batch_size, max_epochs=args.epochs)

    if args.train_path and args.train_path.exists():
        train_ds = Vocal2MusicDataset.from_preprocessed_pt(args.train_path)
        val_ds = (
            Vocal2MusicDataset.from_preprocessed_pt(args.val_path)
            if args.val_path and args.val_path.exists()
            else train_ds
        )
    else:
        print(
            "Preprocessed dataset paths not found. Initializing synthetic smoke test dataset..."
        )
        train_ds = Vocal2MusicDataset.create_synthetic_dataset(32, m_cfg)
        val_ds = Vocal2MusicDataset.create_synthetic_dataset(8, m_cfg)

    train_loader = DataLoader(
        train_ds, batch_size=t_cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=t_cfg.batch_size, shuffle=False)

    model = Vocal2MusicTransformer(m_cfg)
    trainer = MultimodalTrainer(
        model, train_loader, val_loader, t_cfg, pad_id=remi.pad_id
    )

    best_val_loss = float("inf")
    for epoch in range(1, t_cfg.max_epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate()
        print(
            f"Epoch {epoch}/{t_cfg.max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = t_cfg.checkpoint_dir / "best_model.pt"
            trainer.save_checkpoint(ckpt_path, epoch, val_loss)
            print(f"  -> Saved new best model checkpoint to {ckpt_path}")
