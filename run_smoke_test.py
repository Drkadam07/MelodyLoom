from torch.utils.data import DataLoader
from src.config import ModelConfig, TrainingConfig, VocabConfig
from src.models.vocal2music_transformer import Vocal2MusicTransformer
from src.music.tokenizer import REMITokenizer
from src.training.trainer import MultimodalTrainer, Vocal2MusicDataset


def main() -> None:
    vocab_cfg = VocabConfig()
    tokenizer = REMITokenizer(vocab_cfg)

    model_cfg = ModelConfig(music_vocab_size=tokenizer.vocab_size)
    train_cfg = TrainingConfig(batch_size=4, grad_accum_steps=2, max_epochs=2)

    dataset = Vocal2MusicDataset.create_synthetic_dataset(
        num_samples=16, m_cfg=model_cfg
    )
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    model = Vocal2MusicTransformer(model_cfg)
    trainer = MultimodalTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        t_cfg=train_cfg,
        pad_id=tokenizer.pad_id,
    )

    for epoch in range(1, train_cfg.max_epochs + 1):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} | Train Loss: {loss:.4f}")

    # Test MIDI round-trip decoding
    sample_events = [
        "<BAR>",
        "POS_0",
        "TEMPO_96",
        "CHORD_C:maj",
        "INST_PIANO",
        "VEL_80",
        "PITCH_60",
        "DUR_4",
        "INST_BASS",
        "VEL_100",
        "PITCH_36",
        "DUR_8",
    ]
    encoded = tokenizer.encode_events(sample_events)
    pm = tokenizer.tokens_to_midi(encoded, default_bpm=96.0)
    print(f"Round-trip MIDI tracks generated: {len(pm.instruments)}")


if __name__ == "__main__":
    main()
