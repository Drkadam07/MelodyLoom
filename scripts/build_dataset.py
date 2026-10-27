import argparse
import hashlib
from pathlib import Path
from typing import Dict, List
import torch
from tqdm import tqdm

from src.audio.preprocess import VocalPreprocessor
from src.config import AudioConfig, ModelConfig, VocabConfig
from src.data.m4singer_parser import M4SingerParser, M4SingerUtterance
from src.lyrics.alignment import LyricPhonemeTokenizer, WhisperPhonemeAligner
from src.music.tokenizer import REMITokenizer


def assign_song_split(song_identifier: str, train_ratio: float = 0.70, val_ratio: float = 0.15) -> str:
    """
    Deterministic hash-based song-level split assignment to prevent data leakage
    between train, validation, and test sets.
    """
    digest = hashlib.md5(song_identifier.encode("utf-8")).hexdigest()
    bucket = (int(digest[:8], 16) % 1000) / 1000.0
    if bucket < train_ratio:
        return "train"
    elif bucket < (train_ratio + val_ratio):
        return "val"
    return "test"


def pad_or_truncate(seq: List[int], target_len: int, pad_id: int = 0) -> List[int]:
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [pad_id] * (target_len - len(seq))


def build_from_m4singer(
    m4singer_dir: Path,
    output_dir: Path,
    lyric_tokenizer: LyricPhonemeTokenizer,
    remi_tokenizer: REMITokenizer,
    m_cfg: ModelConfig,
) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    parser = M4SingerParser(
        m4singer_root=m4singer_dir,
        remi_tokenizer=remi_tokenizer,
        model_cfg=m_cfg,
    )
    utterances = parser.load_utterances()
    splits: Dict[str, List[Dict[str, torch.Tensor]]] = {"train": [], "val": [], "test": []}

    for utt in tqdm(utterances, desc="Processing M4Singer utterances"):
        if not utt.phonemes or not utt.note_pitches:
            continue

        split_name = assign_song_split(f"{utt.singer}#{utt.song_name}")

        # 1. Encode Lyric Phonemes
        lyric_ids = lyric_tokenizer.encode(
            utt.phonemes, max_len=m_cfg.max_lyrics_len, add_New_tokens=(split_name == "train")
        )

        # 2. Quantize Vocal Melody Notes
        q_pitches, q_durs = parser.quantize_melody_tensors(
            utt.note_pitches, utt.note_durations, utt.estimated_bpm
        )

        # 3. Build & Encode Target Accompaniment REMI Tokens
        remi_events = parser.build_accompaniment_events_from_melody(
            utt.note_pitches, utt.note_durations, utt.estimated_bpm, utt.estimated_key_idx
        )
        music_ids = remi_tokenizer.encode_events(remi_events)
        music_ids = pad_or_truncate(music_ids, m_cfg.max_music_len, pad_id=remi_tokenizer.pad_id)

        record = {
            "lyric_tokens": torch.tensor(lyric_ids, dtype=torch.long),
            "melody_pitches": torch.tensor(q_pitches, dtype=torch.long),
            "melody_durations": torch.tensor(q_durs, dtype=torch.long),
            "genre_id": torch.tensor(0, dtype=torch.long),  # 0: POP default for M4Singer
            "mood_id": torch.tensor(2, dtype=torch.long),   # 2: MELANCHOLIC / expressive default
            "key_id": torch.tensor(utt.estimated_key_idx, dtype=torch.long),
            "norm_tempo": torch.tensor(utt.estimated_bpm / 180.0, dtype=torch.float32),
            "music_tokens": torch.tensor(music_ids, dtype=torch.long),
        }
        splits[split_name].append(record)

    return splits


def build_from_raw_vocals(
    raw_vocals_dir: Path,
    lyric_tokenizer: LyricPhonemeTokenizer,
    remi_tokenizer: REMITokenizer,
    m_cfg: ModelConfig,
    whisper_model_size: str = "small",
    device: str = "cuda",
) -> List[Dict[str, torch.Tensor]]:
    """
    Processes raw unannotated vocal WAV files using Whisper word-level alignment
    and pYIN pitch extraction.
    """
    aligner = WhisperPhonemeAligner(model_size=whisper_model_size, device=device)
    audio_proc = VocalPreprocessor(AudioConfig())
    parser = M4SingerParser(
        m4singer_root=raw_vocals_dir, remi_tokenizer=remi_tokenizer, model_cfg=m_cfg
    )

    records: List[Dict[str, torch.Tensor]] = []
    wav_files = sorted(list(raw_vocals_dir.rglob("*.wav")))

    for wav_path in tqdm(wav_files, desc="Aligning raw vocals with Whisper"):
        y_24k, _ = audio_proc.load_and_normalize(wav_path)
        quant_notes = audio_proc.extract_melody_notes(y_24k, bpm=96.0)
        if not quant_notes:
            continue

        timed_phs = aligner.extract_timed_phonemes(wav_path)
        ph_strings = [tp.phoneme for tp in timed_phs] or ["<SP>"]

        lyric_ids = lyric_tokenizer.encode(ph_strings, max_len=m_cfg.max_lyrics_len)
        pitches = [n.midi for n in quant_notes]
        durs = [n.duration_sec for n in quant_notes]
        key_idx = parser.estimate_key_from_pitches(pitches)

        q_pitches, q_durs = parser.quantize_melody_tensors(pitches, durs, bpm=96.0)
        remi_events = parser.build_accompaniment_events_from_melody(pitches, durs, 96.0, key_idx)
        music_ids = pad_or_truncate(
            remi_tokenizer.encode_events(remi_events),
            m_cfg.max_music_len,
            pad_id=remi_tokenizer.pad_id,
        )

        records.append({
            "lyric_tokens": torch.tensor(lyric_ids, dtype=torch.long),
            "melody_pitches": torch.tensor(q_pitches, dtype=torch.long),
            "melody_durations": torch.tensor(q_durs, dtype=torch.long),
            "genre_id": torch.tensor(0, dtype=torch.long),
            "mood_id": torch.tensor(2, dtype=torch.long),
            "key_id": torch.tensor(key_idx, dtype=torch.long),
            "norm_tempo": torch.tensor(96.0 / 180.0, dtype=torch.float32),
            "music_tokens": torch.tensor(music_ids, dtype=torch.long),
        })

    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Vocal2Music Multimodal Dataset")
    ap.add_argument("--m4singer-dir", type=Path, default=None, help="Path to M4Singer root folder")
    ap.add_argument("--raw-vocals-dir", type=Path, default=None, help="Path to unannotated vocal WAVs")
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory")
    ap.add_argument("--whisper-size", type=str, default="small", help="Faster-Whisper model size")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    v_cfg = VocabConfig()
    remi_tokenizer = REMITokenizer(v_cfg)
    m_cfg = ModelConfig(music_vocab_size=remi_tokenizer.vocab_size)
    lyric_tokenizer = LyricPhonemeTokenizer(max_vocab_size=m_cfg.lyrics_vocab_size)

    splits: Dict[str, List[Dict[str, torch.Tensor]]] = {"train": [], "val": [], "test": []}

    if args.m4singer_dir and args.m4singer_dir.exists():
        m4_splits = build_from_m4singer(
            args.m4singer_dir, args.output_dir, lyric_tokenizer, remi_tokenizer, m_cfg
        )
        for k in splits:
            splits[k].extend(m4_splits[k])

    if args.raw_vocals_dir and args.raw_vocals_dir.exists():
        raw_records = build_from_raw_vocals(
            args.raw_vocals_dir,
            lyric_tokenizer,
            remi_tokenizer,
            m_cfg,
            whisper_model_size=args.whisper_size,
            device=args.device,
        )
        splits["train"].extend(raw_records)

    # Save serialized splits and phoneme vocabulary
    for split_name, records in splits.items():
        out_path = args.output_dir / f"{split_name}.pt"
        torch.save(records, out_path)
        print(f"Saved {len(records)} examples to {out_path}")

    vocab_path = args.output_dir / "lyric_vocab.json"
    lyric_tokenizer.save(vocab_path)
    print(f"Saved phoneme vocabulary ({len(lyric_tokenizer.token_to_id)} tokens) to {vocab_path}")


if __name__ == "__main__":
    main()