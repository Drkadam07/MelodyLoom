from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class AudioConfig:
    analysis_sr: int = 24000
    render_sr: int = 44100
    hop_length: int = 256
    fmin: float = 65.41   # C2
    fmax: float = 1046.50 # C6
    segment_seconds: float = 12.0
    target_lufs: float = -14.0


@dataclass
class VocabConfig:
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"
    bar_token: str = "<BAR>"

    min_tempo: int = 60
    max_tempo: int = 180
    tempo_step: int = 4

    min_midi: int = 24
    max_midi: int = 96

    # Quantized durations in 16th-note grid units (1 = 1/16th, 4 = 1/4th, 16 = whole)
    durations: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 16])
    velocities: List[int] = field(default_factory=lambda: [40, 60, 80, 100, 120])
    positions: int = 16  # 16 sixteenth-note positions per 4/4 bar

    instruments: List[str] = field(default_factory=lambda: ["DRUMS", "BASS", "PIANO", "PAD"])
    chords: List[str] = field(default_factory=lambda: [
        "C:maj", "C:min", "C#:maj", "C#:min", "D:maj", "D:min",
        "D#:maj", "D#:min", "E:maj", "E:min", "F:maj", "F:min",
        "F#:maj", "F#:min", "G:maj", "G:min", "G#:maj", "G#:min",
        "A:maj", "A:min", "A#:maj", "A#:min", "B:maj", "B:min", "N"
    ])
    genres: List[str] = field(default_factory=lambda: ["POP", "ROCK", "LOFI", "EDM", "BALLAD"])
    moods: List[str] = field(default_factory=lambda: ["HAPPY", "SAD", "MELANCHOLIC", "ENERGETIC", "CALM"])
    keys: List[str] = field(default_factory=lambda: [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ])


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    lyrics_enc_layers: int = 4
    melody_enc_layers: int = 4
    music_dec_layers: int = 8

    max_lyrics_len: int = 256
    max_melody_len: int = 512
    max_music_len: int = 2048

    lyrics_vocab_size: int = 5000
    melody_pitch_vocab: int = 130  # 0-127 MIDI + rest + pad
    melody_dur_vocab: int = 32
    music_vocab_size: int = 512    # Dynamically overwritten by REMITokenizer


@dataclass
class TrainingConfig:
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    checkpoint_dir: Path = Path("checkpoints")
    soundfont_path: Path = Path("assets/FluidR3_GM.sf2")