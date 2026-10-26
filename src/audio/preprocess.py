from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import librosa
import numpy as np
import soundfile as sf
from src.config import AudioConfig


@dataclass
class QuantizedNote:
    midi: int
    start_sec: float
    duration_sec: float
    dur_16th: int


@dataclass
class AlignedLyricToken:
    token: str
    start_sec: float
    end_sec: float


class VocalPreprocessor:
    """
    Standardizes raw vocal audio, extracts F0 pitch contours, and quantizes vocal melodies
    onto a musical 16th-note grid.
    """

    def __init__(self, cfg: AudioConfig = AudioConfig()):
        self.cfg = cfg

    def load_and_normalize(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            audio_analysis (24kHz mono float32)
            audio_render (44.1kHz mono float32)
        """
        y_render, _ = librosa.load(str(audio_path), sr=self.cfg.render_sr, mono=True)
        y_trimmed, _ = librosa.effects.trim(y_render, top_db=35)

        # Peak and RMS normalization
        peak = np.max(np.abs(y_trimmed)) + 1e-8
        y_norm = (y_trimmed / peak) * 0.9

        y_analysis = librosa.resample(
            y_norm, orig_sr=self.cfg.render_sr, target_sr=self.cfg.analysis_sr
        )
        return y_analysis.astype(np.float32), y_norm.astype(np.float32)

    def extract_melody_notes(
        self, y_analysis: np.ndarray, bpm: float = 96.0
    ) -> List[QuantizedNote]:
        """
        Extracts F0 via probabilistic YIN (pYIN), filters unvoiced regions,
        and segments into quantized MIDI notes.
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_analysis,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            sr=self.cfg.analysis_sr,
            hop_length=self.cfg.hop_length,
        )

        frame_time = self.cfg.hop_length / float(self.cfg.analysis_sr)
        sec_per_16th = (60.0 / bpm) / 4.0

        midi_contour = np.zeros_like(f0, dtype=np.int32)
        valid_mask = (~np.isnan(f0)) & voiced_flag & (voiced_probs > 0.5)
        midi_contour[valid_mask] = np.round(librosa.hz_to_midi(f0[valid_mask])).astype(
            np.int32
        )

        notes: List[QuantizedNote] = []
        if len(midi_contour) == 0:
            return notes

        cur_pitch = int(midi_contour[0])
        start_idx = 0

        for idx in range(1, len(midi_contour)):
            pitch = int(midi_contour[idx])
            if pitch != cur_pitch:
                dur_sec = (idx - start_idx) * frame_time
                if dur_sec >= 0.08:  # Filter micro-glitches under 80ms
                    dur_16th = max(1, min(16, int(round(dur_sec / sec_per_16th))))
                    notes.append(
                        QuantizedNote(
                            midi=cur_pitch,  # 0 represents a rest
                            start_sec=start_idx * frame_time,
                            duration_sec=dur_sec,
                            dur_16th=dur_16th,
                        )
                    )
                cur_pitch = pitch
                start_idx = idx

        return notes
