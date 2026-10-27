import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pretty_midi

try:
    import tgt
except ImportError:
    tgt = None

from src.config import ModelConfig, VocabConfig
from src.music.tokenizer import REMITokenizer


# Diatonic chord templates (scale degree offsets -> triad intervals & chord quality)
MAJOR_SCALE_CHORDS = {
    0: ("maj", [0, 4, 7]),
    2: ("min", [2, 5, 9]),
    4: ("min", [4, 7, 11]),
    5: ("maj", [5, 9, 0]),
    7: ("maj", [7, 11, 2]),
    9: ("min", [9, 0, 4]),
}

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class M4SingerUtterance:
    item_name: str
    singer: str
    song_name: str
    wav_path: Path
    midi_path: Optional[Path]
    textgrid_path: Optional[Path]
    phonemes: List[str]
    ph_durations: List[float]
    note_pitches: List[int]
    note_durations: List[float]
    estimated_bpm: float
    estimated_key_idx: int


class M4SingerParser:
    """
    Parses M4Singer `meta.json`, Praat `.TextGrid` phoneme alignments, and `.mid` vocal notes,
    and converts them into fixed-length multimodal conditioning + REMI target sequences.
    """

    def __init__(
        self,
        m4singer_root: Path,
        remi_tokenizer: REMITokenizer,
        vocab_cfg: VocabConfig = VocabConfig(),
        model_cfg: ModelConfig = ModelConfig(),
    ):
        self.root = Path(m4singer_root)
        self.remi = remi_tokenizer
        self.v_cfg = vocab_cfg
        self.m_cfg = model_cfg

    def parse_textgrid(self, tg_path: Path) -> Tuple[List[str], List[float]]:
        """Extracts phoneme labels and durations from a Praat .TextGrid file."""
        if tgt is None or not tg_path.exists():
            return [], []

        tg = tgt.io.read_textgrid(str(tg_path))
        tier_names = [t.name.lower() for t in tg.tiers]
        target_tier = None
        for candidate in ("phones", "phoneme", "phone", "ph"):
            if candidate in tier_names:
                target_tier = tg.get_tier_by_name(tg.tiers[tier_names.index(candidate)].name)
                break
        if target_tier is None and len(tg.tiers) > 0:
            target_tier = tg.tiers[-1]

        phonemes, durations = [], []
        for interval in target_tier._objects:
            label = interval.text.strip()
            dur = float(interval.end_time - interval.start_time)
            if dur > 0:
                phonemes.append(label if label else "<SP>")
                durations.append(round(dur, 4))
        return phonemes, durations

    def parse_vocal_midi(self, midi_path: Path) -> Tuple[List[int], List[float], float]:
        """Extracts MIDI note pitches, durations, and tempo from an M4Singer .mid file."""
        if not midi_path.exists():
            return [], [], 96.0

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        tempos = pm.get_tempo_changes()[1]
        bpm = float(tempos[0]) if len(tempos) > 0 else 96.0

        notes = [n for inst in pm.instruments for n in inst.notes]
        notes.sort(key=lambda x: x.start)

        pitches, durations = [], []
        last_end = 0.0
        for n in notes:
            if n.start - last_end > 0.08:
                pitches.append(0)  # Rest
                durations.append(round(n.start - last_end, 4))
            pitches.append(int(n.pitch))
            durations.append(round(n.end - n.start, 4))
            last_end = n.end

        return pitches, durations, bpm

    def estimate_key_from_pitches(self, pitches: List[int]) -> int:
        """Krumhansl-style pitch-class histogram key estimator (returns 0..11 for C..B)."""
        voiced = [p % 12 for p in pitches if p > 0]
        if not voiced:
            return 0
        counts = np.bincount(voiced, minlength=12)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        scores = [np.corrcoef(counts, np.roll(major_profile, k))[0, 1] for k in range(12)]
        return int(np.argmax(scores))

    def load_utterances(self) -> List[M4SingerUtterance]:
        """Loads all utterances from M4Singer meta.json (with fallback to directory scanning)."""
        meta_path = self.root / "meta.json"
        utterances: List[M4SingerUtterance] = []

        if meta_path.exists():
            raw_items: List[Dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
            for item in raw_items:
                item_name = item.get("item_name", "")
                # M4Singer format: "<Singer>#<SongName>#<UtteranceID>"
                parts = item_name.split("#")
                singer = parts[0] if len(parts) > 0 else "Unknown"
                song_name = parts[1] if len(parts) > 1 else item_name

                folder = self.root / f"{singer}#{song_name}"
                utt_id = parts[2] if len(parts) > 2 else item_name
                wav_path = folder / f"{utt_id}.wav"
                if not wav_path.exists():
                    wav_path = self.root / f"{item_name}.wav"

                tg_path = wav_path.with_suffix(".TextGrid")
                midi_path = wav_path.with_suffix(".mid")

                phs = item.get("phs", [])
                ph_durs = [float(d) for d in item.get("ph_dur", [])]
                if not phs and tg_path.exists():
                    phs, ph_durs = self.parse_textgrid(tg_path)

                notes = [int(n) for n in item.get("notes", [])]
                note_durs = [float(d) for d in item.get("notes_dur", [])]
                bpm = 96.0
                if midi_path.exists():
                    m_notes, m_durs, bpm = self.parse_vocal_midi(midi_path)
                    if not notes:
                        notes, note_durs = m_notes, m_durs

                key_idx = self.estimate_key_from_pitches(notes)

                utterances.append(
                    M4SingerUtterance(
                        item_name=item_name,
                        singer=singer,
                        song_name=song_name,
                        wav_path=wav_path,
                        midi_path=midi_path if midi_path.exists() else None,
                        textgrid_path=tg_path if tg_path.exists() else None,
                        phonemes=phs,
                        ph_durations=ph_durs,
                        note_pitches=notes,
                        note_durations=note_durs,
                        estimated_bpm=bpm,
                        estimated_key_idx=key_idx,
                    )
                )
        return utterances

    def quantize_melody_tensors(
        self, pitches: List[int], durations: List[float], bpm: float
    ) -> Tuple[List[int], List[int]]:
        """Quantizes vocal note durations onto the 16th-note grid and pads to max_melody_len."""
        sec_per_16th = (60.0 / max(60.0, min(180.0, bpm))) / 4.0
        q_pitches: List[int] = []
        q_durs: List[int] = []

        for pitch, dur in zip(pitches[: self.m_cfg.max_melody_len], durations[: self.m_cfg.max_melody_len]):
            clamped_pitch = max(0, min(127, int(pitch)))
            dur_16th = max(1, min(31, int(round(dur / sec_per_16th))))
            q_pitches.append(clamped_pitch)
            q_durs.append(dur_16th)

        pad_len = self.m_cfg.max_melody_len - len(q_pitches)
        if pad_len > 0:
            q_pitches.extend([0] * pad_len)
            q_durs.extend([0] * pad_len)

        return q_pitches[: self.m_cfg.max_melody_len], q_durs[: self.m_cfg.max_melody_len]

    def build_accompaniment_events_from_melody(
        self,
        pitches: List[int],
        durations: List[float],
        bpm: float,
        key_idx: int,
    ) -> List[str]:
        """
        Generates a synchronized REMI accompaniment event sequence (Chords, Drums, Bass, Piano)
        aligned to the vocal melody bars.
        """
        quant_bpm = min(
            range(self.v_cfg.min_tempo, self.v_cfg.max_tempo + 1, self.v_cfg.tempo_step),
            key=lambda b: abs(b - bpm),
        )
        key_name = PITCH_CLASS_NAMES[key_idx % 12]
        sec_per_bar = (60.0 / quant_bpm) * 4.0
        total_dur = sum(durations) if durations else 8.0
        num_bars = max(1, min(8, int(np.ceil(total_dur / sec_per_bar))))

        events: List[str] = [f"TEMPO_{quant_bpm}", f"KEY_{key_name}"]

        # Map notes into bars to infer the dominant harmony per bar
        cum_times = np.cumsum([0.0] + durations[:-1])
        for bar_idx in range(num_bars):
            bar_start = bar_idx * sec_per_bar
            bar_end = bar_start + sec_per_bar
            bar_pitches = [
                p
                for p, t in zip(pitches, cum_times)
                if p > 0 and bar_start <= t < bar_end
            ]

            # Match bar pitches against diatonic triads in the estimated key
            best_deg = 0
            best_quality, best_intervals = MAJOR_SCALE_CHORDS[0]
            best_overlap = -1
            for deg, (qual, intervals) in MAJOR_SCALE_CHORDS.items():
                chord_pcs = {(key_idx + iv) % 12 for iv in intervals}
                overlap = sum(1 for p in bar_pitches if (p % 12) in chord_pcs)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_deg = deg
                    best_quality = qual
                    best_intervals = intervals

            root_pc = (key_idx + best_deg) % 12
            chord_symbol = f"{PITCH_CLASS_NAMES[root_pc]}:{best_quality}"
            bass_midi = max(self.v_cfg.min_midi, min(self.v_cfg.max_midi, 36 + root_pc))
            triad_midis = [
                max(self.v_cfg.min_midi, min(self.v_cfg.max_midi, 60 + ((key_idx + iv) % 12)))
                for iv in best_intervals
            ]

            events.append(self.v_cfg.bar_token)
            events.append("POS_0")
            events.append(f"CHORD_{chord_symbol}")

            # Beat 1 (POS_0): Kick + Bass root + Piano triad
            events.extend(["INST_DRUMS", "VEL_100", "PITCH_36", "DUR_4"])
            events.extend(["INST_BASS", "VEL_80", f"PITCH_{bass_midi}", "DUR_8"])
            for m_note in triad_midis:
                events.extend(["INST_PIANO", "VEL_60", f"PITCH_{m_note}", "DUR_8"])

            # Beat 2 (POS_4): Snare
            events.extend(["POS_4", "INST_DRUMS", "VEL_80", "PITCH_38", "DUR_4"])

            # Beat 3 (POS_8): Kick + Bass root + Piano triad
            events.extend(["POS_8", "INST_DRUMS", "VEL_100", "PITCH_36", "DUR_4"])
            events.extend(["INST_BASS", "VEL_80", f"PITCH_{bass_midi}", "DUR_8"])
            for m_note in triad_midis:
                events.extend(["INST_PIANO", "VEL_60", f"PITCH_{m_note}", "DUR_8"])

            # Beat 4 (POS_12): Snare
            events.extend(["POS_12", "INST_DRUMS", "VEL_80", "PITCH_38", "DUR_4"])

        return events