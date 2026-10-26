from typing import Dict, List, Tuple
import pretty_midi
from src.config import VocabConfig


class REMITokenizer:
    """
    Bidirectional REMI-inspired event tokenizer for multi-track symbolic accompaniment.
    Converts structured note/chord events <-> integer token sequences <-> PrettyMIDI.
    """

    def __init__(self, cfg: VocabConfig = VocabConfig()):
        self.cfg = cfg
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_vocab()

    def _add_token(self, token: str) -> None:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def _build_vocab(self) -> None:
        # 1. Special tokens
        for special in [
            self.cfg.pad_token,
            self.cfg.bos_token,
            self.cfg.eos_token,
            self.cfg.unk_token,
            self.cfg.bar_token,
        ]:
            self._add_token(special)

        # 2. Sub-bar position tokens (16th-note resolution)
        for pos in range(self.cfg.positions):
            self._add_token(f"POS_{pos}")

        # 3. Tempo & Key tokens
        for bpm in range(
            self.cfg.min_tempo, self.cfg.max_tempo + 1, self.cfg.tempo_step
        ):
            self._add_token(f"TEMPO_{bpm}")
        for key in self.cfg.keys:
            self._add_token(f"KEY_{key}")

        # 4. Chord tokens
        for chord in self.cfg.chords:
            self._add_token(f"CHORD_{chord}")

        # 5. Instrument tokens
        for inst in self.cfg.instruments:
            self._add_token(f"INST_{inst}")

        # 6. Pitch, Duration, Velocity tokens
        for midi_note in range(self.cfg.min_midi, self.cfg.max_midi + 1):
            self._add_token(f"PITCH_{midi_note}")
        for dur in self.cfg.durations:
            self._add_token(f"DUR_{dur}")
        for vel in self.cfg.velocities:
            self._add_token(f"VEL_{vel}")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.cfg.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.cfg.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.cfg.eos_token]

    def _quantize_value(self, val: int, candidates: List[int]) -> int:
        return min(candidates, key=lambda c: abs(c - val))

    def encode_events(self, events: List[str]) -> List[int]:
        unk_id = self.token_to_id[self.cfg.unk_token]
        tokens = [self.bos_id]
        tokens.extend(self.token_to_id.get(ev, unk_id) for ev in events)
        tokens.append(self.eos_id)
        return tokens

    def decode_ids(self, token_ids: List[int]) -> List[str]:
        events = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, self.cfg.unk_token)
            if tok == self.cfg.eos_token:
                break
            if tok not in (self.cfg.pad_token, self.cfg.bos_token, self.cfg.unk_token):
                events.append(tok)
        return events

    def tokens_to_midi(
        self, token_ids: List[int], default_bpm: float = 96.0
    ) -> pretty_midi.PrettyMIDI:
        """
        Parses a generated REMI token sequence back into a multi-track PrettyMIDI object.
        """
        events = self.decode_ids(token_ids)
        bpm = default_bpm
        for ev in events:
            if ev.startswith("TEMPO_"):
                bpm = float(ev.split("_")[1])
                break

        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        sec_per_beat = 60.0 / bpm
        sec_per_16th = sec_per_beat / 4.0
        sec_per_bar = sec_per_beat * 4.0

        tracks: Dict[str, pretty_midi.Instrument] = {
            "PIANO": pretty_midi.Instrument(program=0, is_drum=False, name="Piano"),
            "BASS": pretty_midi.Instrument(program=33, is_drum=False, name="Bass"),
            "PAD": pretty_midi.Instrument(program=89, is_drum=False, name="Warm Pad"),
            "DRUMS": pretty_midi.Instrument(program=0, is_drum=True, name="Drums"),
        }

        current_bar = -1
        current_pos = 0
        current_inst = "PIANO"
        current_vel = 80
        pending_pitch = None

        for ev in events:
            if ev == self.cfg.bar_token:
                current_bar += 1
                current_pos = 0
            elif ev.startswith("POS_"):
                current_pos = int(ev.split("_")[1])
            elif ev.startswith("INST_"):
                inst_name = ev.split("_")[1]
                if inst_name in tracks:
                    current_inst = inst_name
            elif ev.startswith("VEL_"):
                current_vel = int(ev.split("_")[1])
            elif ev.startswith("PITCH_"):
                pending_pitch = int(ev.split("_")[1])
            elif ev.startswith("DUR_") and pending_pitch is not None:
                dur_16ths = int(ev.split("_")[1])
                bar_idx = max(0, current_bar)
                start_time = (bar_idx * sec_per_bar) + (current_pos * sec_per_16th)
                end_time = start_time + (dur_16ths * sec_per_16th)

                note = pretty_midi.Note(
                    velocity=current_vel,
                    pitch=pending_pitch,
                    start=start_time,
                    end=end_time,
                )
                tracks[current_inst].notes.append(note)
                pending_pitch = None

        for inst in tracks.values():
            if len(inst.notes) > 0:
                pm.instruments.append(inst)

        return pm
