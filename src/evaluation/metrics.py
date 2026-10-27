from typing import Dict, List
import numpy as np
import pretty_midi
from src.audio.preprocess import QuantizedNote

CHORD_PITCH_CLASSES: Dict[str, List[int]] = {
    "C:maj": [0, 4, 7],
    "A:min": [9, 0, 4],
    "F:maj": [5, 9, 0],
    "G:maj": [7, 11, 2],
    "D:min": [2, 5, 9],
    "E:min": [4, 7, 11],
}


def compute_harmony_compatibility(
    vocal_notes: List[QuantizedNote], accompaniment_midi: pretty_midi.PrettyMIDI
) -> float:
    """
    Calculates the proportion of sung vocal notes whose pitch class matches active
    harmonic accompaniment notes within the same time window.
    """
    voiced_notes = [n for n in vocal_notes if n.midi > 0]
    if not voiced_notes:
        return 0.0

    inst_notes = [
        note
        for inst in accompaniment_midi.instruments
        if not inst.is_drum
        for note in inst.notes
    ]
    if not inst_notes:
        return 0.0

    compatible = 0
    for v_note in voiced_notes:
        v_end = v_note.start_sec + v_note.duration_sec
        active_pcs = {
            n.pitch % 12
            for n in inst_notes
            if not (n.end <= v_note.start_sec or n.start >= v_end)
        }
        if (v_note.midi % 12) in active_pcs:
            compatible += 1

    return compatible / float(len(voiced_notes))


def compute_beat_alignment_error(
    accompaniment_midi: pretty_midi.PrettyMIDI, bpm: float
) -> float:
    """
    Calculates mean absolute distance (in seconds) between generated note onsets
    and the nearest quantized musical beat grid position. Lower is better.
    """
    sec_per_subbeat = (60.0 / bpm) / 2.0  # 8th-note beat grid
    onsets = [
        note.start for inst in accompaniment_midi.instruments for note in inst.notes
    ]
    if not onsets:
        return 0.0

    errors = []
    for onset in onsets:
        nearest_grid = round(onset / sec_per_subbeat) * sec_per_subbeat
        errors.append(abs(onset - nearest_grid))

    return float(np.mean(errors))
