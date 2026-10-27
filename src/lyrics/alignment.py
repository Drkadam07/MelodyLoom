import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from g2p_en import G2p
except ImportError:
    G2p = None


@dataclass
class TimedPhoneme:
    phoneme: str
    word: str
    start_sec: float
    end_sec: float


class LyricPhonemeTokenizer:
    """
    Deterministic vocabulary mapping phonemes and word tokens to integer IDs
    for the LyricsEncoder.
    """

    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SP>", "<AP>"]

    # Standard ARPAbet (stress stripped) + common M4Singer Mandarin Pinyin initials/finals
    BASE_PHONEMES = [
        # ARPAbet vowels & consonants
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "ER",
        "EY",
        "F",
        "G",
        "HH",
        "IH",
        "IY",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OY",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UW",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
        # Common M4Singer Pinyin initials & finals
        "b",
        "p",
        "m",
        "f",
        "d",
        "t",
        "n",
        "l",
        "g",
        "k",
        "h",
        "j",
        "q",
        "x",
        "zh",
        "ch",
        "sh",
        "r",
        "z",
        "c",
        "s",
        "y",
        "w",
        "a",
        "o",
        "e",
        "i",
        "u",
        "v",
        "ai",
        "ei",
        "ui",
        "ao",
        "ou",
        "iu",
        "ie",
        "ve",
        "er",
        "an",
        "en",
        "in",
        "un",
        "vn",
        "ang",
        "eng",
        "ing",
        "ong",
        "ia",
        "iao",
        "ian",
        "iang",
        "iong",
        "ua",
        "uo",
        "uai",
        "uan",
        "uang",
    ]

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        for tok in self.SPECIAL_TOKENS + self.BASE_PHONEMES:
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            if len(self.token_to_id) >= self.max_vocab_size:
                return self.token_to_id["<UNK>"]
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def normalize_phoneme(self, ph: str) -> str:
        ph = ph.strip()
        if ph in ("", "sp", "sil", "<sp>", "SP"):
            return "<SP>"
        if ph in ("ap", "ASP", "<ap>"):
            return "<AP>"
        # Strip numeric tone/stress markers (e.g. "AY1" -> "AY")
        cleaned = re.sub(r"\d+$", "", ph)
        return cleaned if cleaned else "<UNK>"

    def encode(
        self, phonemes: List[str], max_len: int = 256, add_New_tokens: bool = True
    ) -> List[int]:
        ids = [self.token_to_id["<BOS>"]]
        for raw_ph in phonemes[: max_len - 2]:
            norm_ph = self.normalize_phoneme(raw_ph)
            if add_New_tokens:
                tid = self._add_token(norm_ph)
            else:
                tid = self.token_to_id.get(norm_ph, self.token_to_id["<UNK>"])
            ids.append(tid)
        ids.append(self.token_to_id["<EOS>"])

        # Pad to fixed length
        pad_id = self.token_to_id["<PAD>"]
        if len(ids) < max_len:
            ids.extend([pad_id] * (max_len - len(ids)))
        return ids[:max_len]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, max_vocab_size: int = 5000) -> "LyricPhonemeTokenizer":
        instance = cls(max_vocab_size=max_vocab_size)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        instance.token_to_id = {k: int(v) for k, v in loaded.items()}
        instance.id_to_token = {int(v): k for k, v in loaded.items()}
        return instance


class WhisperPhonemeAligner:
    """
    Extracts word-level timestamps from raw singing vocals using faster-whisper
    and subdivides each word into timed phonemes using G2P.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model = None
        if WhisperModel is not None:
            self.model = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        self.g2p = G2p() if G2p is not None else None

    def _word_to_phonemes(self, word: str) -> List[str]:
        clean = re.sub(r"[^a-zA-Z']", "", word).lower()
        if not clean:
            return ["<SP>"]
        if self.g2p is not None:
            raw_phs = [p for p in self.g2p(clean) if p.strip() and p != " "]
            if raw_phs:
                return [re.sub(r"\d+$", "", p) for p in raw_phs]
        # Character-level fallback if g2p_en is unavailable
        return [c.upper() for c in clean[:6]]

    def extract_timed_phonemes(
        self, audio_path: Path, language: Optional[str] = "en"
    ) -> List[TimedPhoneme]:
        if self.model is None:
            raise RuntimeError(
                "faster-whisper is not installed. Run `pip install faster-whisper`."
            )

        segments, _ = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        timed_phonemes: List[TimedPhoneme] = []
        last_end = 0.0

        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                w_start = float(w.start)
                w_end = max(w_start + 0.04, float(w.end))

                # Insert silence token if gap between words > 150ms
                if w_start - last_end > 0.15:
                    timed_phonemes.append(
                        TimedPhoneme(
                            phoneme="<SP>",
                            word="<SP>",
                            start_sec=last_end,
                            end_sec=w_start,
                        )
                    )

                phs = self._word_to_phonemes(w.word)
                sub_dur = (w_end - w_start) / max(1, len(phs))
                for idx, ph in enumerate(phs):
                    p_start = w_start + idx * sub_dur
                    p_end = p_start + sub_dur
                    timed_phonemes.append(
                        TimedPhoneme(
                            phoneme=ph,
                            word=w.word.strip(),
                            start_sec=round(p_start, 4),
                            end_sec=round(p_end, 4),
                        )
                    )
                last_end = w_end

        return timed_phonemes
