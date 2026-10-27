from pathlib import Path
import numpy as np
import pretty_midi
import scipy.signal as signal
import soundfile as sf


class DSPMixerAndRenderer:
    """
    Renders PrettyMIDI to WAV and performs vocal + instrumental DSP mixing
    (high-pass filtering, dynamic range compression, Schroeder reverb, and mastering limiter).
    """

    def __init__(
        self, sr: int = 44100, soundfont_path: Path = Path("assets/FluidR3_GM.sf2")
    ):
        self.sr = sr
        self.soundfont_path = soundfont_path

    def render_midi_to_audio(
        self, pm: pretty_midi.PrettyMIDI, target_duration: float
    ) -> np.ndarray:
        """Renders MIDI via FluidSynth if available, falling back to internal synthesis."""
        try:
            if self.soundfont_path.exists():
                audio = pm.fluidsynth(fs=self.sr, sf2_path=str(self.soundfont_path))
            else:
                audio = pm.synthesize(fs=self.sr)
        except Exception:
            audio = pm.synthesize(fs=self.sr)

        target_len = int(target_duration * self.sr)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        peak = np.max(np.abs(audio)) + 1e-8
        return (audio / peak * 0.8).astype(np.float32)

    def _highpass_filter(self, y: np.ndarray, cutoff_hz: float = 90.0) -> np.ndarray:
        sos = signal.butter(4, cutoff_hz, btype="highpass", fs=self.sr, output="sos")
        return signal.sosfilt(sos, y).astype(np.float32)

    def _soft_compressor(
        self, y: np.ndarray, threshold_db: float = -18.0, ratio: float = 3.0
    ) -> np.ndarray:
        thresh_lin = 10.0 ** (threshold_db / 20.0)
        env = np.abs(y) + 1e-8
        gain = np.where(
            env > thresh_lin,
            (thresh_lin + (env - thresh_lin) / ratio) / env,
            1.0,
        )
        return (y * gain).astype(np.float32)

    def _simple_reverb(self, y: np.ndarray, wet: float = 0.15) -> np.ndarray:
        delays_ms = [29.7, 37.1, 41.1, 43.7]
        wet_sig = np.zeros_like(y)
        for d_ms in delays_ms:
            delay_samples = int((d_ms / 1000.0) * self.sr)
            if delay_samples < len(y):
                padded = np.pad(y, (delay_samples, 0))[: len(y)]
                wet_sig += padded * 0.25
        return ((1.0 - wet) * y + wet * wet_sig).astype(np.float32)

    def mix_and_master(
        self,
        vocal_audio: np.ndarray,
        inst_audio: np.ndarray,
        output_path: Path,
        vocal_gain_db: float = 0.0,
        inst_gain_db: float = -4.5,
    ) -> Path:
        min_len = min(len(vocal_audio), len(inst_audio))
        v = vocal_audio[:min_len]
        inst = inst_audio[:min_len]

        # 1. Vocal chain: High-pass 90Hz -> Compression -> Subtle Reverb
        v_proc = self._highpass_filter(v, cutoff_hz=90.0)
        v_proc = self._soft_compressor(v_proc, threshold_db=-16.0, ratio=3.5)
        v_proc = self._simple_reverb(v_proc, wet=0.12)

        # 2. Instrumental chain: Glue compression
        inst_proc = self._soft_compressor(inst, threshold_db=-20.0, ratio=2.0)

        # 3. Level balance & summing
        v_gain = 10.0 ** (vocal_gain_db / 20.0)
        i_gain = 10.0 ** (inst_gain_db / 20.0)
        mix = (v_proc * v_gain) + (inst_proc * i_gain)

        # 4. Mastering soft-clipping limiter (-1 dBFS ceiling)
        mastered = np.tanh(mix * 1.15) * 0.89
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), mastered, self.sr, subtype="PCM_16")
        return output_path
