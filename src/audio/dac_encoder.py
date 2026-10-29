from pathlib import Path
import torch
import dac
from audiotools import AudioSignal

class DACInstrumentalEncoder:
    """
    Encodes and decodes instrumental tracks into discrete multi-codebook
    tokens using the Descript Audio Codec (DAC).
    """
    def __init__(self, model_type: str = "44khz", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DAC model ({model_type}) on {self.device}...")
        
        # Automatically downloads official weights if not present locally
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_audio_to_tokens(self, audio_path: Path) -> torch.Tensor:
        """
        Loads an audio file, processes it, and compresses it into discrete DAC codes.
        
        Returns:
            codes: Tensor of shape (n_codebooks, time_frames) containing integer token IDs.
        """
        signal = AudioSignal(str(audio_path))
        
        # Ensure mono 44.1kHz standard
        if signal.sample_rate != self.model.sample_rate:
            signal.resample(self.model.sample_rate)
        if signal.num_channels > 1:
            signal.to_mono()

        signal.to(self.device)
        
        # Preprocess audio into model tensor format
        x = self.model.preprocess(signal.audio_data, signal.sample_rate)
        
        # Encode into continuous latents (z) and discrete vector-quantized codes
        # z, codes, latents, _, _ = self.model.encode(x)
        _, codes, _, _, _ = self.model.encode(x)
        
        # codes shape: (Batch=1, n_codebooks, time_frames)
        return codes.squeeze(0).cpu()

    @torch.no_grad()
    def decode_tokens_to_audio(self, codes: torch.Tensor, output_path: Path) -> Path:
        """
        Reconstructs an audio waveform file from discrete DAC tokens.
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0) # Add batch dimension -> (1, n_codebooks, time_frames)
            
        codes = codes.to(self.device)
        
        # Quantizer lookup to latent representation
        z = self.model.quantizer.from_codes(codes)[0]
        
        # Decode latents back to raw waveform
        y = self.model.decode(z)
        
        # Save reconstructed audio to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        y_signal = AudioSignal(y.cpu(), self.model.sample_rate)
        y_signal.write(str(output_path))
        
        print(f"Successfully reconstructed audio saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Smoke test execution
    encoder = DACInstrumentalEncoder(model_type="44khz")
    
    # Test path (replace with your path)
    test_audio = Path("storage/test_instrumental.wav")
    if test_audio.exists():
        tokens = encoder.encode_audio_to_tokens(test_audio)
        print(f"Encoded token shape (n_codebooks, time_frames): {tokens.shape}")
        
        recon_path = Path("storage/reconstructed_test.wav")
        encoder.decode_tokens_to_audio(tokens, recon_path)
    else:
        print(f"Place a test WAV file at {test_audio} to run encoding.")