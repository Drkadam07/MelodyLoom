from pathlib import Path
import torch
import pretty_midi

from src.config import ModelConfig, VocabConfig
from src.models.vocal2music_transformer import Vocal2MusicTransformer
from src.music.tokenizer import REMITokenizer
from src.audio.preprocess import VocalPreprocessor
from src.generation.mix import DSPMixerAndRenderer

def run_inference(input_wav: Path, output_wav: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize tokenizer and model architecture
    v_cfg = VocabConfig()
    tokenizer = REMITokenizer(v_cfg)
    m_cfg = ModelConfig(music_vocab_size=tokenizer.vocab_size)
    
    model = Vocal2MusicTransformer(m_cfg).to(device)
    
    # 2. Load trained weights from Kaggle
    ckpt_path = Path("checkpoints/best_model.pt")
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device)
        # Handle if saved as raw state_dict or full checkpoint dict
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print("Loaded trained model weights successfully!")
    else:
        print("Warning: Checkpoint not found. Running with uninitialized weights.")
        
    model.eval()

    # 3. Preprocess input vocal
    preprocessor = VocalPreprocessor()
    y_24k, y_44k = preprocessor.load_and_normalize(input_wav)
    melody_notes = preprocessor.extract_melody_notes(y_24k, bpm=96.0)
    
    # 4. Prepare conditioning tensors
    pitches = [min(127, max(0, n.midi)) for n in melody_notes[:m_cfg.max_melody_len]] or [60]
    durs = [min(31, max(1, n.dur_16th)) for n in melody_notes[:m_cfg.max_melody_len]] or [4]
    
    melody_pitches = torch.tensor([pitches], dtype=torch.long, device=device)
    melody_durations = torch.tensor([durs], dtype=torch.long, device=device)
    lyric_tokens = torch.ones((1, 16), dtype=torch.long, device=device)
    
    # 5. Generate tokens autoregressively
    with torch.no_grad():
        generated_ids = model.generate(
            lyric_tokens=lyric_tokens,
            melody_pitches=melody_pitches,
            melody_durations=melody_durations,
            genre_id=torch.tensor([0], dtype=torch.long, device=device),
            mood_id=torch.tensor([2], dtype=torch.long, device=device),
            key_id=torch.tensor([0], dtype=torch.long, device=device),
            norm_tempo=torch.tensor([96.0 / 180.0], dtype=torch.float32, device=device),
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=256,
        )
        
    # 6. Render MIDI and mix with vocal
    token_list = generated_ids[0].cpu().tolist()
    pm = tokenizer.tokens_to_midi(token_list, default_bpm=96.0)
    
    mixer = DSPMixerAndRenderer(sr=44100)
    duration_sec = len(y_44k) / 44100.0
    inst_audio = mixer.render_midi_to_audio(pm, target_duration=duration_sec)
    
    mixer.mix_and_master(y_44k, inst_audio, output_wav)
    print(f"Final mastered song saved to {output_wav}")

if __name__ == "__main__":
    run_inference(Path("assets/sample_vocal.wav"), Path("output_song.wav"))