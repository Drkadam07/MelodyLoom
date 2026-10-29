import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple

from src.audio.preprocess import VocalPreprocessor
from src.config import AudioConfig, ModelConfig, VocabConfig
from src.generation.mix import DSPMixerAndRenderer
from src.models.vocal2music_transformer import Vocal2MusicTransformer
from src.music.tokenizer import REMITokenizer

app = FastAPI(title="Vocal2Music Suno-Powered API", version="2.0.0")

# Initialize singletons
vocab_cfg = VocabConfig()
tokenizer = REMITokenizer(vocab_cfg)

model_cfg = ModelConfig(music_vocab_size=tokenizer.vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Vocal2MusicTransformer(model_cfg).to(device)

# --- LOAD TRAINED WEIGHTS ---
ckpt_path = Path("checkpoints/best_model.pt")
if ckpt_path.exists():
    state_dict = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    print(f"✅ Loaded trained weights from {ckpt_path}")
else:
    print("⚠️ Warning: Checkpoint not found. Running with uninitialized weights.")

model.eval()

preprocessor = VocalPreprocessor(AudioConfig())
mixer = DSPMixerAndRenderer(sr=44100)

JOBS_DB: Dict[str, Dict] = {}
STORAGE_DIR = Path("storage/jobs")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class JobResponse(BaseModel):
    job_id: str
    status: str
    audio_url: Optional[str] = None
    error: Optional[str] = None


def parse_structural_lyrics(raw_text: str) -> List[str]:
    """Parses Suno-style tags like [Verse], [Chorus] and returns structured sections/tokens."""
    sections = re.split(r"(\[.*?\])", raw_text)
    tokens = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if sec.startswith("[") and sec.endswith("]"):
            tag_name = sec[1:-1].upper()
            tokens.append(f"SECTION_{tag_name}")
        else:
            # Tokenize words/lines
            tokens.extend(sec.split())
    return tokens if tokens else ["SECTION_VERSE", "Vocals"]


def style_prompt_to_embeddings(style_prompt: str) -> Tuple[int, int]:
    """Maps free-text style prompts to genre/mood indices via keyword analysis."""
    prompt_lower = style_prompt.lower()

    # Genre mapping
    if "rock" in prompt_lower:
        genre_idx = 1
    elif "lofi" in prompt_lower or "lo-fi" in prompt_lower:
        genre_idx = 2
    elif "edm" in prompt_lower or "dance" in prompt_lower:
        genre_idx = 3
    elif "ballad" in prompt_lower or "acoustic" in prompt_lower:
        genre_idx = 4
    else:
        genre_idx = 0  # Default POP

    # Mood mapping
    if "happy" in prompt_lower or "uplifting" in prompt_lower:
        mood_idx = 1
    elif "sad" in prompt_lower or "depressing" in prompt_lower:
        mood_idx = 2
    elif "energetic" in prompt_lower or "hype" in prompt_lower:
        mood_idx = 3
    elif "calm" in prompt_lower or "relax" in prompt_lower:
        mood_idx = 4
    else:
        mood_idx = 0  # Default MELANCHOLIC

    return genre_idx, mood_idx


def _run_generation_pipeline(
    job_id: str,
    input_wav_path: Path,
    style_prompt: str,
    structural_lyrics: str,
    key: str,
    bpm: float,
) -> None:
    try:
        JOBS_DB[job_id]["status"] = "processing"

        # 1. Audio Preprocessing & Pitch Extraction
        y_24k, y_44k = preprocessor.load_and_normalize(input_wav_path)
        melody_notes = preprocessor.extract_melody_notes(y_24k, bpm=bpm)
        duration_sec = len(y_44k) / 44100.0

        # 2. Parse Style Prompt & Structural Lyrics
        genre_idx, mood_idx = style_prompt_to_embeddings(style_prompt)
        parsed_tokens = parse_structural_lyrics(structural_lyrics)

        # Simple token hashing to ID for embedding lookup
        lyric_ids = [
            abs(hash(t)) % (model_cfg.lyrics_vocab_size - 1) + 1 for t in parsed_tokens
        ]
        lyric_ids = lyric_ids[: model_cfg.max_lyrics_len]
        if len(lyric_ids) < model_cfg.max_lyrics_len:
            lyric_ids.extend([0] * (model_cfg.max_lyrics_len - len(lyric_ids)))

        pitches = [
            min(127, max(0, n.midi)) for n in melody_notes[: model_cfg.max_melody_len]
        ] or [60]
        durs = [
            min(31, max(1, n.dur_16th))
            for n in melody_notes[: model_cfg.max_melody_len]
        ] or [4]

        melody_pitches = torch.tensor([pitches], dtype=torch.long, device=device)
        melody_durations = torch.tensor([durs], dtype=torch.long, device=device)
        lyric_tokens = torch.tensor([lyric_ids], dtype=torch.long, device=device)

        key_idx = vocab_cfg.keys.index(key) if key in vocab_cfg.keys else 0

        # 3. Multimodal Transformer Generation
        generated_ids = model.generate(
            lyric_tokens=lyric_tokens,
            melody_pitches=melody_pitches,
            melody_durations=melody_durations,
            genre_id=torch.tensor([genre_idx], dtype=torch.long, device=device),
            mood_id=torch.tensor([mood_idx], dtype=torch.long, device=device),
            key_id=torch.tensor([key_idx], dtype=torch.long, device=device),
            norm_tempo=torch.tensor([bpm / 180.0], dtype=torch.float32, device=device),
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=512,
        )

        # 4. Token-to-MIDI Conversion & Audio Rendering
        token_list = generated_ids[0].cpu().tolist()
        pm = tokenizer.tokens_to_midi(token_list, default_bpm=bpm)
        midi_out_path = STORAGE_DIR / job_id / "accompaniment.mid"
        pm.write(str(midi_out_path))

        inst_audio = mixer.render_midi_to_audio(pm, target_duration=duration_sec)

        # 5. Vocal + Instrumental Mixing & Mastering
        final_wav_path = STORAGE_DIR / job_id / "final_song.wav"
        mixer.mix_and_master(y_44k, inst_audio, final_wav_path)

        JOBS_DB[job_id]["status"] = "completed"
        JOBS_DB[job_id]["audio_url"] = f"/api/v1/download/{job_id}"

    except Exception as exc:
        JOBS_DB[job_id]["status"] = "failed"
        JOBS_DB[job_id]["error"] = str(exc)


@app.post("/api/v1/generate", response_model=JobResponse)
async def create_generation_job(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    style_prompt: str = Form("90s Synth-Pop, driving bass, melancholic mood"),
    structural_lyrics: str = Form(
        "[Verse]\nI walk alone beneath the city lights...\n\n[Chorus]\nAnd the city shines bright..."
    ),
    key: str = Form("C"),
    bpm: float = Form(96.0),
):
    job_id = str(uuid.uuid4())
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_wav_path = job_dir / "input_vocal.wav"
    input_wav_path.write_bytes(await audio.read())

    JOBS_DB[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "audio_url": None,
        "error": None,
    }
    background_tasks.add_task(
        _run_generation_pipeline,
        job_id,
        input_wav_path,
        style_prompt,
        structural_lyrics,
        key.upper(),
        bpm,
    )
    return JobResponse(**JOBS_DB[job_id])


@app.get("/api/v1/generate/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**JOBS_DB[job_id])


@app.get("/api/v1/download/{job_id}")
async def download_generated_song(job_id: str):
    final_path = STORAGE_DIR / job_id / "final_song.wav"
    if not final_path.exists():
        raise HTTPException(status_code=404, detail="Generated audio file not ready")
    return FileResponse(
        str(final_path), media_type="audio/wav", filename=f"vocal2music_{job_id}.wav"
    )
