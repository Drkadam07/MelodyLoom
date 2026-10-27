import uuid
from pathlib import Path
from typing import Dict, Optional
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.audio.preprocess import VocalPreprocessor
from src.config import AudioConfig, ModelConfig, VocabConfig
from src.generation.mix import DSPMixerAndRenderer
from src.models.vocal2music_transformer import Vocal2MusicTransformer
from src.music.tokenizer import REMITokenizer

app = FastAPI(title="Vocal2Music Multimodal Generation API", version="1.0.0")

# Initialize singletons
vocab_cfg = VocabConfig()
tokenizer = REMITokenizer(vocab_cfg)

model_cfg = ModelConfig(music_vocab_size=tokenizer.vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Vocal2MusicTransformer(model_cfg).to(device)
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


def _run_generation_pipeline(
    job_id: str,
    input_wav_path: Path,
    genre: str,
    mood: str,
    key: str,
    bpm: float,
) -> None:
    try:
        JOBS_DB[job_id]["status"] = "processing"

        # 1. Audio Preprocessing & Pitch Extraction
        y_24k, y_44k = preprocessor.load_and_normalize(input_wav_path)
        melody_notes = preprocessor.extract_melody_notes(y_24k, bpm=bpm)
        duration_sec = len(y_44k) / 44100.0

        # 2. Construct Conditioning Tensors
        pitches = [
            min(127, max(0, n.midi)) for n in melody_notes[: model_cfg.max_melody_len]
        ] or [60]
        durs = [
            min(31, max(1, n.dur_16th))
            for n in melody_notes[: model_cfg.max_melody_len]
        ] or [4]

        melody_pitches = torch.tensor([pitches], dtype=torch.long, device=device)
        melody_durations = torch.tensor([durs], dtype=torch.long, device=device)
        lyric_tokens = torch.ones(
            (1, 16), dtype=torch.long, device=device
        )  # Placeholder for ASR tokens

        genre_idx = vocab_cfg.genres.index(genre) if genre in vocab_cfg.genres else 0
        mood_idx = vocab_cfg.moods.index(mood) if mood in vocab_cfg.moods else 0
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
    genre: str = Form("POP"),
    mood: str = Form("MELANCHOLIC"),
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
        genre.upper(),
        mood.upper(),
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
