import time
import requests
import streamlit as st

# FastAPI Backend URL
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Vocal2Music: AI Song Generator",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎵 Vocal2Music AI Accompaniment Generator")
st.markdown(
    "Upload a clean human vocal recording (singing or recitation without background music), "
    "select your musical styling preferences, and let the **Multimodal Transformer** generate "
    "a fully synchronized, mixed, and mastered instrumental accompaniment."
)

with st.sidebar:
    st.header("🎛️ Style & Composition Controls")
    genre = st.selectbox("Genre", ["POP", "ROCK", "LOFI", "EDM", "BALLAD"], index=0)
    mood = st.selectbox(
        "Mood", ["MELANCHOLIC", "HAPPY", "SAD", "ENERGETIC", "CALM"], index=0
    )
    key = st.selectbox(
        "Musical Key",
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        index=0,
    )
    bpm = st.slider("Tempo (BPM)", min_value=60, max_value=180, value=96, step=4)

    st.markdown("---")
    st.markdown("### 🛠️ System Status")
    try:
        health_res = requests.get(
            f"{API_BASE_URL.replace('/api/v1', '')}/docs", timeout=2
        )
        if health_res.status_code == 200:
            st.success("FastAPI Backend: Online 🟢")
        else:
            st.warning("FastAPI Backend: Degraded 🟡")
    except Exception:
        st.error("FastAPI Backend: Offline 🔴 (Start uvicorn)")

uploaded_file = st.file_uploader(
    "Upload Vocal Audio (.wav or .mp3)",
    type=["wav", "mp3"],
    help="Provide a clean vocal recording.",
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info(
        f"File loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)"
    )

    if st.button("🚀 Generate Full Song"):
        with st.spinner("Uploading vocal and initializing generation job..."):
            files = {
                "audio": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")
            }
            data = {
                "genre": genre,
                "mood": mood,
                "key": key,
                "bpm": float(bpm),
            }

            try:
                response = requests.post(
                    f"{API_BASE_URL}/generate", files=files, data=data, timeout=10
                )
                if response.status_code != 200:
                    st.error(f"Generation request failed: {response.text}")
                    st.stop()

                job_data = response.json()
                job_id = job_data["job_id"]
            except Exception as e:
                st.error(f"Connection error to backend: {e}")
                st.stop()

        # Progress tracking loop
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_steps = [
            "Preprocessing audio & extracting pYIN pitch contour...",
            "Running Whisper ASR & phoneme alignment...",
            "Conditioning Multimodal Transformer & generating REMI tokens...",
            "Rendering MIDI accompaniment via SoundFont synth...",
            "Applying DSP mixing, EQ, compression & mastering limiter...",
        ]

        completed = False
        step_idx = 0
        audio_url = None

        while not completed:
            try:
                status_res = requests.get(
                    f"{API_BASE_URL}/generate/{job_id}", timeout=5
                )
                status_json = status_res.json()
                status = status_json["status"]

                if status == "completed":
                    progress_bar.progress(100)
                    status_text.success("✨ Song generation completed successfully!")
                    audio_url = status_json["audio_url"]
                    completed = True
                elif status == "failed":
                    progress_bar.progress(100)
                    st.error(
                        f"Generation failed: {status_json.get('error', 'Unknown error')}"
                    )
                    st.stop()
                else:
                    step_idx = min(step_idx + 1, len(status_steps) - 1)
                    progress_bar.progress(min(90, (step_idx + 1) * 18))
                    status_text.text(status_steps[step_idx])
                    time.sleep(2.5)
            except Exception:
                time.sleep(2.0)

        if audio_url:
            full_audio_url = f"http://localhost:8000{audio_url}"
            st.markdown("### 🎵 Final Mastered Song (`final_song.wav`)")

            # Fetch and play audio stream
            try:
                audio_bytes = requests.get(full_audio_url).content
                st.audio(audio_bytes, format="audio/wav")

                st.download_button(
                    label="📥 Download Mastered WAV",
                    data=audio_bytes,
                    file_name="vocal2music_mastered.wav",
                    mime="audio/wav",
                )
            except Exception as e:
                st.error(f"Could not retrieve audio stream: {e}")

else:
    st.markdown("""
        > **Getting Started:**
        > 1. Start your FastAPI server: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
        > 2. Run this Streamlit frontend: `streamlit run frontend/app.py`
        > 3. Upload a vocal clip, choose your style parameters, and click **Generate Full Song**!
        """)
