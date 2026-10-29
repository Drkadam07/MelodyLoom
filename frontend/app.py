import time
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Vocal2Music: Suno-Powered AI Generator",
    page_icon="🎵",
    layout="centered",
)

st.title("🎵 Vocal2Music: Suno-Style AI Generator")
st.markdown(
    "Upload your raw vocal recording, describe your desired musical style in natural language, "
    "and structure your song with tags like `[Verse]`, `[Chorus]`, and `[Bridge]`."
)

with st.sidebar:
    st.header("🎛️ Composition Settings")
    key = st.selectbox(
        "Musical Key",
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        index=0,
    )
    bpm = st.slider("Tempo (BPM)", min_value=60, max_value=180, value=96, step=4)

    st.markdown("---")
    st.markdown("### 🛠️ Backend Status")
    try:
        res = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/docs", timeout=2)
        if res.status_code == 200:
            st.success("API Server: Online 🟢")
        else:
            st.warning("API Server: Degraded 🟡")
    except Exception:
        st.error("API Server: Offline 🔴")

# Free-text style prompt (Suno style)
style_prompt = st.text_input(
    "🎨 Style Prompt (Genre & Vibe)",
    value="90s synth-pop, driving bass, melancholic mood, atmospheric pads",
    help="Describe the instruments, era, genre, and mood in plain text.",
)

# Structural lyrics with tags
structural_lyrics = st.text_area(
    "📝 Structural Lyrics ([Verse], [Chorus], etc.)",
    value="[Verse]\nI walk alone beneath the city lights...\n\n[Chorus]\nAnd the city shines bright tonight...",
    height=150,
    help="Use [Verse], [Chorus], [Bridge], or [Outro] tags to guide the arrangement.",
)

uploaded_file = st.file_uploader(
    "🎤 Upload Vocal Audio (.wav or .mp3)", type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🚀 Generate Suno-Style Song"):
        with st.spinner("Uploading audio and generating song arrangement..."):
            files = {
                "audio": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")
            }
            data = {
                "style_prompt": style_prompt,
                "structural_lyrics": structural_lyrics,
                "key": key,
                "bpm": float(bpm),
            }

            try:
                response = requests.post(
                    f"{API_BASE_URL}/generate", files=files, data=data, timeout=10
                )
                if response.status_code != 200:
                    st.error(f"Error: {response.text}")
                    st.stop()
                job_id = response.json()["job_id"]
            except Exception as e:
                st.error(f"Connection failed: {e}")
                st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        steps = [
            "Analyzing audio and extracting pitch contour...",
            "Parsing Suno-style structural tags and style prompt...",
            "Conditioning Multimodal Transformer & generating accompaniment tokens...",
            "Rendering MIDI tracks and synthesizing instruments...",
            "Applying studio mixing, EQ, compression & mastering...",
        ]

        completed = False
        step_idx = 0
        audio_url = None

        while not completed:
            try:
                res = requests.get(
                    f"{API_BASE_URL}/generate/{job_id}", timeout=5
                ).json()
                status = res["status"]
                if status == "completed":
                    progress_bar.progress(100)
                    status_text.success("✨ Song generation completed successfully!")
                    audio_url = res["audio_url"]
                    completed = True
                elif status == "failed":
                    st.error(f"Failed: {res.get('error')}")
                    st.stop()
                else:
                    step_idx = min(step_idx + 1, len(steps) - 1)
                    progress_bar.progress(min(90, (step_idx + 1) * 18))
                    status_text.text(steps[step_idx])
                    time.sleep(2.5)
            except Exception:
                time.sleep(2.0)

        if audio_url:
            full_url = f"http://localhost:8000{audio_url}"
            st.markdown("### 🎵 Final Mastered Track")
            audio_bytes = requests.get(full_url).content
            st.audio(audio_bytes, format="audio/wav")
            st.download_button(
                "📥 Download Mastered WAV",
                data=audio_bytes,
                file_name="suno_output.wav",
                mime="audio/wav",
            )
