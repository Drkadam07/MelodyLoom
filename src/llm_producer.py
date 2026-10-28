import json
import os
from typing import Dict, Any
from openai import OpenAI  # Or use HuggingFace / Local LLM


class LLMMusicProducer:
    """
    Acts as a virtual music producer. Takes extracted lyrics, pitch contours,
    and audio metadata, then uses an LLM to determine the ideal emotional vibe,
    genre sub-style, chord progressions, and professional instrumentation.
    """

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate_arrangement_blueprint(
        self, lyrics: str, detected_bpm: float, detected_key: str
    ) -> Dict[str, Any]:
        prompt = f"""
        You are a professional Grammy-winning music producer and arranger.
        Analyze the following song details and create a professional production blueprint.

        Song Details:
        - Lyrics: "{lyrics}"
        - Tempo (BPM): {detected_bpm}
        - Detected Key: {detected_key}

        Respond ONLY with a valid JSON object using this exact schema:
        {{
            "primary_genre": "string (e.g., Synth-Pop, Indie Rock, Lo-Fi Hip-Hop)",
            "emotional_mood": "string (e.g., Melancholic, Uplifting, Moody, Energetic)",
            "arrangement_style": "string (e.g., Build from acoustic guitar to full driving beat at chorus)",
            "instruments": [
                {{"track": "Drums", "pattern": "string", "intensity": "low/med/high"}},
                {{"track": "Bass", "pattern": "string", "timbre": "string"}},
                {{"track": "Harmony/Pads", "chords": ["C", "Am", "F", "G"], "synth_type": "string"}}
            ],
            "mixing_notes": {{
                "vocal_effects": "string (e.g., plate reverb, warm compression)",
                "mastering_limiter_ceiling_db": -1.0
            }}
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            return json.loads(response.choices.message.content)
        except Exception as e:
            # Fallback default blueprint if API is offline
            return {
                "primary_genre": "Pop",
                "emotional_mood": "Melancholic",
                "arrangement_style": "Standard 4/4 pop ballad arrangement",
                "instruments": [
                    {"track": "Drums", "pattern": "Standard 4/4 kick-snare", "intensity": "med"},
                    {"track": "Bass", "pattern": "Root notes", "timbre": "sub bass"},
                    {"track": "Harmony/Pads", "chords": [detected_key, "Am", "F", "G"], "synth_type": "warm pad"}
                ],
                "mixing_notes": {"vocal_effects": "standard reverb", "mastering_limiter_ceiling_db": -1.0}
            }