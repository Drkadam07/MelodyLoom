# 🎵 MelodyLoom: Multimodal Deep Learning for Automatic Musical Accompaniment Generation

MelodyLoom is an end-to-end generative AI project that transforms raw human vocal recordings into complete musical arrangements. Given a vocal recording containing lyrics and melody, the system aims to generate synchronized instrumental accompaniment, including drums, bass, chords, and melodic layers, followed by audio rendering, mixing, and mastering.

The system combines pretrained audio and speech-processing components with a custom Multimodal Cross-Attention Transformer that learns to coordinate lyrics, vocal melody, and musical structure.

**Project status:** Research and development. Output quality, real-time generation, and production-grade mastering depend on the trained model, dataset quality, rendering setup, and implementation maturity.

---

## ✨ Key Features

- 🎤 **Vocal-driven generation**: Uses a human vocal recording as the primary input.
- 📝 **Lyrics extraction**: Supports automatic speech recognition and word-level timestamp extraction.
- 🎼 **Melody analysis**: Extracts vocal pitch contours using pYIN.
- 🧠 **Multimodal learning**: Combines lyrics and melody features through cross-attention.
- 🎹 **Symbolic music generation**: Generates structured musical events using a REMI-inspired token representation.
- 🥁 **Instrumental arrangement**: Supports the generation of drums, bass, chords, and additional instrumental parts.
- 🎧 **MIDI-to-audio rendering**: Uses SoundFont-based synthesis to render generated MIDI.
- 🎚️ **Audio post-processing**: Provides a foundation for equalization, compression, reverb, and loudness optimization.
- 🌐 **Web interface**: Provides an interactive Streamlit interface for uploading vocals and configuring generation.
- ⚡ **REST API**: Exposes generation functionality through a FastAPI backend.
- 📊 **Musical evaluation**: Tracks harmony compatibility and beat synchronization metrics.

---

## 🏗️ System Architecture

```
┌────────────────────────┐
│   Raw Vocal Recording  │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│   Audio Preprocessing  │
│  Resampling / Normalize│
└────────────┬───────────┘
             │
   ┌─────────┴─────────┐
   ▼                   ▼
┌────────────────────┐ ┌────────────────────┐
│   Lyrics / ASR     │ │   Vocal Pitch      │
│ Whisper Timestamps │ │   pYIN Contour     │
└──────────┬─────────┘ └──────────┬─────────┘
           │                      │
           └──────────┬───────────┘
                      ▼
           ┌────────────────────────┐
           │   Multimodal Cross-    │
           │ Attention Transformer  │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │   REMI-Inspired Music  │
           │ Event Token Generation │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │     MIDI Generation    │
           │ & SoundFont Rendering  │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │  DSP Mixing & Mastering│
           │ EQ / Dynamics / Reverb │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │   Final Audio Output   │
           └────────────────────────┘
```

### Processing Pipeline

1. **Audio preprocessing**: Loads the vocal recording, resamples the waveform, normalizes audio levels, and prepares model inputs.
2. **Lyrics extraction**: Uses automatic speech recognition to estimate the words and their timestamps. A grapheme-to-phoneme (G2P) tokenizer can convert text into phonetic representations.
3. **Melody extraction**: Applies pYIN pitch tracking to estimate the vocal fundamental frequency and identify voiced regions.
4. **Multimodal encoding**: Encodes linguistic and melodic information into representations that can be combined by cross-attention.
5. **Music token generation**: Predicts a sequence of structured musical events representing the accompaniment.
6. **MIDI rendering**: Converts symbolic music events into MIDI and synthesizes audio through a SoundFont-compatible renderer.
7. **Mixing and mastering**: Applies configurable signal processing to produce the final audio output.

---

## 🧠 Model Architecture

The core model is a multimodal Transformer designed to coordinate linguistic context and melodic information.

### Input Modalities

| Modality | Processing component | Purpose |
|----------|---------------------|---------|
| Vocal audio | Audio preprocessing | Normalize and prepare the waveform |
| Lyrics | ASR and G2P | Represent linguistic and phonetic content |
| Vocal melody | pYIN | Represent pitch and melodic movement |
| Audio features (optional) | DAC encoder | Provide learned audio representations |
| Musical events | REMI-inspired tokenizer | Represent rhythmic and harmonic structure |

### Multimodal Cross-Attention Transformer

The model uses separate feature representations for lyrics and melody, allowing one modality to attend to information from the other.

Conceptually, cross-attention is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Here, queries, keys, and values are learned projections of the relevant modality representations.

The model is intended to learn relationships between:

- Lyrics and musical phrasing.
- Vocal pitch and chord selection.
- Syllable timing and rhythmic placement.
- Musical structure and accompaniment density.
- Generated events and instrumental arrangement.

The model predicts a sequence of music tokens that can subsequently be decoded into symbolic musical events.

> **Important:** A cross-attention architecture alone does not guarantee musically coherent arrangements. This requires suitable training examples, correct temporal alignment, appropriate target representations, and effective generation constraints.

---



## ⚙️ Requirements

Recommended environment:

- Python 3.10 or 3.11, subject to dependency compatibility.
- PyTorch with CPU or CUDA support.
- FFmpeg or another compatible audio decoding backend, where required.
- FluidSynth for SoundFont-based audio synthesis.
- A compatible SoundFont file.
- Sufficient memory and storage for audio data, model checkpoints, and intermediate tensors.

A CUDA-enabled GPU is recommended for model training and larger inference workloads. CPU execution may be possible for smaller workloads, depending on the selected components.

---

## 📊 Evaluation Metrics

Vocal2Music is designed to evaluate both harmonic compatibility and rhythmic alignment.

### 1. Harmony Compatibility Score

Measures how frequently sung vocal pitch classes agree with the active accompaniment harmony.

A simplified metric is:

$$H = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[p_i \in C_i] \times 100$$

Where:

- $N$ is the number of valid evaluated vocal frames or notes.
- $p_i$ is the vocal pitch class at evaluation position $i$.
- $C_i$ is the set of pitch classes in the active accompaniment chord.
- $\mathbb{1}[\cdot]$ is an indicator function.

Higher values indicate greater pitch-class compatibility under the chosen chord representation. This metric does not fully capture harmonic quality, voice leading, dissonance treatment, or musical style.

### 2. Beat Synchronization Error

Measures the temporal distance between generated note onsets and their corresponding rhythmic grid positions.

$$E_{\text{beat}} = \frac{1}{M} \sum_{j=1}^{M} |t_j - \hat{t}_j|$$

Where:

- $M$ is the number of evaluated note onsets.
- $t_j$ is a generated note onset.
- $\hat{t}_j$ is its corresponding reference beat-grid position.

The result is expressed in seconds. Lower values indicate closer alignment with the selected grid.

For meaningful evaluation, define how note onsets are matched to beat positions and how unmatched events are handled. A note-onset grid error is not the same as a full beat-tracking error.

### Recommended Additional Metrics

| Metric | Purpose |
|--------|---------|
| Chroma similarity | Compare harmonic content between generated and reference accompaniment |
| Onset F1 score | Evaluate agreement between generated and reference note onsets |
| Note precision / recall | Measure symbolic note-generation accuracy |
| Melodic contour similarity | Compare pitch movement and melodic shape |
| Audio embedding similarity | Compare high-level musical characteristics |
| Human listening evaluation | Assess musicality, arrangement quality, and perceived synchronization |

Objective scores should be complemented by listening tests because musical quality cannot be fully captured by a small set of numerical metrics.

---

## 🛠️ Configuration

Training and model parameters are intended to be managed through:

```
configs/training.yaml
```

Depending on the implementation, configurable parameters may include:

- Audio sample rate.
- Model dimensions and attention heads.
- Transformer depth.
- Maximum sequence length.
- Batch size and gradient accumulation.
- Learning rate and weight decay.
- Number of training epochs.
- Mixed-precision settings.
- Dataset and checkpoint paths.
- Generation temperature and sampling strategy.
- MIDI tempo and rendering settings.
- Mixing and mastering parameters.

Confirm the supported configuration keys in `src/config.py` and `configs/training.yaml` before changing them.

---

## 🔍 Troubleshooting

### Missing Python dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-data.txt
```

If installation fails, verify your Python version and the compatibility of PyTorch, audio libraries, and other pinned packages.

### FluidSynth or SoundFont errors

- Confirm that FluidSynth is installed.
- Verify that `assets/FluidR3_GM.sf2` exists.
- Check that the SoundFont is readable and supported by the renderer.
- Verify that the relevant Python bindings or subprocess configuration are installed.

### CUDA out-of-memory errors

- Reduce the batch size.
- Reduce the maximum sequence length.
- Enable supported mixed-precision training.
- Use gradient accumulation where appropriate.
- Reduce model dimensions if necessary.

### Dataset files are missing

Verify that preprocessing completed successfully and produced the expected files:

```
data/processed/train.pt
data/processed/val.pt
data/processed/test.pt
```

Inspect the dataset builder's logs and output schema if files are absent or incompatible.

### Generated audio sounds incorrect

Check the following:

- Vocal pitch-tracking accuracy.
- Lyrics and timestamp alignment.
- MIDI tokenizer vocabulary consistency.
- Model checkpoint compatibility.
- Generated event decoding.
- MIDI tempo and instrument assignments.
- SoundFont availability and rendering settings.
- Audio mixing levels and clipping.

---

## 🔐 Data, Privacy, and Responsible Use

- Use vocal recordings only when you have the necessary rights and permissions.
- Obtain appropriate consent before processing identifiable voices.
- Review the licenses of datasets, pretrained models, SoundFonts, and third-party dependencies.
- Avoid uploading confidential or sensitive recordings to public deployments.
- Apply file validation, upload-size limits, and access controls when exposing the API.
- Do not assume that a dataset's research availability automatically permits commercial use.

---

## 🚧 Current Limitations and Future Work

Potential areas for future development include:

- Stronger alignment between phonemes, vocal notes, and accompaniment events.
- Better chord inference and voice-leading constraints.
- Multi-track generation with explicit instrument conditioning.
- Improved handling of tempo changes, expressive timing, and rubato.
- Joint optimization of symbolic generation and audio synthesis.
- Better separation of vocals from recordings that already contain music.
- More robust evaluation across genres and singing styles.
- Perceptual audio-quality evaluation and listening-test benchmarks.
- Improved inference speed and deployment scalability.
- Integration of neural audio generation for more realistic instrumental timbres.

These are development goals rather than guaranteed capabilities of every implementation.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Document configuration changes and new dependencies.
5. Submit a pull request describing the changes and their validation.

When reporting issues, include the Python version, operating system, relevant dependency versions, error logs, and a minimal reproducible example where possible.

---

## 📜 License

This project is intended to be distributed under the MIT License. See [LICENSE](LICENSE) for the complete terms.

Ensure that the repository's actual license file and all bundled datasets, model weights, and audio assets permit their intended use.

---

## 🙌 Acknowledgments

Vocal2Music builds on ideas and tools from the open-source audio and machine-learning ecosystem, including:

- [PyTorch](https://pytorch.org/) — deep learning and model training.
- [Whisper](https://github.com/openai/whisper) — speech recognition and transcription.
- [pYIN](https://github.com/librosa/librosa) — probabilistic fundamental-frequency estimation.
- [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec) — neural audio representation, when configured.
- REMI-inspired representations — structured symbolic music tokenization.
- [FluidSynth](https://www.fluidsynth.org/) — SoundFont-based MIDI synthesis.
- [FastAPI](https://fastapi.tiangolo.com/) — asynchronous API development.
- [Streamlit](https://streamlit.io/) — interactive web application development.

Refer to each project's official documentation and license for implementation details and usage conditions.

---

**Vocal2Music — From human voice to musical accompaniment. 🎶**

*An experimental multimodal AI framework for exploring the relationship between lyrics, melody, and automatic music arrangement.*