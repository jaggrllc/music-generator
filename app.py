import streamlit as st
import numpy as np
from scipy.io import wavfile
import io
import os
import warnings
warnings.filterwarnings("ignore")

# Graceful pydub import with audioop-lts support
try:
    from pydub import AudioSegment, effects
    from pydub.effects import normalize as pydub_normalize
    PYDUB_AVAILABLE = True
except Exception as e:
    st.error(f"pydub failed to load: {e}. WAV export still works. Install audioop-lts or use Python 3.12.")
    PYDUB_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from serpapi import GoogleSearch
    SERP_AVAILABLE = True
except ImportError:
    SERP_AVAILABLE = False

SAMPLE_RATE = 44100
MAX_DURATION_MIN = 45  # safer limit for cloud

st.set_page_config(page_title="Original Sleep Music Generator", layout="wide")
st.title("Copyright-Free Sleep & Ambient Music Generator")
st.caption("Fully local synthesis • 432 Hz • 528 Hz • Binaural Beats • Export WAV / MP3")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Optional API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    serp_key = st.text_input("SerpAPI Key", type="password", key="serp_key")
    st.divider()
    st.info("All audio is generated locally from pure math. No samples are used.")

# ---------- Controls ----------
col1, col2 = st.columns(2)
with col1:
    duration_min = st.slider("Duration (minutes)", 1, MAX_DURATION_MIN, 8)
    base_freq = st.selectbox("Base Frequency", [432, 528, 440, 220, "Custom"])
    if base_freq == "Custom":
        base_freq = st.number_input("Custom Hz", 40.0, 1200.0, 432.0, step=1.0)
    beat_freq = st.slider("Binaural Beat (Hz)", 0.5, 8.0, 2.0, 0.1)

with col2:
    genre = st.selectbox(
        "Genre / Style",
        [
            "Deep Sleep 432Hz Binaural Delta",
            "528Hz Healing Meditation",
            "Soft Pink Noise + Binaural",
            "Lo-fi Ambient",
            "Cello & Piano Soft Pads",
            "Classical Ambient",
            "Warm Drone + Heartbeat",
        ]
    )

st.subheader("Audio Processing")
c1, c2, c3, c4 = st.columns(4)
with c1:
    do_normalize = st.checkbox("Normalize", value=True)
with c2:
    fade_sec = st.slider("Fade in/out (sec)", 0, 20, 4)
with c3:
    bass_boost_db = st.slider("Bass boost (dB)", -10, 10, 0)
with c4:
    high_shelf_db = st.slider("High shelf (dB)", -6, 6, 0)

if st.button("Generate Music", type="primary"):
    duration_sec = duration_min * 60
    base = float(base_freq)

    with st.spinner("Generating original music…"):
        def noise(duration, kind="pink", amp=0.25):
            n = int(duration * SAMPLE_RATE)
            white = np.random.randn(n).astype(np.float32)
            if kind == "pink":
                sig = np.cumsum(white)
            elif kind == "brown":
                sig = np.cumsum(np.cumsum(white))
            else:
                sig = white
            return sig / (np.max(np.abs(sig)) + 1e-8) * amp

        def binaural(duration, carrier, beat, amp=0.2):
            n = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, n, endpoint=False)
            left = np.sin(2 * np.pi * carrier * t) * amp
            right = np.sin(2 * np.pi * (carrier + beat) * t) * amp
            return np.column_stack([left, right])

        def pad(duration, freq, harmonics=5, amp=0.18):
            n = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, n, endpoint=False)
            wave = np.zeros_like(t)
            for h in range(1, harmonics + 1):
                wave += np.sin(2 * np.pi * freq * h * t) / (h ** 1.2)
            lfo = 0.65 + 0.35 * np.sin(2 * np.pi * 0.04 * t)
            wave *= lfo * amp / harmonics
            return np.column_stack([wave, wave])

        # Layer selection by genre
        if "Deep Sleep" in genre or "Pink Noise" in genre:
            mix = binaural(duration_sec, base, beat_freq, 0.16)
            mix += np.column_stack([noise(duration_sec, "pink", 0.22)] * 2)
            mix += pad(duration_sec, base * 0.5, 4, 0.12)
        elif "528" in genre or "Healing" in genre or "Meditation" in genre:
            mix = binaural(duration_sec, 528 if base == 528 else base, beat_freq * 0.6, 0.14)
            mix += pad(duration_sec, base * 0.5, 6, 0.2)
            mix += np.column_stack([noise(duration_sec, "pink", 0.1)] * 2)
        elif "Lo-fi" in genre:
            mix = pad(duration_sec, base * 0.4, 4, 0.2)
            mix += np.column_stack([noise(duration_sec, "brown", 0.08)] * 2)
        elif "Cello" in genre or "Piano" in genre:
            mix = pad(duration_sec, base * 0.5, 8, 0.2)
            mix += pad(duration_sec, base * 0.75, 5, 0.12)
        elif "Classical" in genre:
            mix = pad(duration_sec, base * 0.5, 7, 0.18)
            mix += pad(duration_sec, base, 4, 0.1)
        else:  # Warm Drone
            mix = pad(duration_sec, base * 0.5, 6, 0.22)
            mix += binaural(duration_sec, base, 1.5, 0.1)

        # Normalize peak
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 0.82

        # Convert to int16 WAV in memory
        mix_int = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        wavfile.write(wav_io, SAMPLE_RATE, mix_int)
        wav_bytes = wav_io.getvalue()

        # Optional pydub processing + MP3
        mp3_bytes = None
        if PYDUB_AVAILABLE:
            try:
                wav_io.seek(0)
                audio = AudioSegment.from_wav(wav_io)
                if fade_sec > 0:
                    audio = audio.fade_in(fade_sec * 1000).fade_out(fade_sec * 1000)
                if bass_boost_db != 0:
                    low = audio.low_pass_filter(100)
                    audio = audio.overlay(low + bass_boost_db)
                if high_shelf_db != 0:
                    high = audio.high_pass_filter(6000)
                    audio = audio.overlay(high + high_shelf_db)
                if do_normalize:
                    audio = pydub_normalize(audio, headroom=1.0)

                # Re-export processed WAV
                out_wav = io.BytesIO()
                audio.export(out_wav, format="wav")
                wav_bytes = out_wav.getvalue()

                # MP3
                out_mp3 = io.BytesIO()
                audio.export(out_mp3, format="mp3", bitrate="192k")
                mp3_bytes = out_mp3.getvalue()
            except Exception as e:
                st.warning(f"Advanced processing skipped: {e}")

    st.success("Music generated!")
    st.audio(wav_bytes, format="audio/wav")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download WAV",
            wav_bytes,
            file_name=f"sleep_{genre.replace(' ', '_')}_{duration_min}min.wav",
            mime="audio/wav"
        )
    with col_dl2:
        if mp3_bytes:
            st.download_button(
                "Download MP3",
                mp3_bytes,
                file_name=f"sleep_{genre.replace(' ', '_')}_{duration_min}min.mp3",
                mime="audio/mpeg"
            )
        else:
            st.info("MP3 export requires ffmpeg + working pydub")

    # Optional OpenAI title/description
    if openai_key and OPENAI_AVAILABLE:
        try:
            client = OpenAI(api_key=openai_key)
            prompt = (
                f"Write a calm, professional YouTube title and short description "
                f"for an original {duration_min}-minute {genre} track using {base} Hz "
                f"and {beat_freq} Hz binaural beats. Keep it suitable for a sleep channel."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=180
            )
            st.subheader("Suggested Title & Description")
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.warning(f"OpenAI error: {e}")

st.markdown("---")
st.caption("Tip: For full-length YouTube videos, generate multiple segments and loop/crossfade them in your editor. Headphones recommended for binaural effect.")
