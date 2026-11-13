import streamlit as st
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment, effects
from pydub.effects import normalize as pydub_normalize
import io
import os
from openai import OpenAI
from serpapi import GoogleSearch
import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE = 44100
MAX_DURATION_MIN = 60  # safety limit

st.set_page_config(page_title="Original Sleep Music Generator", layout="wide")
st.title("Copyright-Free Sleep / Ambient Music Generator")
st.caption("Local synthesis (432 Hz, 528 Hz, binaural, etc.) + optional OpenAI/SerpAPI helpers. Perfect for YouTube deep-sleep tracks.")

# ---------- Sidebar: API keys (secure) ----------
with st.sidebar:
    st.header("API Keys (optional)")
    st.caption("Used only for titles/descriptions or research. Leave blank for pure local generation.")
    openai_key = st.text_input("OpenAI API Key", type="password", value=st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else "")
    serp_key = st.text_input("SerpAPI Key", type="password", value=st.secrets.get("SERPAPI_API_KEY", "") if hasattr(st, "secrets") else "")
    st.divider()
    st.info("All music is generated locally from math. No samples, no copyrighted material.")

# ---------- Controls ----------
col1, col2 = st.columns(2)
with col1:
    duration_min = st.slider("Duration (minutes)", 1, MAX_DURATION_MIN, 10)
    base_freq = st.selectbox("Base / Carrier Frequency", [432, 528, 440, 200, "Custom"])
    if base_freq == "Custom":
        base_freq = st.number_input("Custom base Hz", 50.0, 1000.0, 432.0)
    beat_freq = st.slider("Binaural Beat Frequency (Hz) – Delta 1-4 for deep sleep", 0.5, 10.0, 2.0, 0.1)

with col2:
    genre = st.selectbox(
        "Genre / Style",
        [
            "Deep Sleep 432Hz Binaural Delta",
            "528Hz Healing Meditation",
            "Lo-fi Hip Hop Ambient",
            "Cello & Piano Soft",
            "Classical Ambient",
            "Pink Noise + Binaural",
            "Custom (describe below)"
        ]
    )
    custom_genre = st.text_area("Custom genre description (optional, uses OpenAI if key provided)", height=80)

# Advanced audio processing
st.subheader("Audio Processing")
c1, c2, c3 = st.columns(3)
with c1:
    do_normalize = st.checkbox("Normalize", value=True)
    fade_sec = st.slider("Fade in/out (seconds)", 0, 30, 5)
with c2:
    bass_boost_db = st.slider("Bass boost / cut (dB)", -12, 12, 0)
with c3:
    high_shelf_db = st.slider("High shelf / air (dB)", -6, 6, 0)

generate_btn = st.button("Generate Music", type="primary")

# ---------- Helper synthesis functions ----------
def generate_noise(duration_sec, noise_type="pink", amplitude=0.3):
    n = int(duration_sec * SAMPLE_RATE)
    white = np.random.randn(n).astype(np.float32)
    if noise_type == "white":
        noise = white
    elif noise_type == "pink":
        # simple 1/f approximation via cumulative
        noise = np.cumsum(white)
    elif noise_type == "brown":
        noise = np.cumsum(np.cumsum(white))
    else:
        noise = white
    noise = noise / (np.max(np.abs(noise)) + 1e-8) * amplitude
    return noise

def generate_binaural(duration_sec, carrier, beat, amplitude=0.25):
    n = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    left = np.sin(2 * np.pi * carrier * t) * amplitude
    right = np.sin(2 * np.pi * (carrier + beat) * t) * amplitude
    return np.column_stack([left, right])

def generate_pad(duration_sec, freq, harmonics=5, amplitude=0.2):
    n = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    wave = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        wave += np.sin(2 * np.pi * freq * h * t) / h
    # slow amplitude modulation
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
    wave = wave * lfo * amplitude / harmonics
    return np.column_stack([wave, wave])  # stereo

def simple_lofi_beat(duration_sec, bpm=60):
    n = int(duration_sec * SAMPLE_RATE)
    audio = np.zeros(n, dtype=np.float32)
    beat_interval = int(60 / bpm * SAMPLE_RATE)
    for i in range(0, n, beat_interval):
        # soft kick (low sine burst)
        kick_len = min(8000, n - i)
        t = np.linspace(0, kick_len / SAMPLE_RATE, kick_len)
        kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 8) * 0.4
        audio[i:i+kick_len] += kick
    # light vinyl crackle
    crackle = generate_noise(duration_sec, "white", 0.02)
    audio += crackle
    return np.column_stack([audio, audio])

# ---------- Main generation ----------
if generate_btn:
    duration_sec = duration_min * 60
    if duration_sec > MAX_DURATION_MIN * 60:
        st.error("Duration too long for safety.")
        st.stop()

    with st.spinner("Synthesizing original music..."):
        # Base layers according to genre
        if "Deep Sleep" in genre or "Binaural" in genre or "Pink Noise" in genre:
            binaural = generate_binaural(duration_sec, float(base_freq), beat_freq, 0.18)
            noise = generate_noise(duration_sec, "pink", 0.25)
            noise_stereo = np.column_stack([noise, noise])
            pad = generate_pad(duration_sec, float(base_freq) / 2, 4, 0.15)
            mix = binaural + noise_stereo * 0.6 + pad
        elif "528Hz" in genre or "Healing" in genre or "Meditation" in genre:
            binaural = generate_binaural(duration_sec, 528 if base_freq == 528 else float(base_freq), beat_freq * 0.5, 0.15)
            pad = generate_pad(duration_sec, 264, 6, 0.22)
            noise = generate_noise(duration_sec, "pink", 0.12)
            noise_stereo = np.column_stack([noise, noise])
            mix = binaural * 0.7 + pad + noise_stereo
        elif "Lo-fi" in genre:
            beat = simple_lofi_beat(duration_sec, 55)
            pad = generate_pad(duration_sec, float(base_freq) / 4, 3, 0.18)
            mix = beat * 0.5 + pad
        elif "Cello" in genre or "Piano" in genre:
            # layered soft tones with slow decay feel
            pad1 = generate_pad(duration_sec, float(base_freq) / 2, 8, 0.2)
            pad2 = generate_pad(duration_sec, float(base_freq) * 0.75, 5, 0.12)
            mix = pad1 + pad2
        elif "Classical" in genre:
            pad = generate_pad(duration_sec, float(base_freq) / 2, 7, 0.2)
            higher = generate_pad(duration_sec, float(base_freq), 4, 0.1)
            mix = pad + higher * 0.6
        else:
            # fallback / custom
            mix = generate_binaural(duration_sec, float(base_freq), beat_freq) + generate_pad(duration_sec, float(base_freq)/2)

        # Normalize mix amplitude
        max_amp = np.max(np.abs(mix))
        if max_amp > 0:
            mix = mix / max_amp * 0.85

        # Convert to AudioSegment via temporary WAV in memory
        # Interleave for pydub (int16)
        mix_int = (mix * 32767).astype(np.int16)
        # Write to BytesIO as WAV
        wav_io = io.BytesIO()
        wavfile.write(wav_io, SAMPLE_RATE, mix_int)
        wav_io.seek(0)
        audio = AudioSegment.from_wav(wav_io)

        # Processing
        if fade_sec > 0:
            audio = audio.fade_in(fade_sec * 1000).fade_out(fade_sec * 1000)
        if bass_boost_db != 0:
            # simple low-shelf approximation via low-pass overlay
            low = audio.low_pass_filter(120)
            audio = audio.overlay(low + bass_boost_db)
        if high_shelf_db != 0:
            high = audio.high_pass_filter(8000)
            audio = audio.overlay(high + high_shelf_db)
        if do_normalize:
            audio = pydub_normalize(audio, headroom=0.5)

        # Export options
        wav_out = io.BytesIO()
        audio.export(wav_out, format="wav")
        wav_bytes = wav_out.getvalue()

        mp3_out = io.BytesIO()
        audio.export(mp3_out, format="mp3", bitrate="192k")
        mp3_bytes = mp3_out.getvalue()

    st.success("Music generated!")
    st.audio(wav_bytes, format="audio/wav")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download WAV", wav_bytes, file_name=f"sleep_{genre.replace(' ', '_')}_{duration_min}min.wav", mime="audio/wav")
    with c2:
        st.download_button("Download MP3", mp3_bytes, file_name=f"sleep_{genre.replace(' ', '_')}_{duration_min}min.mp3", mime="audio/mpeg")

    # Optional OpenAI title / description
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            prompt = f"Create a YouTube-friendly title and short description for an original, copyright-free {genre} track of {duration_min} minutes using {base_freq}Hz carrier and {beat_freq}Hz binaural beat for deep sleep / relaxation. Keep it calm and professional."
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            st.subheader("Suggested Title & Description (OpenAI)")
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.warning(f"OpenAI call failed: {e}")

    # Optional SerpAPI research
    if serp_key and st.button("Research frequencies with SerpAPI"):
        try:
            search = GoogleSearch({
                "q": f"best frequencies for {genre} binaural beats deep sleep",
                "api_key": serp_key
            })
            results = search.get_dict()
            st.subheader("Related search snippets")
            for res in results.get("organic_results", [])[:5]:
                st.write(f"**{res.get('title')}**")
                st.caption(res.get("snippet", ""))
                st.markdown(f"[Link]({res.get('link')})")
        except Exception as e:
            st.warning(f"SerpAPI failed: {e}")

st.markdown("---")
st.caption("Tip: For very long YouTube tracks, generate 10–30 min segments and loop/crossfade in your editor. Headphones recommended for binaural effect. All audio is original synthetic content.")
