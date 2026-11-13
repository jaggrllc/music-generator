import streamlit as st
import numpy as np
from scipy.io import wavfile
import io
import warnings
warnings.filterwarnings("ignore")

try:
    from pydub import AudioSegment
    from pydub.effects import normalize as pydub_normalize
    PYDUB_AVAILABLE = True
except Exception as e:
    st.warning(f"pydub not fully available: {e}. WAV export still works.")
    PYDUB_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

SAMPLE_RATE = 44100
MAX_DURATION_MIN = 40

st.set_page_config(page_title="Original Sleep Music Generator", layout="wide")
st.title("Copyright-Free Sleep & Ambient Music Generator")
st.caption("Local synthesis • 432 Hz • 528 Hz • Binaural • Improved instrument textures")

with st.sidebar:
    st.header("Optional API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password")
    st.info("All audio is generated locally from pure math.")

col1, col2 = st.columns(2)
with col1:
    duration_min = st.slider("Duration (minutes)", 1, MAX_DURATION_MIN, 8)
    base_freq = st.selectbox("Base Frequency", [432, 528, 440, 220, "Custom"])
    if base_freq == "Custom":
        base_freq = st.number_input("Custom Hz", 40.0, 1000.0, 432.0)
    beat_freq = st.slider("Binaural Beat (Hz)", 0.5, 7.0, 2.0, 0.1)

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
c1, c2, c3 = st.columns(3)
with c1:
    do_normalize = st.checkbox("Normalize", value=True)
with c2:
    fade_sec = st.slider("Fade in/out (sec)", 0, 20, 5)
with c3:
    bass_boost_db = st.slider("Bass boost (dB)", -8, 10, 1)

if st.button("Generate Music", type="primary"):
    duration_sec = duration_min * 60
    base = float(base_freq)
    n_samples = int(duration_sec * SAMPLE_RATE)

    with st.spinner("Generating improved music..."):
        def noise(duration, kind="pink", amp=0.22):
            n = int(duration * SAMPLE_RATE)
            white = np.random.randn(n).astype(np.float32)
            if kind == "pink":
                sig = np.cumsum(white)
            elif kind == "brown":
                sig = np.cumsum(np.cumsum(white))
            else:
                sig = white
            return sig / (np.max(np.abs(sig)) + 1e-8) * amp

        def binaural(duration, carrier, beat, amp=0.18):
            n = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, n, endpoint=False)
            left = np.sin(2 * np.pi * carrier * t) * amp
            right = np.sin(2 * np.pi * (carrier + beat) * t) * amp
            return np.column_stack([left, right])

        def pad(duration, freq, harmonics=5, amp=0.15):
            n = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, n, endpoint=False)
            wave = np.zeros_like(t)
            for h in range(1, harmonics + 1):
                wave += np.sin(2 * np.pi * freq * h * t) / (h ** 1.25)
            lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.035 * t)
            wave *= lfo * amp / harmonics
            return np.column_stack([wave, wave])

        def soft_piano_note(duration, freq, amp=0.2):
            n = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, n, endpoint=False)
            wave = np.zeros_like(t)
            for h in range(1, 6):
                detune = 1 + h * 0.0007
                wave += np.sin(2 * np.pi * freq * h * detune * t) / (h ** 1.5)
            attack = np.minimum(t * 35, 1.0)
            decay = np.exp(-t * 1.6)
            return wave * attack * decay * amp

        def soft_cello_note(duration, freq, amp=0.18):
            n = int(duration * SAMPLE_RATE)
            t = np.linspace(0, duration, n, endpoint=False)
            wave = np.zeros_like(t)
            for h in range(1, 7):
                wave += np.sin(2 * np.pi * freq * h * t) / (h ** 1.05)
            attack = np.minimum(t * 2.2, 1.0)
            vibrato = 1 + 0.007 * np.sin(2 * np.pi * 5.0 * t)
            return wave * attack * amp * vibrato

        mix = np.zeros((n_samples, 2), dtype=np.float32)

        if "Deep Sleep" in genre or "Pink Noise" in genre:
            mix += binaural(duration_sec, base, beat_freq, 0.16)
            mix += np.column_stack([noise(duration_sec, "pink", 0.24)] * 2)
            mix += pad(duration_sec, base * 0.5, 4, 0.1)

        elif "528" in genre or "Healing" in genre or "Meditation" in genre:
            mix += binaural(duration_sec, 528 if base == 528 else base, beat_freq * 0.55, 0.14)
            mix += pad(duration_sec, base * 0.5, 6, 0.18)
            mix += np.column_stack([noise(duration_sec, "pink", 0.09)] * 2)

        elif "Lo-fi" in genre:
            # Soft kick pattern
            beat_interval = int(0.85 * SAMPLE_RATE)
            for i in range(0, n_samples, beat_interval):
                kick_len = min(6000, n_samples - i)
                t = np.linspace(0, kick_len / SAMPLE_RATE, kick_len)
                kick = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 7) * 0.35
                mix[i:i+kick_len, 0] += kick
                mix[i:i+kick_len, 1] += kick
            # Vinyl crackle
            mix += np.column_stack([noise(duration_sec, "white", 0.018)] * 2)
            # Slow chords
            mix += pad(duration_sec, base * 0.45, 4, 0.16)

        elif "Cello" in genre or "Piano" in genre:
            note_len = 9.0
            piano_notes = [base * 0.5, base * 0.625, base * 0.75, base * 1.0]
            cello_notes = [base * 0.25, base * 0.375, base * 0.5]
            step = int(note_len * SAMPLE_RATE)
            for idx, start in enumerate(range(0, n_samples, step)):
                end = min(start + step, n_samples)
                length = end - start
                if length <= 0:
                    continue
                p_note = soft_piano_note(length / SAMPLE_RATE, piano_notes[idx % len(piano_notes)], 0.16)
                c_note = soft_cello_note(length / SAMPLE_RATE, cello_notes[idx % len(cello_notes)], 0.19)
                mix[start:end, 0] += c_note[:length] * 0.85
                mix[start:end, 1] += p_note[:length] * 0.75
                mix[start:end, 1] += c_note[:length] * 0.25
            mix += pad(duration_sec, base * 0.5, 3, 0.05)

        elif "Classical" in genre:
            # Slow arpeggio-style movement
            notes = [base * 0.5, base * 0.625, base * 0.75, base * 0.875, base]
            note_len = 6.0
            step = int(note_len * SAMPLE_RATE)
            for idx, start in enumerate(range(0, n_samples, step)):
                end = min(start + step, n_samples)
                length = end - start
                if length <= 0:
                    continue
                note = soft_piano_note(length / SAMPLE_RATE, notes[idx % len(notes)], 0.15)
                mix[start:end, 0] += note[:length] * 0.6
                mix[start:end, 1] += note[:length] * 0.7
            mix += pad(duration_sec, base * 0.5, 5, 0.08)

        else:  # Warm Drone
            mix += pad(duration_sec, base * 0.5, 6, 0.2)
            mix += binaural(duration_sec, base, 1.2, 0.09)
            mix += np.column_stack([noise(duration_sec, "brown", 0.06)] * 2)

        # Peak normalize
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 0.8

        # Export WAV
        mix_int = (np.clip(mix, -1.0, 1.0) * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        wavfile.write(wav_io, SAMPLE_RATE, mix_int)
        wav_bytes = wav_io.getvalue()

        mp3_bytes = None
        if PYDUB_AVAILABLE:
            try:
                wav_io.seek(0)
                audio = AudioSegment.from_wav(wav_io)
                if fade_sec > 0:
                    audio = audio.fade_in(fade_sec * 1000).fade_out(fade_sec * 1000)
                if bass_boost_db != 0:
                    low = audio.low_pass_filter(90)
                    audio = audio.overlay(low + bass_boost_db)
                if do_normalize:
                    audio = pydub_normalize(audio, headroom=0.8)

                out_wav = io.BytesIO()
                audio.export(out_wav, format="wav")
                wav_bytes = out_wav.getvalue()

                out_mp3 = io.BytesIO()
                audio.export(out_mp3, format="mp3", bitrate="192k")
                mp3_bytes = out_mp3.getvalue()
            except Exception as e:
                st.warning(f"Processing limited: {e}")

    st.success("Music generated!")
    st.audio(wav_bytes, format="audio/wav")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download WAV", wav_bytes,
                           file_name=f"{genre.replace(' ', '_')}_{duration_min}min.wav",
                           mime="audio/wav")
    with c2:
        if mp3_bytes:
            st.download_button("Download MP3", mp3_bytes,
                               file_name=f"{genre.replace(' ', '_')}_{duration_min}min.mp3",
                               mime="audio/mpeg")
        else:
            st.info("MP3 needs ffmpeg")

    if openai_key and OPENAI_AVAILABLE:
        try:
            client = OpenAI(api_key=openai_key)
            prompt = f"Calm YouTube title + short description for a {duration_min} min original {genre} sleep track at {base} Hz with {beat_freq} Hz binaural."
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            st.subheader("Suggested Title & Description")
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.warning(str(e))

st.caption("Improved textures for all genres. Headphones recommended for binaural beats.")
