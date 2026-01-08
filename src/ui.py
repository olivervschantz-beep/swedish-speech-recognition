# src/ui_mic.py

import tempfile
from io import BytesIO

import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st

from inference.predict import run_prediction


SAMPLE_RATE = 16000
RECORD_SECONDS = 2.0  # sekunder
WORDS = ["sju", "korsord", "kraftskiva"]


st.set_page_config(page_title="Uttalsbedömning", page_icon="")

st.title("Pronounciation scorer – real-time recording")

st.write(
    "1. Choose which word you want to pronounce\n"
    "2. Press **Record** and say the word\n"
    "3. The model will predict what word it thinks you said, and evaluate your pronunciation"
)

# Välj ord användaren ska säga (mest för instruktion + jämförelse)
target_word = st.selectbox("Choose the word you want to pronounce:", WORDS)

# Visa lite info
st.info(f"When you press **Record** ~{RECORD_SECONDS} seconds will be recorded. Say: **{target_word.upper()}**")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
    st.session_state.last_audio = None


def record_audio() -> np.ndarray:
    """Spela in från mikrofonen och returnera numpy-array [T, 1]."""
    st.write("Recording!")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    st.write("Recording finished.")
    return audio


if st.button("Record"):
    try:
        audio = record_audio()
    except Exception as e:
        st.error(f"Error while recording: {e}")
    else:
        # Spara för uppspelning i UI
        st.session_state.last_audio = audio.copy()

        # Skriv till en temporär wav-fil som predict.run_prediction kan läsa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            tmp_path = tmp.name

        try:
            result = run_prediction(tmp_path)
        except Exception as e:
            st.error(f"Error with prediction: {e}")
        else:
            # Spara resultat + target_word 
            st.session_state.last_result = (result, target_word)


# Visa senast inspelade/analysresultat
if st.session_state.last_result is not None:
    result, chosen_word = st.session_state.last_result

    st.subheader("Results from last recording:")

    # Spela upp senaste inspelningen
    if st.session_state.last_audio is not None:
        buf = BytesIO()
        sf.write(buf, st.session_state.last_audio, SAMPLE_RATE, format="wav")
        buf.seek(0)
        st.audio(buf, format="audio/wav")

    st.write(f"**You picked:** `{chosen_word}`")
    st.write(f"**The model predicted the word:** `{result['predicted_word']}`")

    # Matchar modellen användarens valda ord?
    if result["predicted_word"].lower() == chosen_word.lower():
        st.success("Correct word according to the model.")
        word_match = True
    else:
        st.warning("The model thinks you said another word.")
        word_match = False

    # Uttalsbedömning från autoencoder + tröskel
    if result["is_correct"] is True:
        st.write(f"**Pronounciation scorer (autoencoder):** {result['pronunciation_label']}")
    elif result["is_correct"] is False:
        st.write(f"**Pronounciation scorer (autoencoder):** {result['pronunciation_label']}")
    else:
        st.write(f"**Pronounciation scorer:** {result['pronunciation_label']}")

    # En "samlad" bedömning: rätt ord + korrekt uttal
    if result["is_correct"] is True and word_match:
        st.success("Correct word and pronunciation!")
    elif result["is_correct"] is False and word_match:
        st.error("Correct word but incorrect pronunciation.")
    elif not word_match and result["is_correct"] is True:
        st.warning("Wrong word but pronunciation seems correct.")
    else:
        st.error("Wrong word and incorrect pronunciation.")

    st.write("---")
    st.write(f"**Error (MSE):** `{result['error']:.6f}`")
    if result["threshold"] is not None:
        st.write(f"**Threshold for correct pronounciation:** `{result['threshold']:.6f}`")
        st.caption("(The lower the error compared to the threshold, the better your pronunciation.)")
