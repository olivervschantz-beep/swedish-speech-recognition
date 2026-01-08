from pathlib import Path

import librosa
import soundfile as sf

WORDS = ["SJU", "KORSORD", "KRAFTSKIVA"]

RAW_ROOT = Path("data/raw")
WAV_ROOT = Path("data/test")

TARGET_SR = 16_000  # t.ex. 16 kHz, ändra om ni vill


def convert_all():
    for word in WORDS:
        raw_dir = RAW_ROOT / word
        wav_dir = WAV_ROOT / word

        if not raw_dir.exists():
            print(f"[VARNING] {raw_dir} finns inte, hoppar över.")
            continue

        wav_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Bearbetar ord: {word} ===")
        for audio_path in raw_dir.iterdir():
            if not audio_path.is_file():
                continue
            if audio_path.name.startswith("."):
                continue  # hoppa gömda filer

            try:
                # librosa läser massor av format (mp3, m4a, konstiga wav, ...)
                # sr=TARGET_SR => resamplar direkt
                # mono=True => gör 1 kanal
                y, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
            except Exception as e:
                print(f"[FEL] Kunde inte läsa {audio_path}: {e}")
                continue

            # y är en 1D numpy-array (samples), sr är TARGET_SR
            out_name = audio_path.stem + ".wav"
            out_path = wav_dir / out_name

            try:
                # soundfile skriver en "ren" wav som torchaudio sedan kan läsa
                sf.write(str(out_path), y, sr)
                print(f"[OK] {audio_path.name} -> {out_path}")
            except Exception as e:
                print(f"[FEL] Kunde inte spara {out_path}: {e}")


if __name__ == "__main__":
    convert_all()