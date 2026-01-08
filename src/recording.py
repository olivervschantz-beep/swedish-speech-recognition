import sounddevice as sd
import numpy as np
import queue
import argparse
import os
import time
import wave
import sys

# ---------------- CONFIG ---------------- #
SAMPLE_RATE = 16000          # Hz
BLOCK_DURATION = 0.05        # seconds per audio block (50 ms)
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

ENERGY_THRESHOLD = 0.001     # tweak this if it doesn't detect speech
MAX_SILENCE_BLOCKS = 10      # how many "silent" blocks after speech = end of word (10 * 50ms = 0.5s)
MIN_SPEECH_BLOCKS = 5        # minimum length of speech segment (5 * 50ms = 0.25s)

# ---------------------------------------- #

audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Called from sounddevice when there is new audio."""
    if status:
        print(status, file=sys.stderr)
    # indata is float32 [-1, 1], shape (frames, channels)
    audio_q.put(indata.copy())


def save_wav(path, audio, sample_rate):
    """Save a 1D numpy array as a mono 16-bit WAV file."""
    # Ensure mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Clip to [-1, 1] and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)              # mono
        wf.setsampwidth(2)              # 2 bytes = 16 bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def analyze_segment(audio, sample_rate, expected_label=None):
    """
    Här kopplar ni in er analys / modell senare.
    Just nu bara en placeholder som skriver ut längden.
    audio: 1D numpy-array med samples
    """
    duration = len(audio) / sample_rate
    print(f"  -> Segment length: {duration:.2f} s")

    # EXEMPEL: här kan senare göra feature-extraktion + modell:
    # features = extract_features(audio, sample_rate)
    # probs = model(features)
    # predicted_class = ...
    # print("  -> Model prediction:", predicted_class)

    # For now, we do nothing more.
    return


def realtime_record(label, output_dir):
    """
    Lyssnar på mikrofonen, detekterar tal, sparar varje tal-segment som en .wav-fil.
    label = vilken etikett (sju / kräftskiva / korsord / other) sparar filerna under.
    """
    # Prepare folder
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Choose default input device (you can set sd.default.device if needed)
    print("Available audio devices:")
    print(sd.query_devices())
    print("\nUsing default input device. Change with sd.default.device if needed.\n")

    # Start input stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=BLOCK_SIZE
    )
    stream.start()

    print("Listening in real time...")
    print("Speak the word, segments will be saved automatically.")
    print("Press Ctrl+C to stop.\n")

    segment_blocks = []
    recording = False
    silence_blocks = 0
    file_counter = 0

    try:
        while True:
            # Get next block from callback
            block = audio_q.get()
            block = block[:, 0]  # mono

            # Compute simple energy for VAD
            energy = np.mean(block ** 2)

            if energy > ENERGY_THRESHOLD:
                # Speech detected
                if not recording:
                    # Start new segment
                    recording = True
                    segment_blocks = []
                    silence_blocks = 0
                    print("Speech started...")
                segment_blocks.append(block)
                silence_blocks = 0
            else:
                # No speech (below threshold)
                if recording:
                    silence_blocks += 1
                    segment_blocks.append(block)

                    # If we've had enough silence, end segment
                    if silence_blocks > MAX_SILENCE_BLOCKS:
                        recording = False

                        if len(segment_blocks) >= MIN_SPEECH_BLOCKS:
                            # Concatenate all blocks into one array
                            audio_segment = np.concatenate(segment_blocks)

                            # Save to file
                            timestamp = int(time.time())
                            file_counter += 1
                            filename = f"{label}_{timestamp}_{file_counter:03d}.wav"
                            filepath = os.path.join(label_dir, filename)

                            save_wav(filepath, audio_segment, SAMPLE_RATE)
                            print(f"Saved segment as: {filepath}")

                            # Call analysis hook
                            analyze_segment(audio_segment, SAMPLE_RATE, expected_label=label)

                        else:
                            print("Segment too short, discarded.")

                        # Reset
                        segment_blocks = []
                        silence_blocks = 0

    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        stream.stop()
        stream.close()
        print("Stream closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time speech capture: detect speech segments, save to WAV, and analyze."
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label/word for this recording session (e.g. sju, kraftskiva, korsord, other)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data",
        help="Output root directory (default: ./data)"
    )
    args = parser.parse_args()

    realtime_record(args.label, args.out)


if __name__ == "__main__":
    main()