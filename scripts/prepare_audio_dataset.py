import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# Directories
BENIGN_DIR = "data/audio/benign_1000"
PHISH_DIR = "data/audio/phishing"
OUT_DIR = "data/audio/processed"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = "data/audio/audio_dataset.csv"

# Settings
TARGET_SR = 16000
N_MELS = 64
MAX_DURATION = 4.0   # seconds
TOP_DB = 20          # silence threshold


def trim_silence(audio):
    """ Remove leading/trailing silence """
    trimmed, _ = librosa.effects.trim(audio, top_db=TOP_DB)
    return trimmed


def reduce_noise(audio, sr):
    """
    Simple noise reduction using spectral gating.
    Better than nothing + very fast.
    """
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr)


def rms_normalize(audio):
    """ Normalize loudness """
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 0.03
    if rms > 0:
        audio = audio * (target_rms / (rms + 1e-9))
    return audio


def enforce_max_duration(audio, sr):
    """ Cut or pad to fixed duration """
    max_len = int(sr * MAX_DURATION)

    if len(audio) > max_len:
        audio = audio[:max_len]

    elif len(audio) < max_len:
        pad_width = max_len - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")

    return audio


def generate_mel(file_path, out_name):
    """ Full advanced cleaning pipeline → mel spectrogram """
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        # 1. Trim silence
        audio = trim_silence(audio)

        # 2. Reduce background noise
        audio = reduce_noise(audio, TARGET_SR)

        # 3. RMS normalize
        audio = rms_normalize(audio)

        # 4. Enforce duration limit
        audio = enforce_max_duration(audio, TARGET_SR)

        # 5. Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=TARGET_SR, n_mels=N_MELS, fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        out_path = os.path.join(OUT_DIR, f"{out_name}.npy")
        np.save(out_path, mel_db)
        return out_path

    except Exception as e:
        print(f" Error processing {file_path}: {e}")
        return None


def process_folder(folder, label):
    rows = []
    print(f"\n Processing {label} files...")

    for i, fname in enumerate(sorted(os.listdir(folder))):
        if not fname.endswith(".wav"):
            continue

        fpath = os.path.join(folder, fname)
        out_name = f"{label}_{i}"
        print(f" → {fpath}")

        mel_path = generate_mel(fpath, out_name)
        if mel_path:
            rows.append([out_name, mel_path, label])

    return rows


def main():
    print(" Starting ADVANCED audio preprocessing...")

    benign_rows = process_folder(BENIGN_DIR, "benign")
    phishing_rows = process_folder(PHISH_DIR, "phishing")

    all_rows = benign_rows + phishing_rows

    df = pd.DataFrame(all_rows, columns=["id", "path", "label"])
    df.to_csv(CSV_PATH, index=False)

    print("\n Advanced preprocessing complete!")
    print("Processed files →", OUT_DIR)
    print("Dataset CSV →", CSV_PATH)


if __name__ == "__main__":
    main()
