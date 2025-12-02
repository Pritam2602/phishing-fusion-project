"""
Advanced preprocessing v3 (Auto-Read Augmented Folders)
-------------------------------------------------------
Reads WAVs from:
  data/audio/benign_1000
  data/audio/phishing
  data/audio/augmented/benign
  data/audio/augmented/phishing

Does:
 - aggressive silence trimming
 - energy-based VAD cleanup
 - noise reduction (if noisereduce installed)
 - RMS normalization
 - fixed duration padding
 - mel spectrogram conversion
Saves results to:
  data/audio/processed/*.npy
"""

import os
import librosa
import numpy as np
import pandas as pd


RAW_BENIGN = "data/audio/benign_1000"
RAW_PHISH  = "data/audio/phishing"

AUG_BENIGN = "data/audio/augmented/benign"
AUG_PHISH  = "data/audio/augmented/phishing"

# Output folders
OUT_DIR = "data/audio/processed"
CSV_PATH = "data/audio/audio_dataset.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# Settings
SR = 16000
N_MELS = 64
MAX_DURATION = 4.0
TOP_DB = 30
MIN_VOICE_LEN = 0.5



def trim_silence(y, top_db=TOP_DB):
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trim

def remove_quiet_frames(y, sr, frame_length=2048, hop_length=512, energy_thresh=0.01):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    mask_frames = rms > (energy_thresh * np.max(rms) + 1e-9)
    if mask_frames.sum() == 0:
        return y
    samples_per_frame = hop_length
    mask = np.repeat(mask_frames, samples_per_frame)
    mask = np.pad(mask, (0, max(0, len(y)-len(mask))), mode="constant")[:len(y)]
    return y[mask.astype(bool)]

def rms_normalize(y, target_rms=0.03):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    if rms <= 0:
        return y
    return y * (target_rms / rms)

def enforce_duration(y, sr=SR, max_dur=MAX_DURATION):
    max_len = int(sr * max_dur)
    if len(y) > max_len:
        return y[:max_len]
    return np.pad(y, (0, max_len - len(y)), mode="constant")

def maybe_reduce_noise(y, sr=SR):
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=sr)
    except:
        return y

def make_mel(y, sr=SR, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def process_file(path, label, uid):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)

        y = trim_silence(y)
        if len(y) < MIN_VOICE_LEN * SR:
            return None

        y = remove_quiet_frames(y, SR)
        y = maybe_reduce_noise(y)
        y = rms_normalize(y)
        y = enforce_duration(y)

        mel = make_mel(y)

        out_name = f"{label}_{uid}.npy"
        out_path = os.path.join(OUT_DIR, out_name)

        np.save(out_path, mel.astype(np.float32))
        return out_path

    except Exception as e:
        print(f" Error processing {path}: {e}")
        return None



def collect_paths():
    folders = [
        (RAW_BENIGN, "benign"),
        (RAW_PHISH,  "phishing"),
        (AUG_BENIGN, "benign"),
        (AUG_PHISH,  "phishing"),
    ]

    all_items = []

    for folder, label in folders:
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                all_items.append((os.path.join(folder, fname), label))

    return all_items



def main():
    print("\n Collecting WAV files...")
    items = collect_paths()
    print(f" Total audio files found: {len(items)}\n")

    rows = []
    seen = set()
    uid = 0

    for path, label in items:
        # avoid double processing files with same absolute content
        key = os.path.abspath(path)
        if key in seen:
            continue
        seen.add(key)

        out = process_file(path, label, uid)
        if out:
            rows.append([f"{label}_{uid}", out, label])
            uid += 1

    df = pd.DataFrame(rows, columns=["id", "path", "label"])
    df.to_csv(CSV_PATH, index=False)

    print("\n Preprocessing complete!")
    print(f" Saved dataset CSV: {CSV_PATH}")
    print(f" Total processed samples: {len(df)}")


if __name__ == "__main__":
    main()
