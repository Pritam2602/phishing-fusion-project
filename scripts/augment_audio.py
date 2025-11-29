"""
Augmentation script.
Creates augmented WAV files into data/audio/augmented/{benign,phishing}
Augmentations:
 - additive gaussian noise (SNR variety)
 - time stretch (0.9 - 1.1)
 - pitch shift (-2..+2 semitones)
 - random gain
"""

import os, random
import soundfile as sf
import librosa
import numpy as np
from glob import glob

SRC_DIRS = {
    "benign": "data/audio/benign_1000",
    "phishing": "data/audio/phishing"
}
OUT_BASE = "data/audio/augmented"
SR = 16000

os.makedirs(OUT_BASE, exist_ok=True)
for k in SRC_DIRS:
    os.makedirs(os.path.join(OUT_BASE, k), exist_ok=True)

def add_noise(y, snr_db):
    # compute signal power and add gaussian noise to achieve snr_db
    sig_power = np.mean(y**2)
    snr = 10**(snr_db/10.0)
    noise_power = sig_power / (snr + 1e-12)
    noise = np.random.normal(0, np.sqrt(noise_power), size=y.shape)
    return y + noise

def time_stretch(y, rate):
    try:
        return librosa.effects.time_stretch(y, rate)
    except Exception:
        return y

def pitch_shift(y, sr, n_steps):
    try:
        return librosa.effects.pitch_shift(y, sr, n_steps)
    except Exception:
        return y

def random_gain(y, min_db=-6, max_db=6):
    db = random.uniform(min_db, max_db)
    factor = 10**(db/20.0)
    return y * factor

def augment_file(path, out_dir, prefix, i):
    y, sr = librosa.load(path, sr=SR, mono=True)
    variants = []
    # original copy (normalized)
    variants.append(("orig", y.copy()))
    # noise variants
    for snr in [20, 10, 6]:
        variants.append((f"noise{snr}", add_noise(y, snr)))
    # time stretch
    for r in [0.95, 1.05]:
        ys = time_stretch(y, r)
        variants.append((f"ts{r}", ys))
    # pitch shift
    for ps in [-2, 2]:
        ys = pitch_shift(y, SR, ps)
        variants.append((f"ps{ps}", ys))
    # random gain
    variants.append(("gain", random_gain(y)))

    out_paths = []
    for name, yy in variants:
        out_name = f"{prefix}_{i}_{name}.wav"
        out_path = os.path.join(out_dir, out_name)
        # ensure length not too long/short
        if len(yy) < 0.3*SR:
            yy = np.pad(yy, (0, int(0.3*SR)-len(yy)), mode="constant")
        sf.write(out_path, yy, SR)
        out_paths.append(out_path)
    return out_paths

def main(samples_per_file=3):
    print("Starting augmentation...")
    for label, src in SRC_DIRS.items():
        files = glob(os.path.join(src, "*.wav"))
        out_dir = os.path.join(OUT_BASE, label)
        idx = 0
        for p in files:
            augment_file(p, out_dir, label, idx)
            idx += 1
    print("Augmentation finished. Augmented files in", OUT_BASE)

if __name__ == "__main__":
    main()
 