#!/usr/bin/env python
"""
generate_fusion_dataset.py
Generates fusion.csv with text_prob + audio_prob for each sample.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TEXT_MODEL_DIR = "models/text_distilroberta/best"
AUDIO_MODEL_PATH = "models/audio_model/audio_cnn_best.h5"   # <-- TF audio model
INPUT_CSV = "data/fusion/fusion.csv"
OUTPUT_CSV = "data/fusion/fusion.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



print("\n Loading text model...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR).to(device)
text_model.eval()


print(" Loading audio CNN model...")
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
audio_model.trainable = False

N_MELS = 64
MEL_WIDTH = 256


print(f"\n Reading CSV: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

required_cols = ["audio", "text", "label"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f" Missing column '{c}' in {INPUT_CSV}")

df["label"] = df["label"].astype(str).str.lower().str.strip()
label_map = {"benign": 0, "phishing": 1}
df["label"] = df["label"].map(label_map)



@torch.no_grad()
def get_text_prob(text):
    if pd.isna(text):
        text = ""

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    logits = text_model(**encoded).logits
    prob = torch.sigmoid(logits[:, 1] - logits[:, 0])
    return float(prob.item())


def get_audio_prob(path):
    if not os.path.exists(path):
        print(f"Missing audio file: {path}, using default prob=0.5")
        return 0.5

    mel = np.load(path).astype(np.float32)

    # Fix mel dimensions (should be (64, 256))
    if mel.ndim != 2:
        print(f"⚠ Bad mel shape {mel.shape} in {path}, using zeros")
        mel = np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)

    h, w = mel.shape

    # Pad if width < 256
    if w < MEL_WIDTH:
        pad_width = MEL_WIDTH - w
        mel = np.hstack([mel, np.zeros((N_MELS, pad_width), dtype=np.float32)])
    else:
        # Truncate if > 256
        mel = mel[:, :MEL_WIDTH]

    # Now shape is (64, 256)
    mel = tf.convert_to_tensor(mel)
    mel = tf.reshape(mel, (1, N_MELS, MEL_WIDTH, 1))

    pred = audio_model.predict(mel, verbose=0)
    return float(pred[0][0])





print("\n Generating text & audio probabilities...\n")
text_probs = []
audio_probs = []

for i, row in df.iterrows():

    if i % 500 == 0:
        print(f"  → Processed {i}/{len(df)} rows")

    text_probs.append(get_text_prob(row["text"]))
    audio_probs.append(get_audio_prob(row["audio"]))

df["text_prob"] = text_probs
df["audio_prob"] = audio_probs


os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nFusion dataset created successfully!")
print(f" Saved to: {OUTPUT_CSV}")
print("\nPreview:")
print(df.head())
