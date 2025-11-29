#!/usr/bin/env python3
"""
predict_fusion.py

Usage examples:
  # predict single sample with text + audio
  python scripts/predict_fusion.py --text "Your account has been compromised" --audio data/audio/processed/phishing_123.npy

  # batch predict from CSV (expects 'text' and 'audio' columns), outputs csv with prob + label
  python scripts/predict_fusion.py --predict-csv data/fusion/fusion.csv --out-csv outputs/fusion_predictions.csv

  # tune threshold from validation CSV and save threshold.json in model dir
  python scripts/predict_fusion.py --tune --val-csv data/fusion/fusion.csv --min-precision 0.6
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
hf_logging.set_verbosity_error()

# Set TensorFlow logging level before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--text-model-dir", type=str, default="models/text_distilroberta/best", 
                    help="Directory where text model + tokenizer live")
parser.add_argument("--audio-model-path", type=str, default="models/audio_model/audio_cnn_best.h5",
                    help="Path to TensorFlow audio model")
parser.add_argument("--fusion-model-path", type=str, default="models/fusion_model.pt",
                    help="Path to PyTorch fusion model")
parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if not provided)")
parser.add_argument("--cpu", action="store_true", help="Force CPU (also sets CUDA_VISIBLE_DEVICES=-1)")
parser.add_argument("--tune", action="store_true", help="Tune threshold using validation CSV")
parser.add_argument("--val-csv", type=str, default="data/fusion/fusion.csv", 
                    help="CSV used for threshold tuning (must have 'text', 'audio', and 'label')")
parser.add_argument("--min-precision", type=float, default=0.6, 
                    help="Minimum precision requirement during threshold tuning")
parser.add_argument("--text", type=str, default=None, help="Single text to predict")
parser.add_argument("--audio", type=str, default=None, help="Single audio file path (.npy mel spectrogram)")
parser.add_argument("--predict-csv", type=str, default=None, 
                    help="CSV with columns 'text' and 'audio' to predict in batch")
parser.add_argument("--out-csv", type=str, default="outputs/fusion_predictions.csv", 
                    help="Output csv for batch predictions")
parser.add_argument("--threshold-path", type=str, default=None, 
                    help="Explicit threshold.json path (overrides fusion-model-dir)")
parser.add_argument("--step", type=float, default=0.001, help="Threshold grid step for tuning")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for predictions")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# Handle CPU forcing
if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args.device = "cpu"

# ---------- device ----------
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TH_JSON = Path(args.threshold_path) if args.threshold_path else Path(args.fusion_model_path).parent / "threshold.json"

# Constants for audio processing
N_MELS = 64
MEL_WIDTH = 256

if args.verbose:
    print(f"Using device: {device}")
    print(f"Text model dir: {args.text_model_dir}")
    print(f"Audio model path: {args.audio_model_path}")
    print(f"Fusion model path: {args.fusion_model_path}")
    print(f"Threshold file: {TH_JSON}")

# ---------- Fusion Model Definition (must match train_fusion_prob.py) ----------
class FusionMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------- helpers ----------
def load_text_model(model_dir: str):
    """Load text model and tokenizer."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Text model dir not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model, tokenizer

def load_audio_model(model_path: str):
    """Load TensorFlow audio model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Audio model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.trainable = False
    return model

def load_fusion_model(model_path: str):
    """Load PyTorch fusion model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Fusion model not found: {model_path}")
    model = FusionMLP()
    # Try weights_only=True first (safe for state dicts), fallback if needed
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for older model formats that might have non-tensor metadata
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_text_probs(text_model, tokenizer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Return phishing probabilities for text inputs."""
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=256)
        for k in enc:
            enc[k] = enc[k].to(device)
        with torch.no_grad():
            out = text_model(**enc)
            logits = out.logits
            if logits is None:
                raise RuntimeError("Text model returned no logits")
            # Convert to probs (positive class)
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)

def load_audio_mel(audio_path: str) -> np.ndarray:
    """Load and preprocess audio mel spectrogram."""
    if not os.path.exists(audio_path):
        if args.verbose:
            print(f"⚠ Missing audio file: {audio_path}, using zeros")
        return np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)
    
    try:
        mel = np.load(audio_path).astype(np.float32)
    except Exception as e:
        if args.verbose:
            print(f"⚠ Error loading {audio_path}: {e}, using zeros")
        return np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)
    
    # Fix shape if needed
    if mel.ndim != 2:
        if args.verbose:
            print(f"⚠ Bad mel shape {mel.shape} in {audio_path}, reshaping to zeros")
        return np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)
    
    h, w = mel.shape
    
    # Pad or truncate to MEL_WIDTH
    if w < MEL_WIDTH:
        pad_width = MEL_WIDTH - w
        mel = np.hstack([mel, np.zeros((N_MELS, pad_width), dtype=np.float32)])
    else:
        mel = mel[:, :MEL_WIDTH]
    
    return mel

def predict_audio_probs(audio_model, audio_paths: List[str], batch_size: int = 32) -> np.ndarray:
    """Return phishing probabilities for audio inputs."""
    probs = []
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        batch_mels = []
        
        for path in batch_paths:
            mel = load_audio_mel(path)
            batch_mels.append(mel)
        
        # Stack into batch: (batch_size, N_MELS, MEL_WIDTH, 1)
        batch_array = np.stack(batch_mels, axis=0)
        batch_array = np.expand_dims(batch_array, axis=-1)  # Add channel dimension
        
        # Predict with TensorFlow model
        batch_tensor = tf.convert_to_tensor(batch_array)
        preds = audio_model.predict(batch_tensor, verbose=0)
        probs.extend(preds.flatten().tolist())
    
    return np.array(probs)

def predict_fusion_probs(fusion_model, text_probs: np.ndarray, audio_probs: np.ndarray, 
                         batch_size: int = 512) -> np.ndarray:
    """Return final phishing probabilities from fusion model."""
    # Combine text and audio probs
    features = np.stack([text_probs, audio_probs], axis=1).astype(np.float32)
    probs = []
    
    fusion_model.eval()
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).to(device)
            pred = fusion_model(batch_tensor)
            probs.extend(pred.cpu().numpy().tolist())
    
    return np.array(probs)

def save_threshold(path: Path, threshold: float, metadata: Dict = None):
    """Save threshold to JSON file."""
    obj = {"threshold": float(threshold)}
    if metadata:
        obj["meta"] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    if args.verbose:
        print(f"Saved threshold {threshold:.4f} -> {path}")

def load_threshold(path: Path) -> float:
    """Load threshold from JSON file."""
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        j = json.load(f)
    return float(j.get("threshold", 0.5))

def best_threshold_from_probs(probs: np.ndarray, golds: np.ndarray, min_precision: float = 0.6, 
                               step: float = 0.001) -> Tuple[float, Dict]:
    """
    Search thresholds from 0..1 with step to find:
      - the threshold with the highest recall while precision >= min_precision
      - if none satisfies min_precision, pick threshold maximizing f1
    Returns (best_threshold, metrics_dict)
    """
    best = None
    best_rec = -1.0
    best_f1 = -1.0
    best_prec_for_rec = 0.0

    thresholds = np.arange(0.0, 1.0 + 1e-9, step)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average="binary", zero_division=0)
        # Priority: maximize recall subject to precision >= min_precision
        if prec >= min_precision:
            if rec > best_rec or (rec == best_rec and f1 > best_f1):
                best = t
                best_rec = rec
                best_f1 = f1
                best_prec_for_rec = prec
        # Keep track of best f1 in case no threshold meets min_precision
        if f1 > best_f1 and best is None:
            best_f1 = f1
            alt_t = t

    if best is None:
        # Fallback: choose threshold maximizing f1
        best = alt_t
        preds = (probs >= best).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average="binary", zero_division=0)
        return best, {"precision": float(prec), "recall": float(rec), "f1": float(f1), "used_min_precision": False}
    else:
        return best, {"precision": float(best_prec_for_rec), "recall": float(best_rec), "f1": float(best_f1), 
                     "used_min_precision": True}

# ---------- main ----------
if __name__ == "__main__":
    print("\n📥 Loading models...")
    
    # Load all models
    text_model, tokenizer = load_text_model(args.text_model_dir)
    audio_model = load_audio_model(args.audio_model_path)
    fusion_model = load_fusion_model(args.fusion_model_path)
    
    print("✅ All models loaded successfully")

    # If tuning requested: compute probs on val set and choose threshold
    if args.tune:
        if not Path(args.val_csv).exists():
            raise FileNotFoundError(f"Validation CSV not found: {args.val_csv} (needs columns 'text', 'audio', and 'label')")
        print(f"\n🔍 Tuning threshold using: {args.val_csv}")
        val_df = pd.read_csv(args.val_csv)
        
        # Check required columns
        required_cols = ["text", "audio", "label"]
        for col in required_cols:
            if col not in val_df.columns:
                raise ValueError(f"Validation CSV must contain '{col}' column")
        
        texts = val_df["text"].fillna("").astype(str).tolist()
        audio_paths = val_df["audio"].fillna("").astype(str).tolist()
        
        # Convert labels to binary (benign=0, phishing=1)
        golds = val_df["label"].map(lambda x: 1 if str(x).strip().lower() == "phishing" else 0).to_numpy(dtype=np.int32)
        print(f"  Samples: {len(texts)}  Positives: {int(golds.sum())}")

        # Get predictions from all models
        print("  Getting text probabilities...")
        text_probs = predict_text_probs(text_model, tokenizer, texts, batch_size=args.batch_size)
        
        print("  Getting audio probabilities...")
        audio_probs = predict_audio_probs(audio_model, audio_paths, batch_size=args.batch_size)
        
        print("  Getting fusion probabilities...")
        fusion_probs = predict_fusion_probs(fusion_model, text_probs, audio_probs, batch_size=512)
        
        best_t, metrics = best_threshold_from_probs(fusion_probs, golds, min_precision=args.min_precision, step=args.step)
        print(f"\n🏁 Tuning result: threshold={best_t:.4f}, {metrics}")
        save_threshold(TH_JSON, float(best_t), metadata={"min_precision": args.min_precision, "step": args.step})
    else:
        # Try to load threshold if available, otherwise set default 0.5
        if TH_JSON.exists():
            try:
                loaded_t = load_threshold(TH_JSON)
                if args.verbose:
                    print(f"Loaded threshold from {TH_JSON}: {loaded_t:.4f}")
            except Exception as e:
                print(f"Could not load threshold.json, using default 0.5: {e}")
                loaded_t = 0.5
        else:
            if args.verbose:
                print("No threshold.json found; using default threshold=0.5")
            loaded_t = 0.5
        best_t = float(loaded_t)

    # Single predict
    if args.text and args.audio:
        print(f"\n🔍 Predicting single sample...\n")
        print(f"Text: {args.text}")
        print(f"Audio: {args.audio}")
        
        # Get individual model predictions
        text_prob = predict_text_probs(text_model, tokenizer, [args.text], batch_size=1)[0]
        audio_prob = predict_audio_probs(audio_model, [args.audio], batch_size=1)[0]
        
        # Get fusion prediction
        fusion_prob = predict_fusion_probs(fusion_model, np.array([text_prob]), np.array([audio_prob]), batch_size=1)[0]
        
        label = "phishing" if fusion_prob >= best_t else "benign"
        
        print(f"\nText probability: {text_prob:.4f}")
        print(f"Audio probability: {audio_prob:.4f}")
        print(f"Fusion probability: {fusion_prob:.4f}")
        print(f"Threshold used: {best_t:.4f}")
        print(f"Predicted label: {label}\n")

    # Batch predict csv
    if args.predict_csv:
        inpath = Path(args.predict_csv)
        if not inpath.exists():
            raise FileNotFoundError(inpath)
        df = pd.read_csv(inpath)
        
        # Check required columns
        if "text" not in df.columns or "audio" not in df.columns:
            raise ValueError("Input CSV must contain 'text' and 'audio' columns")
        
        texts = df["text"].fillna("").astype(str).tolist()
        audio_paths = df["audio"].fillna("").astype(str).tolist()
        
        print(f"\n🚀 Batch predicting {len(texts)} samples...")
        
        # Get predictions from all models
        print("  Getting text probabilities...")
        text_probs = predict_text_probs(text_model, tokenizer, texts, batch_size=args.batch_size)
        
        print("  Getting audio probabilities...")
        audio_probs = predict_audio_probs(audio_model, audio_paths, batch_size=args.batch_size)
        
        print("  Getting fusion probabilities...")
        fusion_probs = predict_fusion_probs(fusion_model, text_probs, audio_probs, batch_size=512)
        
        # Add results to dataframe
        df["text_prob"] = text_probs
        df["audio_prob"] = audio_probs
        df["fusion_prob"] = fusion_probs
        df["predicted_label"] = df["fusion_prob"].apply(lambda p: "phishing" if p >= best_t else "benign")
        
        # Save output
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"\n✅ Saved predictions to: {outp} (threshold {best_t:.4f})")

