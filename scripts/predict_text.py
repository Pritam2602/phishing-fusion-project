

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
hf_logging.set_verbosity_error()


parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, required=True, help="Directory where model + tokenizer live (from save_pretrained)")
parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if not provided)")
parser.add_argument("--tune", action="store_true", help="Tune threshold using validation CSV")
parser.add_argument("--val-csv", type=str, default="data/text/val.csv", help="CSV used for threshold tuning (must have 'text' and 'label')")
parser.add_argument("--min-precision", type=float, default=0.6, help="Minimum precision requirement during threshold tuning")
parser.add_argument("--predict", type=str, default=None, help="Single text to predict")
parser.add_argument("--predict-csv", type=str, default=None, help="CSV with column 'text' to predict in batch")
parser.add_argument("--out-csv", type=str, default="outputs/text_predictions.csv", help="Output csv for batch predictions")
parser.add_argument("--threshold-path", type=str, default=None, help="Explicit threshold.json path (overrides model-dir)")
parser.add_argument("--step", type=float, default=0.001, help="Threshold grid step for tuning (default 0.001)")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

MODEL_DIR = Path(args.model_dir)
TH_JSON = Path(args.threshold_path) if args.threshold_path else MODEL_DIR / "threshold.json"

if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.verbose:
    print(f"Using device: {device}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Threshold file: {TH_JSON}")


def load_model_and_tokenizer(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model, tokenizer

def predict_batch_texts(model, tokenizer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Return phishing probabilities for the positive class (index 1)."""
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
        for k in enc:
            enc[k] = enc[k].to(device)
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits  # shape (B, num_labels)
            if logits is None:
                raise RuntimeError("Model returned no logits")
            # convert to probs
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # positive class prob
            probs.extend(p.tolist())
    return np.array(probs)

def save_threshold(path: Path, threshold: float, metadata: Dict = None):
    obj = {"threshold": float(threshold)}
    if metadata:
        obj["meta"] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    if args.verbose:
        print(f"Saved threshold {threshold:.4f} -> {path}")

def load_threshold(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        j = json.load(f)
    return float(j.get("threshold", 0.5))

def best_threshold_from_probs(probs: np.ndarray, golds: np.ndarray, min_precision: float = 0.6, step: float = 0.001) -> Tuple[float, Dict]:
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
        if prec >= min_precision:
            if rec > best_rec or (rec == best_rec and f1 > best_f1):
                best = t
                best_rec = rec
                best_f1 = f1
                best_prec_for_rec = prec
        if f1 > best_f1 and best is None:
            best_f1 = f1
            alt_t = t

    if best is None:
        
        best = alt_t
        preds = (probs >= best).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average="binary", zero_division=0)
        return best, {"precision": float(prec), "recall": float(rec), "f1": float(f1), "used_min_precision": False}
    else:
        return best, {"precision": float(best_prec_for_rec), "recall": float(best_rec), "f1": float(best_f1), "used_min_precision": True}

# ---------- main ----------
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)

    
    if args.tune:
        if not Path(args.val_csv).exists():
            raise FileNotFoundError(f"Validation CSV not found: {args.val_csv} (needs columns 'text' and 'label')")
        print("🔍 Tuning threshold using:", args.val_csv)
        val_df = pd.read_csv(args.val_csv)
        if "text" not in val_df.columns or "label" not in val_df.columns:
            raise ValueError("Validation CSV must contain 'text' and 'label' columns")
        texts = val_df["text"].fillna("").astype(str).tolist()
        golds = val_df["label"].map(lambda x: 1 if str(x).strip().lower() == "phishing" else 0).to_numpy(dtype=np.int32)
        print(f"  Samples: {len(texts)}  Positives: {int(golds.sum())}")

        probs = predict_batch_texts(model, tokenizer, texts, batch_size=128)
        best_t, metrics = best_threshold_from_probs(probs, golds, min_precision=args.min_precision, step=args.step)
        print("🏁 Tuning result:", {"threshold": best_t, **metrics})
        save_threshold(TH_JSON, float(best_t), metadata={"min_precision": args.min_precision, "step": args.step})
    else:
        
        if TH_JSON.exists():
            try:
                loaded_t = load_threshold(TH_JSON)
                print(f"Loaded threshold from {TH_JSON}: {loaded_t:.4f}")
            except Exception as e:
                print("Could not load threshold.json, using default 0.5:", e)
                loaded_t = 0.5
        else:
            print("No threshold.json found; using default threshold=0.5")
            loaded_t = 0.5
        best_t = float(loaded_t)

    
    if args.predict:
        text = args.predict
        probs = predict_batch_texts(model, tokenizer, [text], batch_size=1)
        prob = float(probs[0])
        label = "phishing" if prob >= best_t else "benign"
        print("\nPredicting single text...\n")
        print(f"Text: {text}")
        print(f"Phishing probability: {prob:.4f}")
        print(f"Threshold used: {best_t:.4f}")
        print(f"Predicted label: {label}\n")

    if args.predict_csv:
        inpath = Path(args.predict_csv)
        if not inpath.exists():
            raise FileNotFoundError(inpath)
        df = pd.read_csv(inpath)
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain 'text' column")
        texts = df["text"].fillna("").astype(str).tolist()
        probs = predict_batch_texts(model, tokenizer, texts, batch_size=128)
        df["phishing_prob"] = probs
        df["predicted_label"] = df["phishing_prob"].apply(lambda p: "phishing" if p >= best_t else "benign")
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"Saved predictions to: {outp} (threshold {best_t:.4f})")
