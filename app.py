import os
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging


# Silence transformers / TF logs a bit
hf_logging.set_verbosity_error()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

TEXT_MODEL_DIR = MODELS_DIR / "text_distilroberta" / "best"
AUDIO_MODEL_PATH = MODELS_DIR / "audio_model" / "audio_cnn_best.h5"
FUSION_MODEL_PATH = MODELS_DIR / "fusion_model.pt"
THRESHOLD_JSON = (FUSION_MODEL_PATH.parent / "threshold.json")

SR = 16000
N_MELS = 64
MEL_WIDTH = 256
MAX_DURATION = 4.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------- audio preprocessing helpers for recorded WAV ----------


def trim_silence(y, top_db: int = 30):
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trim


def maybe_reduce_noise(y, sr: int = SR):
    try:
        import noisereduce as nr  # type: ignore

        return nr.reduce_noise(y=y, sr=sr)
    except Exception:
        return y


def rms_normalize(y, target_rms: float = 0.03):
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    if rms <= 0:
        return y
    return y * (target_rms / rms)


def enforce_duration(y, sr: int = SR, max_dur: float = MAX_DURATION):
    max_len = int(sr * max_dur)
    if len(y) > max_len:
        return y[:max_len]
    return np.pad(y, (0, max_len - len(y)), mode="constant")


def make_mel(y, sr: int = SR, n_mels: int = N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def load_text_model():
    if not TEXT_MODEL_DIR.exists():
        raise FileNotFoundError(f"Text model dir not found: {TEXT_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(TEXT_MODEL_DIR), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(TEXT_MODEL_DIR))
    model.to(device)
    model.eval()
    return model, tokenizer


def load_audio_model():
    if not AUDIO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Audio model not found: {AUDIO_MODEL_PATH}")
    model = tf.keras.models.load_model(str(AUDIO_MODEL_PATH))
    model.trainable = False
    return model


def load_fusion_model():
    if not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Fusion model not found: {FUSION_MODEL_PATH}")
    model = FusionMLP()
    # Use safe loading where possible
    try:
        state_dict = torch.load(FUSION_MODEL_PATH, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(FUSION_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_threshold(default: float = 0.5) -> float:
    if not THRESHOLD_JSON.exists():
        return float(default)
    try:
        import json

        with open(THRESHOLD_JSON, "r") as f:
            data = json.load(f)
        return float(data.get("threshold", default))
    except Exception:
        return float(default)


def text_prob(text_model, tokenizer, text: str) -> float:
    enc = tokenizer([text], truncation=True, padding=True, return_tensors="pt", max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = text_model(**enc)
        logits = out.logits
        p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()[0]
    return float(p)


def load_audio_mel_from_array(mel: np.ndarray) -> np.ndarray:
    if mel is None:
        return np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)

    mel = mel.astype(np.float32)
    if mel.ndim != 2:
        return np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)

    h, w = mel.shape
    if h != N_MELS:
        # If mel bins dimension is wrong, give up and use zeros to be safe
        return np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)

    if w < MEL_WIDTH:
        pad_width = MEL_WIDTH - w
        mel = np.hstack([mel, np.zeros((N_MELS, pad_width), dtype=np.float32)])
    else:
        mel = mel[:, :MEL_WIDTH]

    return mel


def audio_prob(audio_model, mel: np.ndarray) -> float:
    mel_fixed = load_audio_mel_from_array(mel)
    batch = np.expand_dims(mel_fixed, axis=(0, -1))  # (1, N_MELS, MEL_WIDTH, 1)
    tensor = tf.convert_to_tensor(batch)
    preds = audio_model.predict(tensor, verbose=0)
    return float(preds.flatten()[0])


def fusion_prob(fusion_model, text_p: float, audio_p: float) -> float:
    features = np.array([[text_p, audio_p]], dtype=np.float32)
    with torch.no_grad():
        batch = torch.from_numpy(features).to(device)
        pred = fusion_model(batch)
        return float(pred.cpu().numpy()[0])


def wav_bytes_to_mel(data: bytes) -> np.ndarray:
    """
    Convert raw audio bytes (e.g. WAV/OGG/WebM that librosa can read)
    into a mel-spectrogram using the same pipeline as training.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        y, _ = librosa.load(tmp_path, sr=SR, mono=True)
        y = trim_silence(y)
        y = maybe_reduce_noise(y, SR)
        y = rms_normalize(y)
        y = enforce_duration(y, SR, MAX_DURATION)
        mel = make_mel(y, SR, N_MELS)
        return mel.astype(np.float32)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


class PredictResponse(BaseModel):
    text_prob: float
    audio_prob: float
    fusion_prob: float
    threshold: float
    label: str


app = FastAPI(title="Phishing Fusion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """
    Load all models once at startup so each request is fast.
    """
    global TEXT_MODEL, TOKENIZER, AUDIO_MODEL, FUSION_MODEL, THRESHOLD

    TEXT_MODEL, TOKENIZER = load_text_model()
    AUDIO_MODEL = load_audio_model()
    FUSION_MODEL = load_fusion_model()
    THRESHOLD = load_threshold(default=0.5)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    text: Optional[str] = Form(
        None, description="Message text to classify (optional if audio is provided)"
    ),
    audio_file: Optional[UploadFile] = File(
        None, description="Mel spectrogram as .npy file (optional if text is provided)"
    ),
):
    """
    Predict phishing vs benign for a single sample.

    Works with:
      - text only
      - audio only (.npy mel spectrogram)
      - both text and audio (fusion)
    """
    has_text = bool(text and text.strip())
    has_audio = audio_file is not None

    if not has_text and not has_audio:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least text or audio.",
        )

    # Compute text probability if text is present
    t_prob_val: Optional[float] = None
    if has_text:
        t_prob_val = text_prob(TEXT_MODEL, TOKENIZER, text.strip())

    # Compute audio probability if audio file is present
    a_prob_val: Optional[float] = None
    if has_audio:
        contents = await audio_file.read()
        try:
            # Save to a temp file because np.load prefers file-like / path when allow_pickle=False
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            mel = np.load(tmp_path, allow_pickle=False)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        a_prob_val = audio_prob(AUDIO_MODEL, mel)

    # Decide which probability drives the final decision
    if t_prob_val is not None and a_prob_val is not None:
        f_prob_val = fusion_prob(FUSION_MODEL, t_prob_val, a_prob_val)
    elif t_prob_val is not None:
        f_prob_val = t_prob_val
    elif a_prob_val is not None:
        f_prob_val = a_prob_val
    else:
        # Should not happen due to earlier check
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid input provided.",
        )

    label = "phishing" if f_prob_val >= THRESHOLD else "benign"

    return PredictResponse(
        text_prob=float(t_prob_val) if t_prob_val is not None else 0.0,
        audio_prob=float(a_prob_val) if a_prob_val is not None else 0.0,
        fusion_prob=float(f_prob_val),
        threshold=THRESHOLD,
        label=label,
    )


@app.post("/predict_recorded", response_model=PredictResponse)
async def predict_recorded(
    text: Optional[str] = Form(
        None, description="Message text to classify (optional if audio is provided)"
    ),
    audio_wav: UploadFile = File(..., description="Recorded audio (WAV/OGG/WebM)"),
):
    """
    Predict phishing vs benign for a single (text, recorded-audio) pair.

    This endpoint accepts raw audio formats browsers typically produce
    and converts them to a mel-spectrogram on the backend.
    """
    # Read raw audio bytes and convert to mel spectrogram
    data = await audio_wav.read()
    mel = wav_bytes_to_mel(data)

    t_prob_val: Optional[float] = None
    if text and text.strip():
        t_prob_val = text_prob(TEXT_MODEL, TOKENIZER, text.strip())

    a_prob_val = audio_prob(AUDIO_MODEL, mel)

    # Fusion when both are present, otherwise fall back to the available modality
    if t_prob_val is not None:
        f_prob_val = fusion_prob(FUSION_MODEL, t_prob_val, a_prob_val)
    else:
        f_prob_val = a_prob_val

    label = "phishing" if f_prob_val >= THRESHOLD else "benign"

    return PredictResponse(
        text_prob=float(t_prob_val) if t_prob_val is not None else 0.0,
        audio_prob=float(a_prob_val),
        fusion_prob=float(f_prob_val),
        threshold=THRESHOLD,
        label=label,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


