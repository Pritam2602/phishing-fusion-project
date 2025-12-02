# app.py (replace your existing app.py with this)
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# third-party models
import whisper  # pip install openai-whisper

# ---------------------------
# PROJECT / MODEL PATHS
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

TEXT_MODEL_DIR = MODELS_DIR / "text_distilroberta" / "best"
AUDIO_MODEL_PATH = MODELS_DIR / "audio_model" / "audio_cnn_best.h5"
FUSION_MODEL_PATH = MODELS_DIR / "fusion_model.pt"
THRESHOLD_JSON = (FUSION_MODEL_PATH.parent / "threshold.json")

# audio constants
SR = 16000
N_MELS = 64
MEL_WIDTH = 256
MAX_DURATION = 4.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# MODELS: small fusion MLP
# ---------------------------
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

# ---------------------------
# AUDIO UTILITIES (same pipeline as before)
# ---------------------------
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

# ---------------------------
# LOAD (existing) models helper functions (unchanged)
# ---------------------------
def load_text_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
    hf_logging.set_verbosity_error()
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
    state_dict = torch.load(FUSION_MODEL_PATH, map_location=device)
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

# ---------------------------
# PROBABILITY helpers (unchanged)
# ---------------------------
def text_prob(text_model, tokenizer, text: str) -> float:
    import torch
    import torch.nn.functional as F
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

# ---------------------------
# FFmpeg conversion utility
# ---------------------------
def convert_bytes_to_wav_file(data: bytes, sr: int = SR) -> str:
    """
    Save incoming bytes to a temp file, run ffmpeg to convert to a mono WAV at SR,
    and return the WAV path. Caller should remove the file.
    """
    # create temp input with guessed extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as in_tmp:
        in_tmp.write(data)
        in_path = in_tmp.name

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    # Use ffmpeg to convert input -> WAV (mono, sr)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", in_path,
        "-ar", str(sr),
        "-ac", "1",
        "-vn",
        out_path,
    ]
    # run quietly
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    try:
        # If conversion failed and file empty, ffmpeg may still have failed; let caller handle
        return out_path
    finally:
        try:
            os.remove(in_path)
        except Exception:
            pass

# ---------------------------
# Robust wav_bytes_to_mel using conversion
# ---------------------------
def wav_bytes_to_mel(data: bytes) -> np.ndarray:
    wav_path = None
    try:
        # Convert incoming bytes (webm/ogg/whatever) to wav file
        wav_path = convert_bytes_to_wav_file(data, sr=SR)

        # librosa.load the converted wav
        y, _ = librosa.load(wav_path, sr=SR, mono=True)

        # preprocess pipeline
        y = trim_silence(y)
        y = maybe_reduce_noise(y, SR)
        y = rms_normalize(y)
        y = enforce_duration(y, SR, MAX_DURATION)
        mel = make_mel(y, SR, N_MELS)
        return mel.astype(np.float32)
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

# ---------------------------
# FastAPI + startup (load models + whisper)
# ---------------------------
class PredictResponse(BaseModel):
    text_prob: float
    audio_prob: float
    fusion_prob: float
    threshold: float
    label: str
    transcription: Optional[str] = None

app = FastAPI(title="Phishing Fusion API with STT", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global model placeholders
TEXT_MODEL = None
TOKENIZER = None
AUDIO_MODEL = None
FUSION_MODEL = None
THRESHOLD = 0.5
WHISPER_MODEL = None

@app.on_event("startup")
def startup_event():
    global TEXT_MODEL, TOKENIZER, AUDIO_MODEL, FUSION_MODEL, THRESHOLD, WHISPER_MODEL
    # load models used previously
    try:
        TEXT_MODEL, TOKENIZER = load_text_model()
    except Exception as e:
        print("Warning: text model load failed:", e)
        TEXT_MODEL, TOKENIZER = None, None

    try:
        AUDIO_MODEL = load_audio_model()
    except Exception as e:
        print("Warning: audio model load failed:", e)
        AUDIO_MODEL = None

    try:
        FUSION_MODEL = load_fusion_model()
    except Exception as e:
        print("Warning: fusion model load failed:", e)
        FUSION_MODEL = None

    THRESHOLD = load_threshold(default=0.5)

    # load whisper (tiny) for fast CPU STT
    try:
        WHISPER_MODEL = whisper.load_model("small")
    except Exception as e:
        print("Warning: whisper load failed:", e)
        WHISPER_MODEL = None

# ---------------------------
# /predict (unchanged semantics) - keep compatibility
# ---------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
):
    has_text = bool(text and text.strip())
    has_audio = audio_file is not None

    if not has_text and not has_audio:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide at least text or audio.")

    t_prob_val: Optional[float] = None
    if has_text and TEXT_MODEL is not None:
        t_prob_val = text_prob(TEXT_MODEL, TOKENIZER, text.strip())

    a_prob_val: Optional[float] = None
    if has_audio:
        contents = await audio_file.read()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            mel = np.load(tmp_path, allow_pickle=False)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if AUDIO_MODEL is not None:
            a_prob_val = audio_prob(AUDIO_MODEL, mel)
        else:
            a_prob_val = 0.0

    if t_prob_val is not None and a_prob_val is not None:
        f_prob_val = fusion_prob(FUSION_MODEL, t_prob_val, a_prob_val) if FUSION_MODEL is not None else max(t_prob_val, a_prob_val)
    elif t_prob_val is not None:
        f_prob_val = t_prob_val
    else:
        f_prob_val = a_prob_val

    label = "phishing" if f_prob_val >= THRESHOLD else "benign"

    return PredictResponse(
        text_prob=float(t_prob_val) if t_prob_val is not None else 0.0,
        audio_prob=float(a_prob_val) if a_prob_val is not None else 0.0,
        fusion_prob=float(f_prob_val),
        threshold=float(THRESHOLD),
        label=label,
        transcription=None,
    )

# ---------------------------
# /predict_recorded (NEW: returns transcription)
# ---------------------------
@app.post("/predict_recorded", response_model=PredictResponse)
async def predict_recorded(
    text: Optional[str] = Form(None),
    audio_wav: UploadFile = File(...),
):
    """
    Accepts recorded audio from browser (webm/ogg/mp3/wav...) and optional text.
    Converts to WAV, runs Whisper transcription, computes mel and predictions,
    returns transcription and probabilities.
    """
    # read bytes
    data = await audio_wav.read()

    # convert to wav file path
    wav_path = convert_bytes_to_wav_file(data, sr=SR)

    transcription_text = ""
    if WHISPER_MODEL is not None:
        try:
            # whisper expects a path to file
            wh_res = WHISPER_MODEL.transcribe(wav_path)
            transcription_text = wh_res.get("text", "").strip()
        except Exception:
            transcription_text = ""

    # compute mel from bytes (we already have wav file path)
    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        y = trim_silence(y)
        y = maybe_reduce_noise(y, SR)
        y = rms_normalize(y)
        y = enforce_duration(y, SR, MAX_DURATION)
        mel = make_mel(y, SR, N_MELS).astype(np.float32)
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    # text probability: prefer explicit text param, otherwise use transcription
    t_text = (text.strip() if text and text.strip() else transcription_text).strip()
    t_prob_val: Optional[float] = None
    if t_text and TEXT_MODEL is not None:
        t_prob_val = text_prob(TEXT_MODEL, TOKENIZER, t_text)

    # audio probability
    a_prob_val = audio_prob(AUDIO_MODEL, mel) if AUDIO_MODEL is not None else 0.0

    # fusion
    if t_prob_val is not None and FUSION_MODEL is not None:
        f_prob_val = fusion_prob(FUSION_MODEL, t_prob_val, a_prob_val)
    elif t_prob_val is not None:
        f_prob_val = t_prob_val
    else:
        f_prob_val = a_prob_val

    label = "phishing" if f_prob_val >= THRESHOLD else "benign"

    return PredictResponse(
        text_prob=float(t_prob_val) if t_prob_val is not None else 0.0,
        audio_prob=float(a_prob_val),
        fusion_prob=float(f_prob_val),
        threshold=float(THRESHOLD),
        label=label,
        transcription=transcription_text or None,
    )

@app.get("/health")
def health():
    return {"status": "ok"}
