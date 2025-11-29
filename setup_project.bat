@echo off
echo ============================================
echo   PHISHING FUSION PROJECT - SETUP SCRIPT
echo   Installing into D:\PhishingFusionProject
echo ============================================

REM --- CHANGE TO D DRIVE ---
D:
cd D:\PhishingFusionProject

REM --- CREATE PROJECT FOLDERS ---
mkdir data
mkdir data\chatgpt_dataset
mkdir data\kaggle_datasets
mkdir data\kaggle_datasets\sms_spam
mkdir data\kaggle_datasets\phishing_sms
mkdir data\kaggle_datasets\enron_email
mkdir data\kaggle_datasets\nazario_phishing
mkdir data\kaggle_datasets\scam_calls
mkdir data\kaggle_datasets\common_voice

mkdir data\fusion

mkdir audio
mkdir audio\phishing
mkdir audio\benign
mkdir models
mkdir models\text_model
mkdir models\audio_model
mkdir models\fusion_model
mkdir models\onnx

mkdir scripts
mkdir notebooks

echo Project folder structure created successfully.

REM --- CREATE VIRTUAL ENVIRONMENT ---
echo Creating virtual environment in D:...
python -m venv venv

if errorlevel 1 (
    echo Error: Python not found. Make sure Python is installed.
    pause
    exit /b
)

REM --- ACTIVATE THE ENVIRONMENT ---
call venv\Scripts\activate

echo Installing required Python packages...
pip install --upgrade pip
pip install transformers datasets torchaudio librosa soundfile scikit-learn accelerate TTS onnx onnxruntime joblib tqdm sentencepiece

echo Dependencies installed.

REM --- CREATE PYTHON SCRIPTS ---

echo Writing generate_audio.py...
(
echo from TTS.api import TTS
echo import csv, os
echo from tqdm import tqdm
echo tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
echo for category in ["phishing","benign"]:
echo ^    csv_path = f"data/chatgpt_dataset/{category}_vishing_scripts.csv"
echo ^    outdir = f"audio/{category}"
echo ^    os.makedirs(outdir, exist_ok=True)
echo ^    with open(csv_path, encoding='utf-8') as f:
echo ^        reader = csv.DictReader(f)
echo ^        for row in tqdm(reader):
echo ^            tts.tts_to_file(text=row["script"], file_path=f"{outdir}/{row['id']}.wav")
)>scripts\generate_audio.py

echo Writing data_prep.py...
(
echo import pandas as pd, glob, os
echo phish_text = pd.read_csv("data/chatgpt_dataset/phishing_text.csv")
echo benign_text = pd.read_csv("data/chatgpt_dataset/benign_text.csv")
echo phish_voice = pd.read_csv("data/chatgpt_dataset/phishing_vishing_scripts.csv")
echo benign_voice = pd.read_csv("data/chatgpt_dataset/benign_voice_scripts.csv")
echo text_df = pd.concat([
echo ^    phish_text[['id','text','label']].rename(columns={'text':'transcript'}),
echo ^    benign_text[['id','text','label']].rename(columns={'text':'transcript'})
echo ])
echo text_df['audio_path']=""

echo files = glob.glob("audio/*/*.wav")
echo audio_df = pd.DataFrame([
echo ^    { 'id':os.path.basename(f).replace('.wav',''), 'audio_path':f, 'label':"phishing" if "phishing" in f else "benign" }
echo ^    for f in files
echo ])
echo voice_df = pd.concat([phish_voice[['id','script']], benign_voice[['id','script']]])
echo audio_df = audio_df.merge(voice_df.rename(columns={"script":"transcript"}), on="id")
echo fusion_df = pd.concat([text_df, audio_df], ignore_index=True)
echo fusion_df.to_csv("data/fusion/fusion_dataset.csv", index=False)
)>scripts\data_prep.py

echo ============================================
echo SETUP COMPLETE!
echo Run these next inside VS Code terminal:
echo.
echo   venv\Scripts\activate
echo   python scripts\generate_audio.py
echo   python scripts\data_prep.py
echo ============================================
