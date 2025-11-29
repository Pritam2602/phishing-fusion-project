import os
import asyncio
import random
import edge_tts
import shutil

OUT_DIR = "data/audio/phishing"

# Overwrite existing folder
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

VOICE = "en-US-AriaNeural"
RATE = "+0%"

BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "Bank of Baroda"]
APPS = ["GPay", "PhonePe", "Paytm", "Amazon Pay"]
REASONS = [
    "your K Y C has expired",
    "a suspicious login was detected",
    "your account is temporarily blocked",
    "a fraud transaction was attempted",
    "your U P I ID is under verification",
    "your SIM card is scheduled for deactivation",
]
ACTIONS = [
    "share the O T P",
    "confirm the verification code",
    "provide the four digit number",
    "tell the six digit code",
    "speak the authentication digits"
]

async def generate(text, path):
    tts = edge_tts.Communicate(text, VOICE, rate=RATE)
    await tts.save(path)

async def main():
    print("🎙 Generating 500 unique phishing scam audios...")

    for i in range(500):
        otp_digits = random.randint(100000, 999999)
        sentence = (
            f"This is {random.choice(BANKS)} bank. "
            f"We noticed that {random.choice(REASONS)}. "
            f"Please {random.choice(ACTIONS)} "
            f"{otp_digits} immediately to avoid service interruption."
        )

        out_path = f"{OUT_DIR}/phishing_{i}.wav"
        print(f"[{i+1}/500] {sentence}")
        await generate(sentence, out_path)

    print("\n✅ Finished generating 500 phishing audios.")
    print(f"Saved in: {OUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
