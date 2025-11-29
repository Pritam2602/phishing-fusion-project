import os
import asyncio
import random
import edge_tts
import shutil

OUT_DIR = "data/audio/benign_1000"

# Overwrite existing folder
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

VOICE = "en-US-AriaNeural"
RATE = "+0%"

# Dynamic benign templates
ACTIONS = [
    "play", "open", "start", "stop", "show", "increase", "decrease",
    "remind", "call", "message", "navigate to", "set", "turn on", "turn off"
]

OBJECTS = [
    "my playlist", "Bluetooth", "WiFi", "flashlight", "notes", "calendar",
    "alarm", "reminder", "workout mix", "camera", "photo gallery",
    "timer", "shopping list", "settings", "music"
]

TIMES = [
    "in ten minutes", "tomorrow morning", "at six thirty", "next Monday",
    "after two hours", "right now", "this evening", "tonight"
]

EXTRAS = [
    "please", "for me", "as soon as possible", "when you're ready", "right away"
]

async def generate(text, path):
    tts = edge_tts.Communicate(text, VOICE, rate=RATE)
    await tts.save(path)

async def main():
    print("🎙 Generating 1000 unique benign audios...")

    for i in range(1000):
        sentence = (
            f"{random.choice(ACTIONS)} "
            f"{random.choice(OBJECTS)} "
            f"{random.choice(EXTRAS)} "
            f"{random.choice(TIMES)}."
        )

        out_path = f"{OUT_DIR}/benign_{i}.wav"
        print(f"[{i+1}/1000] {sentence}")
        await generate(sentence, out_path)

    print("\n✅ Finished generating 1000 benign audios.")
    print(f"Saved in: {OUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
