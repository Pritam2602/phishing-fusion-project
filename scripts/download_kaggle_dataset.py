import os
import zipfile

DATASETS = {
    # SMS datasets
    "uciml/sms-spam-collection-dataset": "sms_spam",
    "d4rk-lucif3r/dhamma-spam-sms-dataset": "phishing_sms",   # REPLACEMENT

    # Email datasets
    "mrtherobot/large-text-phishing-email-dataset": "nazario_phishing",
    "wcukierski/enron-email-dataset": "enron_email",

    # Voice / scam call datasets
    "sapdataset/audio-phone-fraud-detection": "scam_calls",

    # Benign voice
    "mozillaorg/common-voice": "common_voice"
}

BASE_DIR = "data/kaggle_datasets"


def folder_has_files(folder):
    """Check if the dataset folder already has extracted files."""
    for root, dirs, files in os.walk(folder):
        if files:
            return True
    return False


def make_folders():
    os.makedirs(BASE_DIR, exist_ok=True)
    for folder in DATASETS.values():
        os.makedirs(f"{BASE_DIR}/{folder}", exist_ok=True)


def download_and_extract():
    for dataset, folder in DATASETS.items():

        target_path = f"{BASE_DIR}/{folder}"

        # Skip if already downloaded
        if folder_has_files(target_path):
            print(f"\nSkipping {dataset} → already downloaded.")
            continue

        print(f"\nDownloading {dataset} ...")
        
        # Try Kaggle API download
        result = os.system(f"kaggle datasets download -d {dataset} -p {target_path}")

        # If result != 0 then download failed
        if result != 0:
            print(f"\n⚠️  Failed to download {dataset}")
            print("➡️  If you see '403 Forbidden', manually open the dataset on Kaggle and accept the terms.")
            print("➡️  Dataset page:", f"https://www.kaggle.com/datasets/{dataset}")
            continue

        # Extract all zip files in folder
        zip_files = [z for z in os.listdir(target_path) if z.endswith(".zip")]
        for zipf in zip_files:
            zip_path = f"{target_path}/{zipf}"
            print(f"Extracting {zipf}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(target_path)
            os.remove(zip_path)

    print("\nALL DATASETS DOWNLOADED & EXTRACTED ")


if __name__ == "__main__":
    make_folders()
    download_and_extract()
