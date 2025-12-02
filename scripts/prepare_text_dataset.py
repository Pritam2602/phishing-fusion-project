import os
import pandas as pd
import re
import chardet


ENRON_PATH = "data/kaggle_datasets/enron_email/emails.csv"
SMS_SPAM_PATH = "data/kaggle_datasets/sms_spam/spam.csv"
CHATGPT_DIR = "data/chatgpt_dataset"

OUT_CSV = "data/text_dataset.csv"


def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_read_csv(path):
    print(f"\n📄 Reading CSV safely: {path}")

    # Try UTF-8
    try:
        df = pd.read_csv(path, encoding="utf-8")
        print("✔ Loaded using UTF-8")
        return df
    except:
        print("❌ UTF-8 failed, detecting encoding...")

    # Auto-detect encoding
    with open(path, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]

    print(f"🔍 Detected encoding: {encoding}")

    # Load using detected encoding
    df = pd.read_csv(path, encoding=encoding)   
    print(f"✔ Loaded using {encoding}")

    return df


rows = []


if os.path.exists(ENRON_PATH):
    print("\n Loading Enron Emails...")
    df_enron = safe_read_csv(ENRON_PATH)
    text_col = "text" if "text" in df_enron.columns else df_enron.columns[-1]
    for text in df_enron[text_col]:
        rows.append([clean(text), "benign"])


if os.path.exists(SMS_SPAM_PATH):
    print("\n Loading SMS Spam Dataset...")
    df_sms = safe_read_csv(SMS_SPAM_PATH)

    # Your file uses v1 for label and v2 for text
    text_col = "v2"
    label_col = "v1"

    for _, row in df_sms.iterrows():
        text = clean(str(row[text_col]))
        label_raw = str(row[label_col]).lower()

        if "spam" in label_raw:
            rows.append([text, "phishing"])
        else:
            rows.append([text, "benign"])



print("\n Loading ChatGPT Dataset...")
for fname in os.listdir(CHATGPT_DIR):
    if fname.endswith(".txt"):
        label = "phishing" if "phish" in fname.lower() else "benign"
        with open(os.path.join(CHATGPT_DIR, fname), "r", encoding="utf-8") as f:
            rows.append([clean(f.read()), label])

df_out = pd.DataFrame(rows, columns=["text", "label"])
df_out.to_csv(OUT_CSV, index=False)

print("\n Text dataset created successfully!")
print(" Saved to:", OUT_CSV)
print(" Total samples:", len(df_out))
print(df_out["label"].value_counts())
