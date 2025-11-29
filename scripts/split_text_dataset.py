import os
import pandas as pd
from sklearn.model_selection import train_test_split

# CHANGE THIS
INPUT_CSV = "data/text_dataset.csv"
OUTPUT_DIR = "data/text/"   # will create train.csv / val.csv / test.csv

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Required columns: ["text", "label"]
assert "text" in df.columns, "❌ 'text' column missing"
assert "label" in df.columns, "❌ 'label' column missing"

# 80/10/10 split
train_df, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

# Save
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("✅ Split complete!")
print("Saved to:", OUTPUT_DIR)
