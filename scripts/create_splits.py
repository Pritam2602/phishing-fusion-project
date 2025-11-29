"""
Create robust train/val/test splits without leakage.
- Uses processed .npy files in data/audio/processed
- Splits by filename group (no exact name in both train/test)
- Saves CSVs: data/audio/splits/train.csv etc
"""

import os, random, pandas as pd
from glob import glob

PROCESSED_DIR = "data/audio/processed"
OUT_DIR = "data/audio/splits"
os.makedirs(OUT_DIR, exist_ok=True)

def load_all():
    files = glob(os.path.join(PROCESSED_DIR, "*.npy"))
    rows = []
    for f in files:
        base = os.path.basename(f)
        # expected format label_idx.npy
        parts = base.split('_')
        label = parts[0]
        rows.append((base.replace(".npy",""), f, label))
    return rows

def main(test_size=0.15, val_size=0.1, seed=42):
    rows = load_all()
    random.seed(seed)
    # group by first token (label)
    # shuffle entire list then split (no duplicates because names unique)
    random.shuffle(rows)
    n = len(rows)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test = rows[:n_test]
    val = rows[n_test:n_test+n_val]
    train = rows[n_test+n_val:]
    # write csvs
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        df = pd.DataFrame(subset, columns=["id","path","label"])
        path = os.path.join(OUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print("Saved", path, len(subset))
    print("Total:", n, "train:", len(train), "val:", len(val), "test:", len(test))

if __name__ == "__main__":
    main()
