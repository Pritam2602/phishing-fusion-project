

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="distilroberta-base")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--accum", type=int, default=1)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--train_csv", default="data/text/train.csv")
parser.add_argument("--val_csv", default="data/text/val.csv")
parser.add_argument("--model_out", default="models/text_distilroberta")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gamma", type=float, default=2.0)   # focal loss focusing strength
parser.add_argument("--weight_phishing", type=float, default=5.0)  # phishing weight
args = parser.parse_args()

torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Using device: {DEVICE}\n")

def ensure(path):
    os.makedirs(path, exist_ok=True)

def find_latest_epoch(base_dir):
    if not os.path.exists(base_dir):
        return 0, None
    nums = []
    for x in os.listdir(base_dir):
        if x.startswith("epoch_"):
            try:
                nums.append(int(x.split("_")[1]))
            except:
                pass
    if not nums:
        return 0, None
    last = max(nums)
    return last, os.path.join(base_dir, f"epoch_{last}")


print("📥 Loading CSVs...")
dataset = load_dataset("csv", data_files={
    "train": args.train_csv,
    "val": args.val_csv
})

label_map = {"benign": 0, "phishing": 1}

def clean_batch(batch):
    texts, labels = [], []
    for t, l in zip(batch["text"], batch["label"]):
        # text
        if t is None or (isinstance(t, float) and np.isnan(t)):
            texts.append("")
        else:
            texts.append(str(t))

        # label
        labels.append(label_map.get(str(l).lower().strip(), 0))
    return {"text": texts, "labels": labels}

dataset = dataset.map(clean_batch, batched=True)

print(" Tokenizing (batched)...")

tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=args.max_len
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(dataset["train"], batch_size=args.batch, shuffle=True)
val_loader = DataLoader(dataset["val"], batch_size=args.batch)

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=5.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)

        # p_t = softmax probabilities of the correct class
        pt = torch.softmax(logits, dim=1)
        pt = pt[range(len(labels)), labels]

        focal_term = (1 - pt) ** self.gamma

        # apply phishing weight (alpha)
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[labels == 1] = self.alpha

        loss = weights * focal_term * ce_loss
        return loss.mean()

focal_loss = WeightedFocalLoss(alpha=args.weight_phishing, gamma=args.gamma)


ensure(args.model_out)
last_epoch, last_dir = find_latest_epoch(args.model_out)
start_epoch = last_epoch + 1
print(f" Latest checkpoint: epoch {last_epoch}")
print(f" Starting from epoch {start_epoch}")


print("\n Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=2
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = max(1, len(train_loader) * args.epochs // max(1, args.accum))
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

# resume model?
if last_dir:
    ckpt_path = os.path.join(last_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"⤵ Loading checkpoint from {ckpt_path}")
        model = AutoModelForSequenceClassification.from_pretrained(last_dir, num_labels=2).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if DEVICE == "cuda" and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1

best_recall = 0.0

for epoch in range(start_epoch, args.epochs + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train E{epoch}")

    optimizer.zero_grad()

    for step, batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(ids, attention_mask=mask).logits
            loss = focal_loss(logits, labels) / args.accum

        if DEVICE == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # accumulate
        if (step + 1) % args.accum == 0:
            if DEVICE == "cuda":
                scaler.unscale_(optimizer)
                try:
                    scaler.step(optimizer)
                except Exception:
                    optimizer.step()
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += float(loss.item() * args.accum)
        pbar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")


    model.eval()
    preds, golds, probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(ids, attention_mask=mask).logits
            soft = torch.softmax(logits, dim=1)[:, 1]  # phishing prob

            probs.extend(soft.cpu())
            preds.extend(torch.argmax(logits, dim=1).cpu())
            golds.extend(labels.cpu())

    acc = accuracy_score(golds, preds)
    print(f"\n Epoch {epoch} Validation Accuracy: {acc:.4f}")
    print(classification_report(golds, preds, target_names=["benign", "phishing"]))

    best_threshold = 0.01
    best_recall_epoch = 0

    for t in np.linspace(0.01, 0.5, 50):
        pred_thr = (np.array(probs) >= t).astype(int)
        recall = np.sum((pred_thr == 1) & (np.array(golds) == 1)) / max(1, np.sum(np.array(golds) == 1))
        if recall > best_recall_epoch:
            best_recall_epoch = recall
            best_threshold = t

    print(f" Best threshold={best_threshold:.3f}, recall={best_recall_epoch:.4f}")

    # save if improved recall
    if best_recall_epoch > best_recall:
        best_recall = best_recall_epoch
        best_dir = os.path.join(args.model_out, "best")
        ensure(best_dir)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

        ckpt = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if DEVICE == "cuda" else None
        }
        torch.save(ckpt, os.path.join(best_dir, "checkpoint.pt"))
        print(" New best recall model saved.")

    # save epoch checkpoint
    epoch_dir = os.path.join(args.model_out, f"epoch_{epoch}")
    ensure(epoch_dir)
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    torch.save({
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if DEVICE == "cuda" else None
    }, os.path.join(epoch_dir, "checkpoint.pt"))

    # prompt
    ans = input("Continue training? (y/n): ").strip().lower()
    if ans != "y":
        break

# final save
final_dir = os.path.join(args.model_out, "final")
ensure(final_dir)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"\n Training complete. Final saved to {final_dir}")

