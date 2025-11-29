import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import json
from datetime import datetime

# ============= CLI ARGS =============
parser = argparse.ArgumentParser(description="Train fusion model on text+audio probabilities")
parser.add_argument("--train-csv", default="data/fusion/fusion.csv", help="Train CSV with text_prob, audio_prob, label")
parser.add_argument("--val-csv", default=None, help="Validation CSV (if None, split from train)")
parser.add_argument("--val-split", type=float, default=0.2, help="Validation split if --val-csv not provided")
parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--model-out", default="models/fusion_model.pt", help="Path to save model")
parser.add_argument("--cpu", action="store_true", help="Force CPU")
args = parser.parse_args()

device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"\n🔧 Using device: {device}")

# ============= LOAD DATA =============
print(f"\n📥 Loading training data from {args.train_csv}...")
train_df = pd.read_csv(args.train_csv)

# Expected columns: text_prob, audio_prob, label
if "label" not in train_df.columns:
    raise ValueError("CSV must have 'label' column")

# Handle missing text_prob/audio_prob (use 0.5 as neutral)
if "text_prob" not in train_df.columns:
    print("⚠️  'text_prob' not found, using 0.5 as default")
    train_df["text_prob"] = 0.5
if "audio_prob" not in train_df.columns:
    print("⚠️  'audio_prob' not found, using 0.5 as default")
    train_df["audio_prob"] = 0.5

# Convert labels to binary (benign=0, phishing=1)
label_map = {"benign": 0, "phishing": 1}
train_df["label"] = train_df["label"].map(lambda x: label_map.get(x, int(x)))

# Extract features and labels
X_train = train_df[["text_prob", "audio_prob"]].values.astype(np.float32)
y_train = train_df["label"].values.astype(np.float32)

# Load validation data if provided, else split
if args.val_csv:
    print(f"📥 Loading validation data from {args.val_csv}...")
    val_df = pd.read_csv(args.val_csv)
    if "text_prob" not in val_df.columns:
        val_df["text_prob"] = 0.5
    if "audio_prob" not in val_df.columns:
        val_df["audio_prob"] = 0.5
    val_df["label"] = val_df["label"].map(lambda x: label_map.get(x, int(x)))
    X_val = val_df[["text_prob", "audio_prob"]].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
else:
    # Split train/val
    n = len(X_train)
    split_idx = int(n * (1 - args.val_split))
    idx = np.random.permutation(n)
    X_val = X_train[idx[split_idx:]]
    y_val = y_train[idx[split_idx:]]
    X_train = X_train[idx[:split_idx]]
    y_train = y_train[idx[:split_idx]]

print(f"  Train: {len(X_train)} samples")
print(f"  Val:   {len(X_val)} samples")

# ============= CREATE DATASETS =============
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# ============= MODEL =============
class FusionMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = FusionMLP().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

print(f"\n🧠 Model:\n{model}")

# ============= TRAINING LOOP =============
print(f"\n🚀 Training for {args.epochs} epochs...\n")
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
best_val_loss = float("inf")

for epoch in range(args.epochs):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += ((pred > 0.5).float() == y_batch).sum().item()
        train_total += y_batch.size(0)
    
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    
    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            
            val_loss += loss.item()
            val_correct += ((pred > 0.5).float() == y_batch).sum().item()
            val_total += y_batch.size(0)
            
            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_auc = roc_auc_score(val_targets, val_preds) if len(set(val_targets)) > 1 else 0.0
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_auc"].append(val_auc)
    
    print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
    
    # Checkpoint on best val loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        torch.save(model.state_dict(), args.model_out)
        print(f"  ✅ Saved checkpoint to {args.model_out}")

# ============= SAVE HISTORY =============
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
hist_path = os.path.join("logs", "fusion", f"history_{run_id}.json")
os.makedirs(os.path.dirname(hist_path), exist_ok=True)
with open(hist_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"\n📊 Training history saved to: {hist_path}")

# ============= FINAL EVAL =============
model.eval()
with torch.no_grad():
    train_preds = []
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch)
        train_preds.extend(pred.cpu().numpy())
    
    val_preds = []
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch)
        val_preds.extend(pred.cpu().numpy())

print(f"\n🎉 Training complete!")
print(f"Best model saved to: {args.model_out}")