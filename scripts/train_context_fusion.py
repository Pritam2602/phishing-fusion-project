import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.getcwd())

from models.context_fusion_model import ContextFusionMLP

def train(args):
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🔧 Using device: {device}")
    
    print(f"📥 Loading training data from {args.train_csv}...")
    df = pd.read_csv(args.train_csv)
    
    required_cols = ["text_prob", "audio_prob", "sender_reputation", "url_risk", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
            
    label_map = {"benign": 0, "phishing": 1}
    df["label"] = df["label"].map(lambda x: label_map.get(x, int(x)))
    
    # Features: text_prob, audio_prob, sender_reputation, url_risk
    X = df[["text_prob", "audio_prob", "sender_reputation", "url_risk"]].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    
    # Split
    n = len(X)
    split_idx = int(n * (1 - args.val_split))
    idx = np.random.permutation(n)
    
    X_train, X_val = X[idx[:split_idx]], X[idx[split_idx:]]
    y_train, y_val = y[idx[:split_idx]], y[idx[split_idx:]]
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = ContextFusionMLP().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"\n🚀 Training ContextFusionMLP for {args.epochs} epochs...\n")
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                val_correct += ((pred > 0.5).float() == y_batch).sum().item()
                val_total += y_batch.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
            torch.save(model.state_dict(), args.model_out)
            
    print(f"\n✅ Training complete. Best model saved to {args.model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/fusion/fusion_context.csv")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model-out", default="models/context_fusion_model.pt")
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.train_csv):
        print(f"Dataset {args.train_csv} not found. Please run generate_context_dataset.py first.")
    else:
        train(args)
