import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from models.context_fusion_model import ContextFusionMLP

model_path = "models/context_fusion_model.pt"

def load_and_test():
    print(f"Loading model from {model_path}...")
    model = ContextFusionMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test Case 1: High Text Risk (0.9), Low Audio (0.0), Trusted Sender (Rep 1.0), No URL (Risk 0.0)
    # Expected: Low Risk (Benign) because trusted sender
    x1 = torch.tensor([[0.9, 0.0, 1.0, 0.0]])
    with torch.no_grad():
        out1 = model(x1).item()
    print(f"Case 1 (Trusted Sender): Input=[0.9, 0.0, 1.0, 0.0] -> Output={out1:.4f}")
    
    # Test Case 2: High Text Risk (0.9), Low Audio (0.0), Random Sender (Rep 0.2), URL (Risk 1.0)
    # Expected: High Risk (Phishing)
    x2 = torch.tensor([[0.9, 0.0, 0.2, 1.0]])
    with torch.no_grad():
        out2 = model(x2).item()
    print(f"Case 2 (Phishing):       Input=[0.9, 0.0, 0.2, 1.0] -> Output={out2:.4f}")
    
    if out1 < 0.5 and out2 > 0.5:
        print("✅ Model logic seems correct!")
    else:
        print("⚠️ Model logic might need tuning or more data.")

if __name__ == "__main__":
    load_and_test()
