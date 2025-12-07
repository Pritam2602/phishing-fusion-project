import requests
import json

BASE_URL = "http://localhost:8000"

def test_context_logic():
    print("Testing Context-Aware Logic...")

    # Case 1: Trusted Sender (Bank)
    # "Your account is blocked" is normally suspicious, but from "VM-HDFCBK" it should be safer.
    payload_trusted = {
        "text": "Your account is blocked due to suspicious login attempts.",
        "sender_id": "VM-HDFCBK",
        "is_short_code": False,
        "has_url": False
    }
    
    # We expect this to be BENIGN or have a low fusion_prob
    # (Existing model might give high text_prob, but sender_reputation should lower the final score)
    try:
        # Note: In a real test we'd need the server running. 
        # Since I can't easily start a background server and keep it running for a simple script without blocking,
        # I will simulate the logic or assume the user will run this.
        # But wait, I can modify this script to IMPORT app and test the logic directly without spinning up uvicorn!
        pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # DIRECT TESTING OF LOGIC (Bypassing HTTP to ensure logic works strictly)
    import sys
    import os
    sys.path.append(os.getcwd())
    
    from context_utils import calculate_sender_reputation, calculate_url_risk, weighted_fusion
    
    print("--- Unit Testing Heuristics ---")
    
    # 1. Test Sender Reputation
    rep_trusted = calculate_sender_reputation("VM-HDFCBK", False)
    print(f"Sender 'VM-HDFCBK': {rep_trusted} (Expected 1.0)")
    assert rep_trusted == 1.0
    
    rep_short = calculate_sender_reputation("56767", True)
    print(f"Sender '56767' (Short): {rep_short} (Expected 0.8)")
    assert rep_short == 0.8
    
    rep_random = calculate_sender_reputation("+919876543210", False)
    print(f"Sender '+919876543210': {rep_random} (Expected 0.2)")
    assert rep_random == 0.2
    
    # 2. Test URL Risk
    risk_url = calculate_url_risk(True)
    print(f"Has URL: {risk_url} (Expected 0.8)")
    assert risk_url == 0.8
    
    risk_no_url = calculate_url_risk(False)
    print(f"No URL: {risk_no_url} (Expected 0.0)")
    assert risk_no_url == 0.0
    
    # 3. Test Fusion Logic
    # Scenario A: "Your account is blocked" (Text High Risk ~0.8) + Trusted Sender (Rep 1.0)
    # Expected: Low Risk (Security Alert)
    score_a = weighted_fusion(text_p=0.8, audio_p=0.0, sender_rep=1.0, url_risk=0.0)
    print(f"Scenario A (High Semantic + Trusted Sender): {score_a}")
    # Logic: if sender_rep >= 0.9 -> min(0.3, content_risk)
    # Should be 0.3
    assert score_a <= 0.3
    
    # Scenario B: "Your account is blocked" (Text High Risk ~0.8) + Random Sender (Rep 0.2) + URL (Risk 0.8)
    # Expected: Very High Risk
    # Logic: 0.5*0.8 + 0.3*(1-0.2) + 0.2*0.8 = 0.4 + 0.24 + 0.16 = 0.8
    score_b = weighted_fusion(text_p=0.8, audio_p=0.0, sender_rep=0.2, url_risk=0.8)
    print(f"Scenario B (High Semantic + Random Sender + URL): {score_b}")
    assert score_b >= 0.7
    
    print("\n✅ All logic checks passed!")
