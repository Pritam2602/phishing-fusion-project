import pandas as pd
import numpy as np
import os
import random

def generate_context_data(input_csv, output_csv):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # We need to add 'sender_reputation' and 'url_risk'
    # Logic:
    # If label is 'phishing':
    #   - High chance of random sender (low reputation)
    #   - High chance of URL presence (high risk)
    # If label is 'benign':
    #   - High chance of trusted sender (high reputation)
    #   - Low chance of URL (low risk)
    
    new_labels = []
    sender_reps = []
    url_risks = []
    
    for label in df['label']:
        # Ensure label is int
        lbl = 1 if label == 'phishing' or label == 1 else 0
        
        # Decide context
        if lbl == 1: # Originally Phishing (High Semantic Risk)
            
            # Scenario A: Real Phishing (Random/Spoofed Sender + Link)
            # 80% chance it stays phishing
            if random.random() < 0.8:
                # Low Reputation
                if random.random() < 0.8:
                    rep = random.uniform(0.0, 0.3) 
                else:
                    rep = random.uniform(0.6, 0.8) # Spoofed
                
                # High URL Risk
                risk = random.uniform(0.8, 1.0) if random.random() < 0.9 else 0.0
                
                # Label stays 1
                new_lbl = 1
                
            # Scenario B: "Security Alert" (Trusted Sender + No Link)
            # 20% chance we simulated a "False Positive" text from a Bank
            else:
                rep = random.uniform(0.9, 1.0) # Trusted
                risk = 0.0 # No hazardous link usually
                
                # CRITICAL: Flip label to BENIGN because it's from a trusted source
                new_lbl = 0
                
        else: # Originally Benign (Low Semantic Risk)
            # Sender Reputation: Mostly High
            rep = random.uniform(0.9, 1.0) if random.random() < 0.8 else random.uniform(0.1, 0.5)
            
            # URL Risk: Mostly Low
            risk = 0.0 if random.random() < 0.9 else random.uniform(0.0, 0.2)
            
            new_lbl = 0
        
        new_labels.append(new_lbl)
        sender_reps.append(rep)
        url_risks.append(risk)
    
    df['label'] = new_labels
    df['sender_reputation'] = sender_reps
    df['url_risk'] = url_risks
    
    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved context-enhanced dataset to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    input_path = "data/fusion/fusion.csv"
    output_path = "data/fusion/fusion_context.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
    else:
        generate_context_data(input_path, output_path)
