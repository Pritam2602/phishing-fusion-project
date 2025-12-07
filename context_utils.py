from typing import Optional

# ---------------------------
# HEURISTIC CONTEXT HELPERS
# ---------------------------
def calculate_sender_reputation(sender_id: Optional[str], is_short_code: bool) -> float:
    """
    Returns a score from 0.0 (Bad/Unknown) to 1.0 (Trusted).
    Logic:
    - Known trusted patterns (e.g. HDFCBK, SBIBNK, GOVT ids) -> High score
    - Short codes -> Medium-High (often transactional)
    - Random/Unknown -> Low
    """
    if not sender_id:
        return 0.0
    
    sender = sender_id.upper().strip()
    
    # 1. Known TRUSTED suffixes/patterns (Bank specific)
    trusted_suffixes = ["HDFCBK", "SBIBNK", "ICICIB", "AXISBK", "SBI", "EPFO", "GOVT", "NIC"]
    for suffix in trusted_suffixes:
        if suffix in sender:
            return 1.0  # Trusted Entity
            
    # 2. Short codes (usually 5-6 digits) are often legit services (though can be spoofed, they are better than random mobile nums)
    if is_short_code:
        # Check if it looks like a valid short code (digits)
        if sender.isdigit() and len(sender) <= 6:
            return 0.8
        # Short-code style alphanumeric sender ids like "VK-HDFCBK" are also trusted if they follow format
        if "-" in sender and len(sender) == 9: # generic sender id format
            return 0.7
            
    # 3. If it looks like a mobile number -> LOW reputation for official notifications
    # (Assuming official things shouldn't come from personal +91 numbers)
    return 0.2

def calculate_url_risk(has_url: bool) -> float:
    """
    Returns a score from 0.0 (Safe/No URL) to 1.0 (High Risk).
    Logic:
    - No URL -> 0.0 risk (neutral)
    - Has URL -> 0.8 risk (default high caution for generic links)
    """
    if not has_url:
        return 0.0
    # In a real system, we would check the specific URL against a blocklist.
    # Here, presence of URL increases risk significantly.
    return 0.8

def weighted_fusion(text_p: float, audio_p: float, sender_rep: float, url_risk: float) -> float:
    """
    Late Fusion Logic:
    Final_Risk = w1*Semantic + w2*Audio + w3*(1 - Sender_Rep) + w4*URL_Risk
    
    We want a Probability of PHISHING (0 to 1).
    - Semantic, Audio, URL_Risk: Higher = Phishing
    - Sender_Rep: Higher = TRUSTED (Lower Phishing Risk) -> so we use (1 - sender_rep)
    """
    
    # Weights for risk contribution
    # If text/audio model is confident, it pushes score up.
    # If sender is trusted, it pushes score DOWN significantly.
    # If URL is present, it pushes score UP.
    
    # Heuristic weights
    W_SEMANTIC = 0.4
    W_AUDIO = 0.1     # audio might be missing in text-only cases
    W_SENDER = 0.3    # significantly reduces risk if trusted
    W_URL = 0.2       # adds risk
    
    # Normalize inputs
    # If audio is 0 (missing), we re-balance or just accept lower sum?
    # Better: simple linear combo for now.
    
    # Term for sender risk: (1.0 - sender_rep). If rep=1.0 (Trusted), risk contribution is 0.
    sender_risk = 1.0 - sender_rep
    
    # Base fusion from model
    # If we have a fusion model output, we can use that as the "Content Risk"
    # But current fusion_prob takes text & audio.
    
    # Let's define Content Risk
    content_risk = text_p
    if audio_p > 0:
        content_risk = max(text_p, audio_p) # or average
        
    # Final Formula
    # fusion_score = 0.5 * content_risk + 0.3 * sender_risk + 0.2 * url_risk
    
    # Refined logic user requested:
    # "if sender_rep HIGH -> benign"
    
    # If Sender is Trusted (rep > 0.9), capped risk.
    if sender_rep >= 0.9:
        # Strongly trusted source. Hard to be phishing unless hacked.
        # Cap risk at 0.3 (Benign threshold usually 0.5)
        return min(0.3, content_risk)
        
    # If Sender is weak/unknown
    score = (0.5 * content_risk) + (0.3 * sender_risk) + (0.2 * url_risk)
    
    return min(1.0, max(0.0, score))
