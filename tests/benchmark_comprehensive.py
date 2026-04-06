import torch
import numpy as np
import os, sys, time, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Unified Importer
from src.higman_sims_quant_v12 import Untouchable_Core as V12_Core
from src.higman_sims_quant_v13 import Infinite_V13 as V13_Core
from src.higman_sims_quant_v14 import Singularity_V14 as V14_Core
from src.higman_sims_quant_v15 import Absolute_V15 as V15_Core
from src.higman_sims_quant_v16 import Final_God_V16 as V16_Core

def harvest_kv(model_id="openai-community/gpt2"):
    print(f"--- HARVESTING DATA ({model_id}) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    prompt = "The Higman-Sims group and the E8 lattice are fundamental to optimal sphere packing theory."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    keys = torch.stack([layer[0] for layer in outputs.past_key_values]).cpu().numpy()
    dim = keys.shape[-1]
    X = keys.reshape(-1, dim)
    # Shuffle for fairness
    np.random.seed(42)
    np.random.shuffle(X)
    return X[:5000].astype(np.float32) 

def configure_version(name, v_class, dim, size_tier):
    """Factory to configure versions for specific BPD targets."""
    # Size Tiers mapping to stages for E8 (roughly 1.5 - 2.5 BPD per stage)
    # Small (8 BPD)
    # Medium (16-18 BPD)
    # Large (48 BPD)
    
    if "V12" in name:
        stages = {"small": 5, "medium": 12, "large": 32}[size_tier]
        return v_class(dim=dim, stages=stages)
    
    if "V16" in name:
        stages = {"small": 2, "medium": 5, "large": 18}[size_tier]
        return v_class(dim=dim, max_stages=stages)
        
    # V13 - V15
    stages = {"small": 4, "medium": 10, "large": 24}[size_tier]
    return v_class(dim=dim, max_stages=stages)

def run_test(name, v_class, X, size_tier):
    dim = X.shape[-1]
    try:
        eng = configure_version(name, v_class, dim, size_tier)
    except:
        eng = v_class(dim=dim)
        
    t0 = time.time()
    eng.fit(X)
    co = eng.encode(X)
    d = eng.decode(co)
    lat = time.time() - t0
    
    mse = np.mean((X - d)**2)
    snr = 10 * np.log10(np.mean(X**2) / max(mse, 1e-35))
    
    if hasattr(eng, 'measure_efficiency'):
        raw_bpd, _ = eng.measure_efficiency(co)
    elif hasattr(eng, 'bpd'):
        raw_bpd = eng.bpd()
    else:
        # Lattice (8-bit) + Scalar (8-bit) = 16 bits per 8D = 2 BPD per stage
        raw_bpd = len(co) * 2.0 
        
    return {"snr": snr, "bpd": raw_bpd, "lat": lat}

if __name__ == "__main__":
    X = harvest_kv()
    tiers = ["small", "medium", "large"]
    
    versions = [
        ("V12", V12_Core),
        ("V13", V13_Core),
        ("V14", V14_Core),
        ("V15", V15_Core),
        ("V16", V16_Core),
    ]
    
    final_matrix = {}
    for tier in tiers:
        print(f"\n--- TIER: {tier.upper()} ---")
        final_matrix[tier] = {}
        for name, v_class in versions:
            res = run_test(name, v_class, X, tier)
            final_matrix[tier][name] = res
            print(f"{name:5} | SNR: {res['snr']:7.2f} dB | BPD: {res['bpd']:5.2f}")

    # Definitive Table Output
    print("\n" + "="*60)
    print("GOD-MODE COMPARISON MATRIX (SNR dB)")
    print("="*60)
    print(f"{'Version':10} | {'Small (8BPD)':12} | {'Med (18BPD)':12} | {'Large (48BPD)':12}")
    for name in [v[0] for v in versions]:
        s = final_matrix['small'][name]['snr']
        m = final_matrix['medium'][name]['snr']
        l = final_matrix['large'][name]['snr']
        print(f"{name:10} | {s:12.2f} | {m:12.2f} | {l:12.2f}")
