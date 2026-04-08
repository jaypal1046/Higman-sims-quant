import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from higman_sims_quant_v16 import Final_God_V16
from higman_sims_quant_v17 import LatticeRSN_V17
from higman_sims_quant_v18 import LatticeRSN_V18
import time

def get_real_activations(model_id="gpt2"):
    print(f"--- Extracting Real Activations from {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Diverse text for complex activations
    text = "The Higman-Sims graph is a strongly regular graph with 100 vertices. It is unique in reaching the E8 Gosset Lattice density for high-dimensional vector quantization in LLM KV-caches."
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get hidden states from a deep layer (layer 8 for GPT-2)
        hidden_states = outputs.hidden_states[8].squeeze(0).numpy()
    
    return hidden_states.astype(np.float32)

def benchmark_all():
    X = get_real_activations()
    dim = X.shape[-1]
    X64 = X.astype(np.float64)
    signal = np.mean(X64**2)
    
    results = []
    
    # 1. Tiers and Version Configs
    # Note: V16 has a 4.0 BPD floor, so its 'Extreme' handles differently.
    configs = [
        # Version, Tier, Meta-Params
        ("V16", "Extreme", {"stages": 1}),
        ("V16", "Medium",  {"stages": 2}),
        ("V16", "Loose",   {"stages": 4}),
        
        ("V17", "Extreme", {"stages": 1}),
        ("V17", "Medium",  {"stages": 3}),
        ("V17", "Loose",   {"stages": 8}),
        
        ("V18", "Extreme", {"target": 2.2}),
        ("V18", "Medium",  {"target": 4.5}),
        ("V18", "Loose",   {"target": 8.0}),
    ]
    
    print("\n--- Running Multi-Tier Benchmark ---")
    for ver, tier, params in configs:
        print(f"Testing {ver} [{tier}]...")
        
        if ver == "V16":
            eng = Final_God_V16(dim, max_stages=params["stages"])
            eng.fit(X)
            co = eng.encode(X)
            d = eng.decode(co)
            bpd = eng.measure_efficiency(co)
        elif ver == "V17":
            eng = LatticeRSN_V17(dim, max_stages=params["stages"])
            eng.fit(X)
            co = eng.encode(X)
            d = eng.decode(co)
            bpd = eng.measure_efficiency(co)
        elif ver == "V18":
            eng = LatticeRSN_V18(dim, target_bpd=params["target"], max_stages=8)
            eng.fit(X)
            co = eng.encode(X)
            d = eng.decode(co)
            bpd = eng.measure_efficiency(co)
            
        mse = np.mean((X64 - d.astype(np.float64))**2)
        snr = 10 * np.log10(signal / max(mse, 1e-30))
        
        results.append({
            "Algorithm": f"Lattice-RSN ({ver})",
            "Tier": tier,
            "Bitrate (BPD)": f"{bpd:.2f}",
            "Peak SNR (dB)": f"{snr:.2f}",
            "Status": "STABLE" if snr > 20 else "N/A"
        })

    df = pd.DataFrame(results)
    print("\n--- GLOBAL PERFORMANCE MATRIX: TurboQuant Reality Check ---")
    print(df.to_string(index=False))
    
    # High-level Branding Summary
    print("\nTurboQuant: Redefining AI efficiency with extreme compression sayed.")
    
    return df

if __name__ == "__main__":
    benchmark_all()
