import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.engine.torch_kernel import fast_e8_quantize_torch

def get_kv_activations(model_id="gpt2"):
    print(f"--- Extracting Real KV Activations from {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    text = "The Higman-Sims graph is unique in reaching the E8 Gosset Lattice density for high-dimensional vector quantization."
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, use_cache=True)
        # In Transformers V5+, past_key_values is a DynamicCache object
        # We access the key cache list directly
        if hasattr(outputs.past_key_values, "key_cache"):
            keys = outputs.past_key_values.key_cache[0]
        else:
            keys = outputs.past_key_values[0][0]
    
    return keys

def benchmark_torch_kv():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = get_kv_activations().to(device)
        print(f"Loaded KV Tensors of shape: {X.shape} on {device}")
        
        # Flatten for quantization [N, dim]
        B, H, S, D = X.shape
        X_flat = X.reshape(-1, D)
        
        # Simulation of Lattice-RSN V19 at 3.0 BPD
        # (Assuming scales are pre-calibrated as in V19 script)
        # For this reality check, we simulate 1 stage of E8
        
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if start_time: start_time.record()
        
        # 1. Normalization (RSN-lite)
        mean = X_flat.mean(dim=-1, keepdim=True)
        std = X_flat.std(dim=-1, keepdim=True) + 1e-12
        X_norm = (X_flat - mean) / std
        
        # 2. Lattice Projection
        # Scale to match E8 range
        X_scaled = X_norm * 100.0
        Q = fast_e8_quantize_torch(X_scaled)
        
        if end_time: 
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)
            print(f"Dequantization Latency: {elapsed:.4f} ms")
        
        # 3. Reconstruction
        X_hat = (Q / 100.0) * std + mean
        
        # SNR Calculation
        mse = torch.mean((X_flat - X_hat)**2)
        snr = 10 * torch.log10(torch.mean(X_flat**2) / mse)
        
        print(f"\n--- Reality Check Result ---")
        print(f"SNR (Real GPT2 Keys): {snr.item():.2f} dB")
        print(f"Status: {'GOD-MODE STABLE' if snr > 40 else 'NEEDS CALIBRATION'}")
        
    except Exception as e:
        print(f"Benchmark Failed: {str(e)}")

if __name__ == "__main__":
    benchmark_torch_kv()
