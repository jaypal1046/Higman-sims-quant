import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.higman_sims_quant_v12 import Untouchable_Core

def generate_local_kv_data(data_dir="tests/real_kv_data"):
    """Download a lightweight model and save its KV cache as .npy for direct testing."""
    os.makedirs(data_dir, exist_ok=True)
    model_name = "openai-community/gpt2" # Fast 500MB model proxy
    
    print(f"--- GENERATOR MODE: Loading {model_name} to bake KV files ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    prompt = "The Higman-Sims group and the E8 lattice are fundamental to optimal sphere packing theory."
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    # GPT2: past_key_values is (key, value) per layer
    keys_list = [layer[0] for layer in outputs.past_key_values]
    values_list = [layer[1] for layer in outputs.past_key_values]
    
    keys_arr = torch.cat(keys_list, dim=0).cpu().numpy().astype(np.float32)
    values_arr = torch.cat(values_list, dim=0).cpu().numpy().astype(np.float32)
    
    np.save(os.path.join(data_dir, "keys.npy"), keys_arr)
    np.save(os.path.join(data_dir, "values.npy"), values_arr)
    
    print(f"✅ Successfully baked KV cache to {data_dir}")
    print(f"   Keys shape:   {keys_arr.shape}")
    print(f"   Values shape: {values_arr.shape}")
    return keys_arr, values_arr

def run_benchmark(keys_arr, stages=8):
    """Run V12 Untouchable on the provided tensors."""
    # Flatten: (total_samples, head_dim)
    head_dim = keys_arr.shape[-1]
    X = keys_arr.reshape(-1, head_dim)
    
    print(f"\n--- BENCHMARK MODE: V12 Untouchable ({stages} Stages) ---")
    eng = Untouchable_Core(dim=head_dim, stages=stages)
    
    t0 = time.time()
    eng.fit(X[:5000])
    c = eng.encode(X)
    d = eng.decode(c)
    t1 = time.time()
    
    mse = np.mean((X - d)**2)
    snr = 10 * np.log10(np.mean(X**2) / (mse + 1e-12))
    
    print(f"🎯 SNR:      {snr:.2f} dB")
    print(f"🎯 Bitrate:  {eng.bpd():.2f} BPD")
    print(f"🎯 Time:     {t1-t0:.2f}s")
    return snr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="store_true", help="Skip model load, load from npy.")
    parser.add_argument("--stages", type=int, default=8, help="Number of E8 stages.")
    args = parser.parse_args()

    data_dir = "tests/real_kv_data"
    keys_path = os.path.join(data_dir, "keys.npy")

    if args.direct and os.path.exists(keys_path):
        print(f"Loading cached KV files from {keys_path}...")
        keys_arr = np.load(keys_path)
    else:
        keys_arr, _ = generate_local_kv_data(data_dir)

    run_benchmark(keys_arr, stages=args.stages)
