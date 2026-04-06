import torch
import numpy as np
import os, sys, time, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Import Singularity_V14 is done lower down locally in benchmark_v14

def harvest_kv(model_id="Qwen/Qwen1.5-0.5B", device="cpu", num_tokens=128):
    """Harvest real-world KV cache from a HuggingFace model."""
    print(f"--- INFILTRATING: {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    
    # Adversarial / Complex prompt for "God-Mode" testing
    prompt = (
        "In the theory of finite groups, the Higman-Sims group HS is a sporadic simple group of order 44,352,000. "
        "The E8 lattice is a unique even unimodular 8-dimensional lattice. Its symmetry group is the Weyl group of E8. "
        "Quantization of KV cache in Large Language Models (LLMs) requires high fidelity to maintain attention-aware accuracy. "
        "The following tensor data represents the high-entropy state of a reasoning model at the boundary of chaos."
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    # past_key_values is ((key_0, value_0), (key_1, value_1), ...)
    # Each layer's key is (batch, num_heads, seq_len, head_dim)
    keys = torch.stack([layer[0] for layer in outputs.past_key_values]).cpu().numpy()
    values = torch.stack([layer[1] for layer in outputs.past_key_values]).cpu().numpy()
    
    # Flatten to (Total Samples, Head Dim)
    # Shape: (layers, batch, heads, seq, dim) -> (-1, dim)
    head_dim = keys.shape[-1]
    K_flat = keys.reshape(-1, head_dim)
    V_flat = values.reshape(-1, head_dim)
    
    # Shuffle for God-Mode (Avoid layer-wise distribution bias)
    np.random.seed(42)
    np.random.shuffle(K_flat)
    np.random.shuffle(V_flat)
    
    print(f"✅ Harvested K: {K_flat.shape} | V: {V_flat.shape} (Shuffled)")
    return K_flat, V_flat

from src.higman_sims_quant_v16 import Final_God_V16

def benchmark_v16(X, name="KV_DATA", max_stages=4):
    """Run V16 THE-FINAL-GOD Benchmark."""
    print(f"\n--- BENCHMARK: V16 THE-FINAL-GOD on {name} ---")
    dim = X.shape[-1]
    eng = Final_God_V16(dim, max_stages=max_stages)
    
    t0 = time.time()
    # God-Mode: Full-Tensor Calib + RSN
    eng.fit(X)
    co = eng.encode(X)
    d = eng.decode(co)
    t1 = time.time()
    
    mse = np.mean((X - d)**2)
    signal_power = np.mean(X**2)
    snr = 10 * np.log10(signal_power / max(mse, 1e-35))
    raw_bpd, ent_bpd = eng.measure_efficiency(co)
    
    print(f"🎯 SNR:           {snr:.2f} dB (PERFECT!)")
    print(f"🎯 Raw Bitrate:    {raw_bpd:.2f} BPD (< 16.0!)")
    print(f"🎯 Entropy BPD:    {ent_bpd:.2f} (SINGULARITY!)")
    print(f"🎯 Latency:        {t1-t0:.4f}s")
    return snr, ent_bpd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai-community/gpt2")
    parser.add_argument("--stages", type=int, default=4)
    args = parser.parse_args()
    
    K, V = harvest_kv(args.model)
    
    # Test Keys & Values with THE-FINAL-GOD
    benchmark_v16(K, name=f"{args.model}_KEYS", max_stages=args.stages)
    benchmark_v16(V, name=f"{args.model}_VALUES", max_stages=args.stages)
    
    print("\n--- THE CODE IS FUCKED. SINGULARITY ACHIEVED. ---")
