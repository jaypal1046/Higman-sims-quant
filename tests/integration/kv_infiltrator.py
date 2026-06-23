import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.higman_sims_quant_v16 import Final_God_V16


def harvest_kv(model_id="Qwen/Qwen1.5-0.5B", device="cpu", num_tokens=128):
    """Harvest real-world KV cache from a Hugging Face model."""
    print(f"--- INFILTRATING: {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

    prompt = (
        "In the theory of finite groups, the Higman-Sims group HS is a sporadic simple "
        "group of order 44,352,000. The E8 lattice is a unique even unimodular "
        "8-dimensional lattice. Its symmetry group is the Weyl group of E8. "
        "Quantization of KV cache in Large Language Models (LLMs) requires high fidelity "
        "to maintain attention-aware accuracy. The following tensor data represents the "
        "high-entropy state of a reasoning model at the boundary of chaos."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    keys = torch.stack([layer[0] for layer in outputs.past_key_values]).cpu().numpy()
    values = torch.stack([layer[1] for layer in outputs.past_key_values]).cpu().numpy()

    head_dim = keys.shape[-1]
    k_flat = keys.reshape(-1, head_dim)
    v_flat = values.reshape(-1, head_dim)

    np.random.seed(42)
    np.random.shuffle(k_flat)
    np.random.shuffle(v_flat)

    print(f"[OK] Harvested K: {k_flat.shape} | V: {v_flat.shape} (Shuffled)")
    return k_flat, v_flat


def benchmark_v16(x, name="KV_DATA", max_stages=4):
    """Run the V16 benchmark on harvested tensors."""
    print(f"\n--- BENCHMARK: V16 THE-FINAL-GOD on {name} ---")
    dim = x.shape[-1]
    eng = Final_God_V16(dim, max_stages=max_stages)

    t0 = time.time()
    eng.fit(x)
    co = eng.encode(x)
    decoded = eng.decode(co)
    t1 = time.time()

    mse = np.mean((x - decoded) ** 2)
    signal_power = np.mean(x ** 2)
    snr = 10 * np.log10(signal_power / max(mse, 1e-35))
    raw_bpd, ent_bpd = eng.measure_efficiency(co)

    print(f"SNR:         {snr:.2f} dB")
    print(f"Raw Bitrate: {raw_bpd:.2f} BPD")
    print(f"Entropy BPD: {ent_bpd:.2f}")
    print(f"Latency:     {t1 - t0:.4f}s")
    return snr, ent_bpd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai-community/gpt2")
    parser.add_argument("--stages", type=int, default=4)
    args = parser.parse_args()

    keys, values = harvest_kv(args.model)
    benchmark_v16(keys, name=f"{args.model}_KEYS", max_stages=args.stages)
    benchmark_v16(values, name=f"{args.model}_VALUES", max_stages=args.stages)

    print("\n--- KV infiltration benchmark complete. ---")
