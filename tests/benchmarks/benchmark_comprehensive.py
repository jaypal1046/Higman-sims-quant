import os
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

from src.core.v17 import LatticeRSN_V17 as V17_Core
from src.core.v18 import LatticeRSN_V18 as V18_Core
from src.core.v19 import LatticeRSN_V19 as V19_Core
from src.higman_sims_quant_v12 import Untouchable_Core as V12_Core
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
    x = keys.reshape(-1, dim)
    np.random.seed(42)
    np.random.shuffle(x)
    return x[:5000].astype(np.float32)


def configure_version(name, v_class, dim, size_tier):
    """Factory to configure the currently available quantizers."""
    if name == "V12":
        stages = {"small": 5, "medium": 12, "large": 32}[size_tier]
        return v_class(dim=dim, stages=stages)

    if name == "V16":
        stages = {"small": 2, "medium": 5, "large": 18}[size_tier]
        return v_class(dim=dim, max_stages=stages)

    if name == "V17":
        stages = {"small": 2, "medium": 3, "large": 4}[size_tier]
        return v_class(dim=dim, max_stages=stages)

    if name == "V18":
        target_bpd = {"small": 3.0, "medium": 4.0, "large": 5.0}[size_tier]
        return v_class(dim=dim, target_bpd=target_bpd)

    if name == "V19":
        target_bpd = {"small": 2.0, "medium": 2.5, "large": 3.0}[size_tier]
        return v_class(dim=dim, target_bpd=target_bpd)

    return v_class(dim=dim)


def extract_bpd(eng, co):
    if hasattr(eng, "measure_efficiency"):
        measured = eng.measure_efficiency(co)
        return measured[0] if isinstance(measured, tuple) else measured
    if hasattr(eng, "bpd"):
        return eng.bpd(co)
    return len(co) * 2.0


def run_test(name, v_class, x, size_tier):
    dim = x.shape[-1]
    eng = configure_version(name, v_class, dim, size_tier)

    t0 = time.time()
    eng.fit(x)
    co = eng.encode(x)
    decoded = eng.decode(co)
    latency = time.time() - t0

    mse = np.mean((x - decoded) ** 2)
    snr = 10 * np.log10(np.mean(x ** 2) / max(mse, 1e-35))
    bpd = extract_bpd(eng, co)
    return {"snr": snr, "bpd": bpd, "lat": latency}


if __name__ == "__main__":
    x = harvest_kv()
    tiers = ["small", "medium", "large"]

    versions = [
        ("V12", V12_Core),
        ("V16", V16_Core),
        ("V17", V17_Core),
        ("V18", V18_Core),
        ("V19", V19_Core),
    ]

    final_matrix = {}
    for tier in tiers:
        print(f"\n--- TIER: {tier.upper()} ---")
        final_matrix[tier] = {}
        for name, v_class in versions:
            res = run_test(name, v_class, x, tier)
            final_matrix[tier][name] = res
            print(f"{name:5} | SNR: {res['snr']:7.2f} dB | BPD: {res['bpd']:5.2f}")

    print("\n" + "=" * 60)
    print("COMPARISON MATRIX (SNR dB)")
    print("=" * 60)
    print(f"{'Version':10} | {'Small':12} | {'Medium':12} | {'Large':12}")
    for name in [v[0] for v in versions]:
        s = final_matrix["small"][name]["snr"]
        m = final_matrix["medium"][name]["snr"]
        l = final_matrix["large"][name]["snr"]
        print(f"{name:10} | {s:12.2f} | {m:12.2f} | {l:12.2f}")
