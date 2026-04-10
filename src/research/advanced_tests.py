"""
Advanced AI Retest (God-Mode PPL Eval)
=====================================
Comprehensive Perplexity Analysis across V16, V17, and V18.
Testing 3 models on scientific-sample text.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..core.v16 import Final_God_V16
from ..core.v17 import LatticeRSN_V17
from ..core.v18 import LatticeRSN_V18
from ..core.v19 import LatticeRSN_V19
import time

def evaluate_ppl_tier(model_id, text, version, tier, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        # Baseline
        outputs_orig = model(input_ids, labels=input_ids, output_hidden_states=True)
        base_ppl = np.exp(outputs_orig.loss.item())
        
        # We simulate KV-cache/Activation quantization on the last hidden state before LM Head
        hidden_states = outputs_orig.hidden_states[-1]
        orig_shape = hidden_states.shape
        flattened = hidden_states.view(-1, orig_shape[-1]).cpu().numpy()
        X64 = flattened.astype(np.float64)
        
        dim = orig_shape[-1]
        
        # Engine Config
        if version == "V16":
            stages = {"Extreme": 1, "Medium": 2, "Loose": 4}[tier]
            eng = Final_God_V16(dim, max_stages=stages)
        elif version == "V17":
            stages = {"Extreme": 1, "Medium": 3, "Loose": 8}[tier]
            eng = LatticeRSN_V17(dim, max_stages=stages)
        elif version == "V18":
            target = {"Extreme": 2.2, "Medium": 4.5, "Loose": 8.0}[tier]
            eng = LatticeRSN_V18(dim, target_bpd=target, max_stages=8)
        elif version == "V19":
            target = {"Extreme": 2.99, "Medium": 4.5, "Loose": 8.0}[tier]
            eng = LatticeRSN_V19(dim, target_bpd=target, max_stages=8)
            
        eng.fit(flattened)
        co = eng.encode(flattened)
        dequantized = eng.decode(co)
        bpd = eng.measure_efficiency(co)
        
        # Measure Signal Integrity
        mse = np.mean((X64 - dequantized.astype(np.float64))**2)
        snr = 10 * np.log10(np.mean(X64**2) / max(mse, 1e-30))
        
        # Measure PPL Impact
        hidden_states_q = torch.from_numpy(dequantized).view(orig_shape).to(device).to(model.dtype)
        lm_head = model.get_output_embeddings()
        lm_logits = lm_head(hidden_states_q)
        
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss_q = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        q_ppl = np.exp(loss_q.item())
        
    return {
        "Model": model_id,
        "Version": version,
        "Tier": tier,
        "BPD": bpd,
        "SNR": snr,
        "Base PPL": base_ppl,
        "Quant PPL": q_ppl,
        "PPL Delta": q_ppl - base_ppl
    }

def run_advanced_benchmark():
    models = ["gpt2", "facebook/opt-125m", "EleutherAI/pythia-160m"]
    
    # Ultimate Diversity Sweep (10 Categories)
    prompts = [
        "Lattice-RSN compression relies on the geometric stability of the E8 Gosset lattice.", # Technical
        "The silicon mind dreamt of 8-dimensional lattices, stretching across the digital horizon.", # Creative
        "def optimize_cache(tensor, target_bpd=3.0): \"\"\"Lattice-RSN Singularity Core\"\"\" return q", # Code
        "If we compress the KV-cache too much, does the model lose its memory of the context?", # Conversational
        "A quick brown fox jumped over the lazy dog while the E8 lattice calculated its packing.", # Standard
        "Hola, ¿cómo estás? El sistema de cuantización de Lattice es muy eficiente.", # Spanish
        "Let P be a prime number. Then the integers modulo P form a field. Algebra is key.", # Math/Logic
        "What is the nature of consciousness? Is it emergent property of processing?", # Philosophy
        "gh!!678234-==--=  asdf qwer zxcv. . . . . . . .. . . . . . . . .", # Glitch/Noise
        "The quick red fox jumped over the lazy sleeping cat under the bright yellow sun." # Variation
    ]
    prompt_text = " ".join(prompts) * 3
    
    results = []
    versions = ["V17", "V19"] # Focus on the kings
    tiers = ["Extreme", "Medium"]
    
    print(f"--- Starting ULTIMATE Diversity AI Retest (10+ Domains) ---")
    
    for m in models:
        for v in versions:
            for t in tiers:
                try:
                    print(f"Benchmarking {m} | {v} | {t}...")
                    res = evaluate_ppl_tier(m, prompt_text, v, t)
                    results.append(res)
                except Exception as e:
                    print(f"Error evaluating {m} {v} {t}: {e}")

    df = pd.DataFrame(results)
    print("\n--- GENERALIZED AI PERFORMANCE MATRIX ---")
    print(df.to_string(index=False))
    
    df.to_json("advanced_results.json", orient="records")
    return df

if __name__ == "__main__":
    run_advanced_benchmark()
