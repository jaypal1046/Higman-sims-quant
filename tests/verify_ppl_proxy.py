import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.v19 import LatticeRSN_V19


def get_kv_layer0(past_key_values):
    """Extract layer-0 KV cache across common Hugging Face cache layouts."""
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values[0][0], past_key_values[0][1]
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[0], past_key_values.value_cache[0]
    if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
        first_layer = past_key_values.layers[0]
        if hasattr(first_layer, "keys") and hasattr(first_layer, "values"):
            return first_layer.keys, first_layer.values
    if hasattr(past_key_values, "key_states") and hasattr(past_key_values, "value_states"):
        return past_key_values.key_states[0], past_key_values.value_states[0]
    return past_key_values[0][0], past_key_values[0][1]


def quantize_cache_tensor(tensor, target_bpd=2.0):
    flat = tensor.detach().cpu().numpy().reshape(-1, tensor.shape[-1])
    calib_rows = min(500, len(flat))

    engine = LatticeRSN_V19(dim=tensor.shape[-1], target_bpd=target_bpd)
    engine.fit(flat[:calib_rows])
    codes = engine.encode(flat)
    recon = engine.decode(codes)
    recon = torch.from_numpy(recon).reshape(tensor.shape).to(tensor.device).to(tensor.dtype)

    mse = torch.mean((tensor - recon) ** 2)
    snr = 10 * torch.log10(torch.mean(tensor ** 2) / mse.clamp_min(1e-30))
    return recon, snr.item()


def reshape_heads(q_tensor, num_heads, head_dim):
    batch, seq_len, _ = q_tensor.shape
    return q_tensor.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def verify_ppl_impact():
    print("--- Singularity-HS: Attention Drift Proxy (GPT-2, Layer 0) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai-community/gpt2"

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    text = (
        "The Higman-Sims group and the E8 lattice provide the mathematical foundation "
        "for optimal sphere packing in high dimensions. This property is leveraged by "
        "Singularity-HS for extreme KV cache compression."
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)

    k, v = get_kv_layer0(outputs.past_key_values)

    print("Quantizing layer-0 keys and values with V19...")
    k_recon, k_snr = quantize_cache_tensor(k, target_bpd=2.0)
    v_recon, v_snr = quantize_cache_tensor(v, target_bpd=2.0)

    print(f"Key SNR:   {k_snr:.2f} dB")
    print(f"Value SNR: {v_snr:.2f} dB")

    hidden_in = outputs.hidden_states[0]
    block = model.transformer.h[0]
    with torch.no_grad():
        qkv = block.attn.c_attn(hidden_in)

    q, _, _ = qkv.chunk(3, dim=2)
    num_heads = k.shape[1]
    head_dim = k.shape[-1]
    q = reshape_heads(q, num_heads, head_dim)
    q_last = q[:, :, -1:, :]
    scale = math.sqrt(head_dim)

    scores_orig = torch.matmul(q_last, k.transpose(-1, -2)) / scale
    scores_quant = torch.matmul(q_last, k_recon.transpose(-1, -2)) / scale
    probs_orig = torch.softmax(scores_orig, dim=-1)
    probs_quant = torch.softmax(scores_quant, dim=-1)

    ctx_orig = torch.matmul(probs_orig, v)
    ctx_quant = torch.matmul(probs_quant, v_recon)

    eps = 1e-12
    prob_kl = torch.sum(
        probs_orig
        * (
            torch.log(probs_orig.clamp_min(eps))
            - torch.log(probs_quant.clamp_min(eps))
        ),
        dim=-1,
    ).mean()
    prob_l1 = torch.mean(torch.abs(probs_orig - probs_quant))
    score_mse = torch.mean((scores_orig - scores_quant) ** 2)
    ctx_rel_error = (
        torch.linalg.vector_norm(ctx_orig - ctx_quant)
        / torch.linalg.vector_norm(ctx_orig).clamp_min(eps)
    )
    ctx_cos = F.cosine_similarity(ctx_orig.reshape(1, -1), ctx_quant.reshape(1, -1)).item()

    print("\n--- Measured Proxy Metrics ---")
    print(f"Attention score MSE:       {score_mse.item():.6e}")
    print(f"Attention prob KL:         {prob_kl.item():.6e}")
    print(f"Attention prob mean L1:    {prob_l1.item():.6e}")
    print(f"Context relative error:    {ctx_rel_error.item():.6e}")
    print(f"Context cosine similarity: {ctx_cos:.8f}")

    print("\n--- Conclusion ---")
    print("This script measures actual layer-0 attention drift.")
    print("It is still a proxy, but it is more concrete than projecting PPL from SNR alone.")


if __name__ == "__main__":
    verify_ppl_impact()
