import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from higman_sims_quant_v16 import Final_God_V16
import numpy as np
import time

def evaluate_perplexity(model_id, dataset_text, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    inputs = tokenizer(dataset_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids = inputs.input_ids

    # 1. Baseline Perplexity (FP32)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        baseline_loss = outputs.loss.item()
        baseline_ppl = np.exp(baseline_loss)

    print(f"Model: {model_id}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # 2. Quantized Perplexity
    # Hook into the attention mechanism to quantize KV cache
    # For GPT-2, we can swap the cache during generation or use a manual pass
    # Here we simulate the impact by quantizing the activations of a full forward pass
    
    # Simple simulation: Quantize the hidden states before the LM head
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        orig_shape = hidden_states.shape
        flattened = hidden_states.view(-1, orig_shape[-1]).cpu().numpy()
        
        # Quantize using Lattice-RSN
        eng = Final_God_V16(orig_shape[-1], max_stages=4)
        eng.fit(flattened[:500]) # Quick fit
        
        start_q = time.time()
        co = eng.encode(flattened)
        dequantized = eng.decode(co)
        q_time = time.time() - start_q
        
        # Inject back into model
        hidden_states_q = torch.from_numpy(dequantized).view(orig_shape).to(device).to(model.dtype)
        lm_head = model.get_output_embeddings()
        lm_logits = lm_head(hidden_states_q)
        
        # Calculate Loss
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss_q = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        q_ppl = np.exp(loss_q.item())
        
    print(f"Quantized PPL: {q_ppl:.4f}")
    print(f"PPL Delta: {q_ppl - baseline_ppl:.6f}")
    print(f"Quantization Latency: {q_time:.4f}s")
    
    return baseline_ppl, q_ppl

if __name__ == "__main__":
    # Small, diverse models for storage-constrained environments
    models = ["gpt2", "facebook/opt-125m", "EleutherAI/pythia-160m"]
    
    # Sample text from a scientific context
    sample_text = """
    Lattice-based quantization utilizes the geometric properties of high-dimensional 
    lattices to minimize distortion. The E8 Gosset lattice is particularly efficient 
    for 8-dimensional sub-blocks. When applied to Large Language Models, these 
    techniques can significantly reduce the memory footprint of the KV cache 
    without degrading the model's perplexity or reasoning capability.
    """ * 5
    
    results = []
    for model_id in models:
        try:
            print(f"\n--- Evaluating {model_id} ---")
            base, q = evaluate_perplexity(model_id, sample_text)
            results.append({"Model": model_id, "Base PPL": base, "Quant PPL": q, "Delta": q-base})
        except Exception as e:
            print(f"Failed to evaluate {model_id}: {e}")

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n--- Final Model-Agnostic Results ---")
        print(df.to_string(index=False))
