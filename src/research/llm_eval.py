"""End-to-end autoregressive evaluation with quantized KV caches."""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from src.core.v16 import Final_God_V16
    from src.core.v17 import LatticeRSN_V17
    from src.core.v18 import LatticeRSN_V18
    from src.core.v19 import LatticeRSN_V19
else:
    from ..core.v16 import Final_God_V16
    from ..core.v17 import LatticeRSN_V17
    from ..core.v18 import LatticeRSN_V18
    from ..core.v19 import LatticeRSN_V19


SAMPLE_TEXT = (
    "Lattice-based quantization uses the geometry of high-dimensional spaces to "
    "reduce memory while preserving useful structure. The E8 lattice is especially "
    "efficient for 8-dimensional blocks, which makes it a natural candidate for KV "
    "cache compression in transformer models. Accurate end-to-end evaluation requires "
    "feeding the quantized cache back into the model during autoregressive decoding, "
    "because small reconstruction errors can accumulate across time. "
) * 12


@dataclass
class EvalResult:
    ppl: float
    mean_nll: float
    tokens_evaluated: int
    top1_predictions: list[int]
    elapsed_s: float


def build_engine(engine_name: str, dim: int, target_bpd: float, max_stages: int):
    engine_name = engine_name.lower()
    if engine_name == "v16":
        return Final_God_V16(dim, max_stages=max_stages)
    if engine_name == "v17":
        return LatticeRSN_V17(dim, max_stages=max_stages)
    if engine_name == "v18":
        return LatticeRSN_V18(dim, target_bpd=target_bpd, max_stages=max_stages)
    if engine_name == "v19":
        return LatticeRSN_V19(dim, target_bpd=target_bpd, max_stages=max_stages)
    raise ValueError(f"Unsupported engine '{engine_name}'. Use one of: v16, v17, v18, v19.")


def load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    return SAMPLE_TEXT


def get_layer_count(cache) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache)


def get_layer_kv(cache, layer_idx: int):
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    layer = cache[layer_idx]
    return layer[0], layer[1]


def set_layer_kv(cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
    if hasattr(cache, "layers"):
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
        return cache

    layer = cache[layer_idx]
    layer_tail = tuple(layer[2:]) if len(layer) > 2 else ()
    replaced = (keys, values, *layer_tail)
    if isinstance(cache, tuple):
        cache = list(cache)
        cache[layer_idx] = replaced
        return tuple(cache)
    cache[layer_idx] = replaced
    return cache


def quantize_tensor(tensor: torch.Tensor, engine) -> torch.Tensor:
    flat = tensor.detach().cpu().numpy().reshape(-1, tensor.shape[-1]).astype(np.float64)
    codes = engine.encode(flat)
    decoded = engine.decode(codes)
    return torch.from_numpy(decoded).reshape(tensor.shape).to(device=tensor.device, dtype=tensor.dtype)


class LayerwiseKVQuantizer:
    """Fit one quantizer per layer for keys and values, then reuse it during decoding."""

    def __init__(self, engine_name: str, target_bpd: float, max_stages: int):
        self.engine_name = engine_name
        self.target_bpd = target_bpd
        self.max_stages = max_stages
        self.layer_engines: list[tuple[object, object]] = []

    def fit(self, model, calibration_ids: torch.Tensor):
        with torch.no_grad():
            outputs = model(calibration_ids, use_cache=True)

        cache = outputs.past_key_values
        self.layer_engines = []
        for layer_idx in range(get_layer_count(cache)):
            keys, values = get_layer_kv(cache, layer_idx)
            head_dim = keys.shape[-1]

            key_engine = build_engine(self.engine_name, head_dim, self.target_bpd, self.max_stages)
            value_engine = build_engine(self.engine_name, head_dim, self.target_bpd, self.max_stages)

            with contextlib.redirect_stdout(io.StringIO()):
                key_engine.fit(keys.detach().cpu().numpy().reshape(-1, head_dim).astype(np.float64))
                value_engine.fit(values.detach().cpu().numpy().reshape(-1, head_dim).astype(np.float64))

            self.layer_engines.append((key_engine, value_engine))

    def quantize_cache(self, cache):
        for layer_idx, (key_engine, value_engine) in enumerate(self.layer_engines):
            keys, values = get_layer_kv(cache, layer_idx)
            quant_keys = quantize_tensor(keys, key_engine)
            quant_values = quantize_tensor(values, value_engine)
            cache = set_layer_kv(cache, layer_idx, quant_keys, quant_values)
        return cache


def evaluate_teacher_forced_stream(
    model,
    input_ids: torch.Tensor,
    quantizer: LayerwiseKVQuantizer | None = None,
) -> EvalResult:
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least two tokens for perplexity evaluation.")

    current_input = input_ids[:, :1]
    past = None
    nll_values = []
    top1_predictions = []

    start = time.time()
    with torch.no_grad():
        for position in range(1, input_ids.shape[1]):
            outputs = model(
                input_ids=current_input,
                past_key_values=past,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            target = input_ids[:, position]
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(1, target.unsqueeze(-1)).squeeze(-1)

            nll_values.append(nll)
            top1_predictions.append(int(torch.argmax(logits, dim=-1).item()))

            past = outputs.past_key_values
            if quantizer is not None:
                past = quantizer.quantize_cache(past)

            current_input = input_ids[:, position : position + 1]

    elapsed_s = time.time() - start
    stacked_nll = torch.stack(nll_values, dim=1)
    mean_nll = float(stacked_nll.mean().item())
    ppl = float(math.exp(mean_nll))

    return EvalResult(
        ppl=ppl,
        mean_nll=mean_nll,
        tokens_evaluated=input_ids.shape[1] - 1,
        top1_predictions=top1_predictions,
        elapsed_s=elapsed_s,
    )


def evaluate_perplexity(
    model_id: str,
    dataset_text: str,
    device: str = "cpu",
    engine_name: str = "v19",
    target_bpd: float = 2.0,
    max_stages: int = 6,
    max_length: int = 256,
    calibration_length: int = 64,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    inputs = tokenizer(dataset_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = inputs.input_ids
    calibration_length = max(2, min(calibration_length, input_ids.shape[1]))

    print(f"Model: {model_id}")
    print(f"Tokens: {input_ids.shape[1]}")
    print(f"Evaluator: engine={engine_name}, target_bpd={target_bpd}, max_stages={max_stages}")

    with torch.no_grad():
        full_outputs = model(input_ids, labels=input_ids)
    baseline_full_loss = float(full_outputs.loss.item())
    baseline_full_ppl = float(math.exp(baseline_full_loss))
    print(f"Baseline Full-Forward PPL: {baseline_full_ppl:.6f}")

    baseline_stream = evaluate_teacher_forced_stream(model, input_ids)
    print(f"Baseline Streaming PPL:    {baseline_stream.ppl:.6f}")

    quantizer = LayerwiseKVQuantizer(
        engine_name=engine_name,
        target_bpd=target_bpd,
        max_stages=max_stages,
    )

    fit_start = time.time()
    quantizer.fit(model, input_ids[:, :calibration_length])
    fit_elapsed = time.time() - fit_start
    print(f"Calibration Tokens:        {calibration_length}")
    print(f"Quantizer Fit Time:        {fit_elapsed:.4f}s")

    quant_stream = evaluate_teacher_forced_stream(model, input_ids, quantizer=quantizer)
    print(f"Quantized Streaming PPL:   {quant_stream.ppl:.6f}")

    ppl_delta = quant_stream.ppl - baseline_stream.ppl
    ppl_delta_pct = (ppl_delta / baseline_stream.ppl) * 100.0
    prediction_match = (
        np.mean(np.array(baseline_stream.top1_predictions) == np.array(quant_stream.top1_predictions)) * 100.0
    )

    print(f"PPL Delta:                 {ppl_delta:+.6f}")
    print(f"PPL Delta %:               {ppl_delta_pct:+.4f}%")
    print(f"Top-1 Agreement:           {prediction_match:.2f}%")
    print(f"Baseline Stream Time:      {baseline_stream.elapsed_s:.4f}s")
    print(f"Quantized Stream Time:     {quant_stream.elapsed_s:.4f}s")

    return {
        "model": model_id,
        "tokens": int(input_ids.shape[1]),
        "baseline_full_ppl": baseline_full_ppl,
        "baseline_stream_ppl": baseline_stream.ppl,
        "quant_stream_ppl": quant_stream.ppl,
        "ppl_delta": ppl_delta,
        "ppl_delta_pct": ppl_delta_pct,
        "top1_agreement_pct": prediction_match,
        "fit_time_s": fit_elapsed,
        "baseline_stream_time_s": baseline_stream.elapsed_s,
        "quant_stream_time_s": quant_stream.elapsed_s,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end quantized KV cache evaluation.")
    parser.add_argument("--model", default="openai-community/gpt2", help="Causal LM to evaluate.")
    parser.add_argument("--device", default="cpu", help="Evaluation device, e.g. cpu or cuda.")
    parser.add_argument("--engine", default="v19", choices=["v16", "v17", "v18", "v19"])
    parser.add_argument("--target-bpd", type=float, default=2.0, help="Target bitrate for V18/V19.")
    parser.add_argument("--max-stages", type=int, default=6, help="Maximum refinement stages.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum number of tokens to evaluate.")
    parser.add_argument(
        "--calibration-length",
        type=int,
        default=64,
        help="How many initial tokens to use for fitting the per-layer quantizers.",
    )
    parser.add_argument("--text", help="Inline evaluation text.")
    parser.add_argument("--text-file", help="Path to a UTF-8 text file for evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text = load_text(args)
    evaluate_perplexity(
        model_id=args.model,
        dataset_text=text,
        device=args.device,
        engine_name=args.engine,
        target_bpd=args.target_bpd,
        max_stages=args.max_stages,
        max_length=args.max_length,
        calibration_length=args.calibration_length,
    )
