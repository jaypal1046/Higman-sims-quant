from __future__ import annotations

import argparse
import json
import math
import os
import runpy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

try:
    from huggingface_hub.errors import GatedRepoError
except Exception:
    GatedRepoError = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_v9_namespace() -> Dict[str, Any]:
    module_path = Path(__file__).with_name("higman_sims_quant_v8_recursive.PY")
    if not module_path.exists():
        raise FileNotFoundError(f"Unable to locate V9 module: {module_path}")
    return runpy.run_path(str(module_path))


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_arg]


def format_gated_repo_help(model_id: str, tokenizer_id: str, token_supplied: bool) -> str:
    lines = [
        f"Unable to access gated Hugging Face repo: {model_id}",
        "",
        "This is not a V9 quantizer bug. The model/tokenizer download was rejected before inference started.",
        "",
        "Fix options:",
        f"1. Request/confirm access on https://huggingface.co/{model_id}",
        "2. Log in with Hugging Face CLI:",
        "   huggingface-cli login",
        "3. Or set a token in PowerShell before running:",
        "   $env:HF_TOKEN='***REMOVED***'",
        "4. Or pass --hf-token ***REMOVED***",
        f"5. Or point --model-id / --tokenizer-id to a local downloaded checkpoint directory",
        "",
        f"Model id: {model_id}",
        f"Tokenizer id: {tokenizer_id}",
        f"HF token supplied: {'yes' if token_supplied else 'no'}",
    ]
    return "\n".join(lines)


def load_tokenizer_and_model(
    model_id: str,
    tokenizer_id: str,
    hf_token: Optional[str],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Any, Any]:
    token_supplied = bool(hf_token)
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=hf_token, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=hf_token, use_fast=False)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            dtype=dtype,
        ).to(device)
        return tokenizer, model
    except Exception as exc:
        text = str(exc)
        is_gated = "gated repo" in text.lower() or "401" in text or (
            GatedRepoError is not None and isinstance(exc, GatedRepoError)
        )
        if is_gated:
            raise SystemExit(format_gated_repo_help(model_id, tokenizer_id, token_supplied)) from exc
        raise


def sample_rows(X: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= max_rows:
        return np.asarray(X, dtype=np.float32)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return np.asarray(X[idx], dtype=np.float32)


def flatten_kv_tensor_cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float32).cpu().reshape(-1, tensor.shape[-1]).numpy()


def unflatten_kv_tensor_cpu(flat: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
    return torch.from_numpy(np.asarray(flat, dtype=np.float32)).reshape(*shape)


def build_custom_long_token_ids(tokenizer: Any, min_tokens: int) -> List[int]:
    sections = [
        """
Section: System design notes.
The service stores long-running conversation state, asynchronous jobs, cache residency metadata,
and audit events. A request may touch multiple shards, multiple retrieval indices, and multiple
versions of the same artifact. Operators care about consistency, latency percentiles, partial
failure semantics, observability cardinality, and the shape of the residual errors after every
compression stage.
""",
        """
Section: Research memo.
Recursive vector quantizers decompose a vector into a hierarchy of coarse and fine components.
When the direction codebook is strong, later stages spend their bits cleaning up the remaining
energy instead of rediscovering the same direction. Real KV caches are anisotropic, often contain
head-specific outliers, and benefit from a calibration pass that observes the true residuals after
dequantization rather than idealized pre-quantization norms.
""",
        """
Section: Long-context retrieval story.
An analyst reviews contracts, transcripts, bug reports, code snippets, and tables spread across
dozens of pages. The answer depends on details introduced early in the context and referenced much
later. The model must preserve entity identities, chronology, exceptions, and numeric thresholds,
even after the prompt grows from a few hundred tokens to many thousands.
""",
        """
Section: Mixed-format content.
Table A: region=north, latency=118ms, retries=2, outcome=success.
Table B: region=west, latency=241ms, retries=5, outcome=timeout.
Code sample:
for step in range(total_steps):
    score += residual_norms[step] * weights[step]
Narrative:
The operator compared the stage-one approximation with the final reconstruction and documented the
largest absolute error, average cosine similarity, and the effect on the next-token loss.
""",
    ]

    header_ids = tokenizer(
        "Long-context KV cache evaluation document.\n",
        add_special_tokens=False,
    ).input_ids
    section_ids = [
        tokenizer(section, add_special_tokens=False).input_ids for section in sections
    ]

    tokens: List[int] = []
    doc_idx = 0
    while len(tokens) < min_tokens:
        heading = tokenizer(
            f"\n\n### Document {doc_idx}\n",
            add_special_tokens=False,
        ).input_ids
        body = section_ids[doc_idx % len(section_ids)]
        tokens.extend(header_ids)
        tokens.extend(heading)
        tokens.extend(body)
        doc_idx += 1
    return tokens[:min_tokens]


def build_longbench_prompt(
    tokenizer: Any,
    dataset_name: str,
    subset: str,
    split: str,
    index: int,
) -> Tuple[List[int], str]:
    if load_dataset is None:
        raise RuntimeError(
            "LongBench support requires the datasets package. Install it with `pip install datasets`."
        )
    dataset = load_dataset(dataset_name, subset, split=split)
    example = dataset[int(index)]
    context = example.get("context") or example.get("article") or example.get("input", "")
    question = example.get("question") or example.get("input", "")
    answers = example.get("answers") or example.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    answer = answers[0] if answers else ""
    prompt = (
        "You are reading a long-context benchmark example.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
    return prompt_ids + answer_ids, prompt


def load_token_ids(args: argparse.Namespace, tokenizer: Any) -> Tuple[List[int], str]:
    if args.prompt_source == "custom":
        needed = args.context_tokens + args.eval_tokens + 1
        token_ids = build_custom_long_token_ids(tokenizer, needed)
        return token_ids, "custom"

    if args.prompt_source == "longbench":
        token_ids, prompt = build_longbench_prompt(
            tokenizer=tokenizer,
            dataset_name=args.longbench_dataset,
            subset=args.longbench_subset,
            split=args.longbench_split,
            index=args.longbench_index,
        )
        return token_ids, prompt

    if args.prompt_file is None:
        raise ValueError("prompt_source=file requires --prompt-file")

    text = Path(args.prompt_file).read_text(encoding="utf-8")
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    return token_ids, text


def maybe_trim_to_model_limit(
    token_ids: Sequence[int],
    max_position_embeddings: Optional[int],
) -> List[int]:
    if max_position_embeddings is None or max_position_embeddings <= 0:
        return list(token_ids)
    if len(token_ids) <= max_position_embeddings:
        return list(token_ids)
    return list(token_ids[:max_position_embeddings])


def prefill_cache(
    model: Any,
    prefix_ids: torch.Tensor,
    chunk_size: int,
) -> Any:
    past_key_values = None
    with torch.inference_mode():
        for start in range(0, prefix_ids.shape[1], chunk_size):
            chunk = prefix_ids[:, start : start + chunk_size]
            outputs = model(
                input_ids=chunk,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
    return past_key_values


def extract_cache_layers_cpu(past_key_values: Any) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], str]:
    layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

    if isinstance(past_key_values, DynamicCache) or hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            layers.append(
                (
                    layer.keys.detach().to(torch.float32).cpu().contiguous(),
                    layer.values.detach().to(torch.float32).cpu().contiguous(),
                )
            )
        return layers, "dynamic"

    if isinstance(past_key_values, (tuple, list)):
        for key, value in past_key_values:
            layers.append(
                (
                    key.detach().to(torch.float32).cpu().contiguous(),
                    value.detach().to(torch.float32).cpu().contiguous(),
                )
            )
        return layers, "legacy"

    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")


def rebuild_cache(
    layers: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    cache_format: str,
    device: torch.device,
    dtype: torch.dtype,
    model_config: Any,
) -> Any:
    if cache_format == "dynamic":
        cache = DynamicCache(config=model_config)
        for layer_idx, (key, value) in enumerate(layers):
            cache.update(
                key.to(device=device, dtype=dtype),
                value.to(device=device, dtype=dtype),
                layer_idx,
            )
        return cache

    return tuple(
        (
            key.to(device=device, dtype=dtype),
            value.to(device=device, dtype=dtype),
        )
        for key, value in layers
    )


class MetricAccumulator:
    def __init__(self) -> None:
        self.signal_sq = 0.0
        self.error_sq = 0.0
        self.cos_sum = 0.0
        self.count = 0
        self.max_abs = 0.0

    def update(self, original: np.ndarray, reconstructed: np.ndarray) -> None:
        original = np.asarray(original, dtype=np.float32)
        reconstructed = np.asarray(reconstructed, dtype=np.float32)
        error = original - reconstructed
        self.signal_sq += float(np.sum(original * original, dtype=np.float64))
        self.error_sq += float(np.sum(error * error, dtype=np.float64))
        self.max_abs = max(self.max_abs, float(np.max(np.abs(error))))

        orig_norm = np.linalg.norm(original, axis=1)
        recon_norm = np.linalg.norm(reconstructed, axis=1)
        denom = np.maximum(orig_norm * recon_norm, 1e-12)
        self.cos_sum += float(np.sum(np.sum(original * reconstructed, axis=1) / denom, dtype=np.float64))
        self.count += int(original.shape[0])

    def summary(self) -> Dict[str, float]:
        snr = 10.0 * math.log10(self.signal_sq / (self.error_sq + 1e-12))
        cosine = self.cos_sum / max(1, self.count)
        return {
            "snr_db": snr,
            "cosine": cosine,
            "max_abs": self.max_abs,
        }


class ScalarUniformQuantizer:
    def __init__(self, dim: int, bits: int = 3) -> None:
        self.dim = dim
        self.bits = bits
        self.lo = np.zeros(dim, dtype=np.float64)
        self.scale = np.ones(dim, dtype=np.float64)

    @property
    def bits_per_dim(self) -> float:
        return float(self.bits)

    def fit(self, X: np.ndarray) -> None:
        self.lo = np.quantile(X, 0.001, axis=0).astype(np.float64)
        hi = np.quantile(X, 0.999, axis=0).astype(np.float64)
        self.scale = np.maximum(hi - self.lo, 1e-8)

    def encode(self, X: np.ndarray) -> np.ndarray:
        levels = 1 << self.bits
        return np.clip(
            np.floor((X - self.lo) / self.scale * levels),
            0,
            levels - 1,
        ).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        levels = 1 << self.bits
        return self.lo + (codes.astype(np.float64) + 0.5) / levels * self.scale

    def encode_decode(self, X: np.ndarray) -> Tuple[np.ndarray, float, float]:
        t0 = time.perf_counter()
        codes = self.encode(X)
        t1 = time.perf_counter()
        reconstructed = self.decode(codes)
        t2 = time.perf_counter()
        return reconstructed, t1 - t0, t2 - t1


class PolarStyleQuantizer:
    """
    Lightweight PolarQuant-style baseline.

    It recursively converts pairs into polar coordinates, quantizes each angle
    uniformly, quantizes the final norm in log-space, and reconstructs by
    reversing the polar tree. This is not Google TurboQuant, but it is a
    reasonable 2D-angle recursive baseline for comparison.
    """

    def __init__(self, dim: int, angle_bits: int = 3, norm_bits: int = 3, seed: int = 0) -> None:
        self.dim = dim
        self.angle_bits = angle_bits
        self.norm_bits = norm_bits
        self.pad_dim = 1 << int(math.ceil(math.log2(max(1, dim))))
        self._rng = np.random.default_rng(seed)
        self._rotation_vectors = self._build_rotation()
        self._log_norm_range = (-4.0, 4.0)

    @property
    def bits_per_dim(self) -> float:
        total_bits = (self.pad_dim - 1) * self.angle_bits + self.norm_bits
        return total_bits / self.dim

    def _build_rotation(self) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for _ in range(4):
            v = self._rng.standard_normal(self.pad_dim)
            v /= np.linalg.norm(v) + 1e-12
            vectors.append(v.astype(np.float64))
        return vectors

    def _apply_rotation(self, X: np.ndarray, reverse: bool = False) -> np.ndarray:
        Y = np.asarray(X, dtype=np.float64).copy()
        vectors = list(reversed(self._rotation_vectors)) if reverse else self._rotation_vectors
        for v in vectors:
            Y -= 2.0 * (Y @ v)[:, None] * v[None, :]
        return Y

    def _pad(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == self.pad_dim:
            return np.asarray(X, dtype=np.float64)
        return np.pad(np.asarray(X, dtype=np.float64), ((0, 0), (0, self.pad_dim - self.dim)))

    def _encode_recursive(self, X: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        angles: List[np.ndarray] = []
        radii = X
        while radii.shape[1] > 1:
            pairs = radii.reshape(radii.shape[0], -1, 2)
            theta = np.arctan2(pairs[:, :, 1], pairs[:, :, 0])
            angles.append(theta)
            radii = np.linalg.norm(pairs, axis=2)
        return angles, radii[:, 0]

    def fit(self, X: np.ndarray) -> None:
        Xp = self._apply_rotation(self._pad(X))
        _, norms = self._encode_recursive(Xp)
        log_norms = np.log(np.maximum(norms, 1e-12))
        lo = float(np.quantile(log_norms, 0.001))
        hi = float(np.quantile(log_norms, 0.999))
        self._log_norm_range = (lo, max(hi, lo + 1e-3))

    def encode(self, X: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        Xp = self._apply_rotation(self._pad(X))
        angles, norms = self._encode_recursive(Xp)

        angle_codes: List[np.ndarray] = []
        levels_angle = 1 << self.angle_bits
        for theta in angles:
            codes = np.clip(
                np.floor((theta + math.pi) / (2.0 * math.pi) * levels_angle),
                0,
                levels_angle - 1,
            ).astype(np.uint8)
            angle_codes.append(codes)

        log_lo, log_hi = self._log_norm_range
        levels_norm = 1 << self.norm_bits
        scaled = np.clip((np.log(np.maximum(norms, 1e-12)) - log_lo) / (log_hi - log_lo + 1e-12), 0.0, 1.0 - 1e-12)
        norm_codes = np.floor(scaled * levels_norm).astype(np.uint8)
        return angle_codes, norm_codes

    def decode(self, encoded: Tuple[List[np.ndarray], np.ndarray]) -> np.ndarray:
        angle_codes, norm_codes = encoded
        log_lo, log_hi = self._log_norm_range
        levels_norm = 1 << self.norm_bits
        log_norms = log_lo + (norm_codes.astype(np.float64) + 0.5) / levels_norm * (log_hi - log_lo)
        radii = np.exp(log_norms)[:, None]

        levels_angle = 1 << self.angle_bits
        for codes in reversed(angle_codes):
            theta = -math.pi + (codes.astype(np.float64) + 0.5) / levels_angle * (2.0 * math.pi)
            x = radii * np.cos(theta)
            y = radii * np.sin(theta)
            radii = np.stack([x, y], axis=-1).reshape(codes.shape[0], -1)

        Xp = radii[:, : self.pad_dim]
        X_hat = self._apply_rotation(Xp, reverse=True)
        return X_hat[:, : self.dim]

    def encode_decode(self, X: np.ndarray) -> Tuple[np.ndarray, float, float]:
        t0 = time.perf_counter()
        encoded = self.encode(X)
        t1 = time.perf_counter()
        reconstructed = self.decode(encoded)
        t2 = time.perf_counter()
        return reconstructed, t1 - t0, t2 - t1


class V9LayerQuantizer:
    def __init__(self, namespace: Dict[str, Any], template_engine: Any, seed: int) -> None:
        V9Engine = namespace["V9Engine"]
        self.engine = V9Engine(
            dim=template_engine.dim,
            num_stages=template_engine.num_stages,
            norm_bits=template_engine.stage_norm_bits,
            bits_tail=template_engine.bits_tail,
            use_rotation=template_engine.use_rotation,
            rotation_reflections=template_engine.rotation_reflections,
            use_qjl=template_engine.use_qjl,
            qjl_rank=template_engine.qjl_rank,
            seed=seed,
        )

    @property
    def bits_per_dim(self) -> float:
        return float(self.engine.bits_per_dim())

    def fit(self, X: np.ndarray) -> None:
        self.engine.fit(X)

    def encode_decode(self, X: np.ndarray) -> Tuple[np.ndarray, float, float]:
        t0 = time.perf_counter()
        encoded = self.engine.encode(X)
        t1 = time.perf_counter()
        reconstructed = self.engine.decode(encoded)
        t2 = time.perf_counter()
        return reconstructed, t1 - t0, t2 - t1


def encode_decode_in_chunks(
    quantizer: Any,
    X: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, float, float]:
    recon_batches: List[np.ndarray] = []
    encode_time = 0.0
    decode_time = 0.0
    for start in range(0, len(X), batch_size):
        batch = np.asarray(X[start : start + batch_size], dtype=np.float64)
        reconstructed, t_enc, t_dec = quantizer.encode_decode(batch)
        recon_batches.append(np.asarray(reconstructed, dtype=np.float32))
        encode_time += t_enc
        decode_time += t_dec
    return np.concatenate(recon_batches, axis=0), encode_time, decode_time


@dataclass
class MethodResult:
    name: str
    effective_bpd: float
    key_metrics: Dict[str, float]
    value_metrics: Dict[str, float]
    encode_ms: float
    decode_ms: float
    perplexity: float
    delta_ppl_pct: float
    notes: str


def describe_v9_template(template: Any) -> str:
    return (
        f"stages={template.num_stages}, "
        f"norm_bits={tuple(round(float(v), 4) for v in template.stage_norm_bits)}, "
        f"rotation={template.use_rotation}, "
        f"qjl_rank={template.qjl_rank if template.use_qjl else 0}"
    )


def find_v9_templates(
    namespace: Dict[str, Any],
    grouped_samples: Dict[Tuple[str, int], np.ndarray],
    target_bpd: float,
    max_stages: int,
    seed: int,
) -> Dict[Tuple[str, int], Any]:
    search_target_engine = namespace["_search_target_engine"]
    templates: Dict[Tuple[str, int], Any] = {}
    for (kind, dim), sample in grouped_samples.items():
        if len(sample) < 64:
            X_tr = sample
            X_cal = sample
        else:
            split = max(64, len(sample) // 2)
            X_tr = sample[:split]
            X_cal = sample[split:] if len(sample) > split else sample[:split]
        templates[(kind, dim)] = search_target_engine(
            dim=dim,
            X_tr=np.asarray(X_tr, dtype=np.float64),
            X_cal=np.asarray(X_cal, dtype=np.float64),
            target_bpd=target_bpd,
            max_stages=max_stages,
            seed=seed + dim + (0 if kind == "key" else 10_000),
        )
    return templates


def evaluate_continuation_perplexity(
    model: Any,
    past_key_values: Any,
    bridge_token: torch.Tensor,
    continuation_tokens: torch.Tensor,
) -> float:
    losses: List[float] = []
    with torch.inference_mode():
        outputs = model(
            input_ids=bridge_token.view(1, 1),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        losses.append(F.cross_entropy(logits, continuation_tokens[:1], reduction="sum").item())
        past_key_values = outputs.past_key_values

        for idx in range(len(continuation_tokens) - 1):
            outputs = model(
                input_ids=continuation_tokens[idx : idx + 1].view(1, 1),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            losses.append(
                F.cross_entropy(logits, continuation_tokens[idx + 1 : idx + 2], reduction="sum").item()
            )
            past_key_values = outputs.past_key_values

    mean_nll = float(sum(losses) / max(1, len(losses)))
    return float(math.exp(min(mean_nll, 20.0)))


def maybe_plot_projection(
    plot_path: Optional[str],
    vectors: Dict[str, np.ndarray],
    cloud: np.ndarray,
) -> None:
    if plot_path is None:
        return
    if plt is None:
        raise RuntimeError("Plotting requested but matplotlib is not installed.")

    cloud = np.asarray(cloud, dtype=np.float64)
    center = cloud.mean(axis=0, keepdims=True)
    centered = cloud - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    proj_cloud = centered @ basis

    plt.figure(figsize=(7, 6))
    plt.scatter(proj_cloud[:, 0], proj_cloud[:, 1], s=10, alpha=0.18, color="gray", label="sample cloud")

    for name, vector in vectors.items():
        point = (np.asarray(vector, dtype=np.float64) - center.ravel()) @ basis
        plt.scatter(point[0], point[1], s=120, label=name)

    plt.title("KV vector compression in local PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def markdown_table(rows: Sequence[MethodResult]) -> str:
    header = (
        "| Method | Eff. bpd | K SNR (dB) | K Cosine | V SNR (dB) | V Cosine | "
        "Max |err| | Encode ms | Decode ms | PPL | Delta PPL % | Notes |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                [
                    row.name,
                    f"{row.effective_bpd:.3f}",
                    f"{row.key_metrics['snr_db']:.2f}",
                    f"{row.key_metrics['cosine']:.5f}",
                    f"{row.value_metrics['snr_db']:.2f}",
                    f"{row.value_metrics['cosine']:.5f}",
                    f"{max(row.key_metrics['max_abs'], row.value_metrics['max_abs']):.4e}",
                    f"{row.encode_ms:.2f}",
                    f"{row.decode_ms:.2f}",
                    f"{row.perplexity:.4f}",
                    f"{row.delta_ppl_pct:+.3f}",
                    row.notes,
                ]
            )
            + " |"
        )
    return "\n".join([header, sep, *body])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Higman-Sims V9 on real LLM KV caches.")
    parser.add_argument("--model-id", default="google/gemma-2-9b-it")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--prompt-source", choices=["custom", "longbench", "file"], default="custom")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--longbench-dataset", default="THUDM/LongBench")
    parser.add_argument("--longbench-subset", default="qasper")
    parser.add_argument("--longbench-split", default="test")
    parser.add_argument("--longbench-index", type=int, default=0)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--eval-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--fit-vectors-per-layer", type=int, default=2048)
    parser.add_argument("--search-pool-per-layer", type=int, default=256)
    parser.add_argument("--quant-batch-size", type=int, default=16384)
    parser.add_argument("--target-bpd", type=float, default=3.0)
    parser.add_argument("--max-stages", type=int, default=4)
    parser.add_argument("--include-polar-style", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--plot-layer", type=int, default=0)
    parser.add_argument("--plot-kind", choices=["key", "value"], default="key")
    parser.add_argument("--plot-vector-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    namespace = load_v9_namespace()

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    print(f"Loading model {args.model_id} on {device} with dtype={dtype}")

    tokenizer_id = args.tokenizer_id or args.model_id
    tokenizer, model = load_tokenizer_and_model(
        model_id=args.model_id,
        tokenizer_id=tokenizer_id,
        hf_token=args.hf_token,
        dtype=dtype,
        device=device,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    token_ids, prompt_descriptor = load_token_ids(args, tokenizer)
    token_ids = maybe_trim_to_model_limit(
        token_ids,
        getattr(model.config, "max_position_embeddings", None),
    )

    min_needed = args.context_tokens + args.eval_tokens + 1
    if len(token_ids) < min_needed:
        raise RuntimeError(
            f"Need at least {min_needed} tokens but only found {len(token_ids)}. "
            "Use a longer prompt, LongBench input, or lower --context-tokens."
        )

    prefix_cache_ids = token_ids[: args.context_tokens - 1]
    bridge_token_id = token_ids[args.context_tokens - 1]
    continuation_ids = token_ids[args.context_tokens : args.context_tokens + args.eval_tokens]

    prefix_tensor = torch.tensor(prefix_cache_ids, dtype=torch.long, device=device).unsqueeze(0)
    bridge_token = torch.tensor([bridge_token_id], dtype=torch.long, device=device)
    continuation_tokens = torch.tensor(continuation_ids, dtype=torch.long, device=device)

    print(
        f"Prefilling {len(prefix_cache_ids)} tokens from {args.prompt_source} prompt "
        f"({prompt_descriptor if isinstance(prompt_descriptor, str) else 'custom'})"
    )
    start_prefill = time.perf_counter()
    past_key_values = prefill_cache(model, prefix_tensor, args.prefill_chunk_size)
    prefill_time = time.perf_counter() - start_prefill
    print(f"Prefill complete in {prefill_time:.2f}s")

    original_layers, cache_format = extract_cache_layers_cpu(past_key_values)
    del past_key_values

    print(f"Captured {len(original_layers)} layers of KV cache tensors")

    baseline_cache = rebuild_cache(original_layers, cache_format, device, dtype, model.config)
    fp_ppl = evaluate_continuation_perplexity(model, baseline_cache, bridge_token, continuation_tokens)

    grouped_search_samples: Dict[Tuple[str, int], List[np.ndarray]] = {}
    per_layer_fit_samples: Dict[Tuple[int, str], np.ndarray] = {}
    plot_cloud: Optional[np.ndarray] = None
    plot_original: Optional[np.ndarray] = None

    for layer_idx, (key_cpu, value_cpu) in enumerate(original_layers):
        for kind, tensor_cpu in (("key", key_cpu), ("value", value_cpu)):
            flat = flatten_kv_tensor_cpu(tensor_cpu)
            sample_fit = sample_rows(flat, args.fit_vectors_per_layer, rng)
            sample_search = sample_rows(flat, args.search_pool_per_layer, rng)
            per_layer_fit_samples[(layer_idx, kind)] = sample_fit
            grouped_search_samples.setdefault((kind, flat.shape[1]), []).append(sample_search)

            if layer_idx == args.plot_layer and kind == args.plot_kind:
                plot_cloud = sample_rows(flat, min(1024, len(flat)), rng)
                plot_original = np.asarray(flat[min(args.plot_vector_index, len(flat) - 1)], dtype=np.float64)

    pooled_search_samples = {
        key: np.concatenate(value, axis=0) for key, value in grouped_search_samples.items()
    }

    print("Searching exact-rate V9 configurations by KV kind/head_dim")
    v9_templates = find_v9_templates(
        namespace=namespace,
        grouped_samples=pooled_search_samples,
        target_bpd=args.target_bpd,
        max_stages=args.max_stages,
        seed=args.seed,
    )
    for (kind, dim), template in sorted(v9_templates.items()):
        print(f"  V9 template for {kind}/{dim}D -> {describe_v9_template(template)}")

    methods: List[Tuple[str, Dict[Tuple[int, str], Any], str]] = []

    scalar_quantizers: Dict[Tuple[int, str], ScalarUniformQuantizer] = {}
    for layer_idx, (key_cpu, value_cpu) in enumerate(original_layers):
        for kind, tensor_cpu in (("key", key_cpu), ("value", value_cpu)):
            quantizer = ScalarUniformQuantizer(dim=tensor_cpu.shape[-1], bits=3)
            quantizer.fit(per_layer_fit_samples[(layer_idx, kind)])
            scalar_quantizers[(layer_idx, kind)] = quantizer
    methods.append(("Scalar 3-bit", scalar_quantizers, "Per-dimension uniform 3-bit"))

    v9_quantizers: Dict[Tuple[int, str], V9LayerQuantizer] = {}
    for layer_idx, (key_cpu, value_cpu) in enumerate(original_layers):
        for kind, tensor_cpu in (("key", key_cpu), ("value", value_cpu)):
            template = v9_templates[(kind, tensor_cpu.shape[-1])]
            quantizer = V9LayerQuantizer(namespace, template, seed=args.seed + layer_idx + (0 if kind == "key" else 1000))
            quantizer.fit(per_layer_fit_samples[(layer_idx, kind)])
            v9_quantizers[(layer_idx, kind)] = quantizer
    methods.append(("V9 exact 3.0 bpd", v9_quantizers, "Recursive E8 at searched exact-rate config"))

    if args.include_polar_style:
        polar_quantizers: Dict[Tuple[int, str], PolarStyleQuantizer] = {}
        for layer_idx, (key_cpu, value_cpu) in enumerate(original_layers):
            for kind, tensor_cpu in (("key", key_cpu), ("value", value_cpu)):
                quantizer = PolarStyleQuantizer(dim=tensor_cpu.shape[-1], angle_bits=3, norm_bits=3, seed=args.seed + 7 * layer_idx)
                quantizer.fit(per_layer_fit_samples[(layer_idx, kind)])
                polar_quantizers[(layer_idx, kind)] = quantizer
        methods.append(("Polar-style", polar_quantizers, "Recursive 2D-angle baseline"))

    plot_vectors: Dict[str, np.ndarray] = {}
    if plot_original is not None:
        plot_vectors["original"] = plot_original

    results: List[MethodResult] = [
        MethodResult(
            name="FP cache",
            effective_bpd=float("nan"),
            key_metrics={"snr_db": float("inf"), "cosine": 1.0, "max_abs": 0.0},
            value_metrics={"snr_db": float("inf"), "cosine": 1.0, "max_abs": 0.0},
            encode_ms=0.0,
            decode_ms=0.0,
            perplexity=fp_ppl,
            delta_ppl_pct=0.0,
            notes="Uncompressed reference",
        )
    ]

    for method_name, quantizers, note in methods:
        print(f"Compressing full prefix cache with {method_name}")
        key_metrics = MetricAccumulator()
        value_metrics = MetricAccumulator()
        reconstructed_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        encode_total = 0.0
        decode_total = 0.0

        for layer_idx, (key_cpu, value_cpu) in enumerate(original_layers):
            reconstructed_pair: List[torch.Tensor] = []
            for kind, tensor_cpu in (("key", key_cpu), ("value", value_cpu)):
                flat = flatten_kv_tensor_cpu(tensor_cpu)
                recon_flat, t_enc, t_dec = encode_decode_in_chunks(
                    quantizers[(layer_idx, kind)],
                    flat,
                    args.quant_batch_size,
                )
                encode_total += t_enc
                decode_total += t_dec

                if kind == "key":
                    key_metrics.update(flat, recon_flat)
                else:
                    value_metrics.update(flat, recon_flat)

                if layer_idx == args.plot_layer and kind == args.plot_kind and plot_original is not None:
                    plot_vectors[method_name] = np.asarray(
                        recon_flat[min(args.plot_vector_index, len(recon_flat) - 1)],
                        dtype=np.float64,
                    )

                reconstructed_pair.append(unflatten_kv_tensor_cpu(recon_flat, tensor_cpu.shape))

            reconstructed_layers.append((reconstructed_pair[0], reconstructed_pair[1]))

        quantized_cache = rebuild_cache(reconstructed_layers, cache_format, device, dtype, model.config)
        ppl = evaluate_continuation_perplexity(model, quantized_cache, bridge_token, continuation_tokens)
        delta_pct = 100.0 * (ppl - fp_ppl) / max(fp_ppl, 1e-12)

        eff_bpd = float(np.mean([q.bits_per_dim for q in quantizers.values()]))
        results.append(
            MethodResult(
                name=method_name,
                effective_bpd=eff_bpd,
                key_metrics=key_metrics.summary(),
                value_metrics=value_metrics.summary(),
                encode_ms=encode_total * 1000.0,
                decode_ms=decode_total * 1000.0,
                perplexity=ppl,
                delta_ppl_pct=delta_pct,
                notes=note,
            )
        )

    table = markdown_table(results)
    print()
    print(table)
    print()

    v9_result = next((row for row in results if row.name.startswith("V9")), None)
    if v9_result is not None:
        near_zero = abs(v9_result.delta_ppl_pct) <= 1.0
        summary = (
            "V9 maintains near-zero downstream loss on this KV snapshot."
            if near_zero
            else "V9 introduces a measurable downstream loss on this KV snapshot."
        )
        print(summary)
        print(
            f"V9 delta perplexity: {v9_result.delta_ppl_pct:+.3f}% "
            f"(K SNR={v9_result.key_metrics['snr_db']:.2f} dB, "
            f"V SNR={v9_result.value_metrics['snr_db']:.2f} dB)"
        )

    if args.markdown_out:
        Path(args.markdown_out).write_text(table + "\n", encoding="utf-8")
    if args.json_out:
        json_payload = {
            "model_id": args.model_id,
            "device": str(device),
            "dtype": str(dtype),
            "prefill_seconds": prefill_time,
            "prompt_source": args.prompt_source,
            "context_tokens": args.context_tokens,
            "eval_tokens": args.eval_tokens,
            "results": [row.__dict__ for row in results],
        }
        Path(args.json_out).write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    if args.plot_path and plot_cloud is not None and plot_vectors:
        maybe_plot_projection(args.plot_path, plot_vectors, plot_cloud)
        print(f"Saved PCA projection plot to {args.plot_path}")


if __name__ == "__main__":
    main()
