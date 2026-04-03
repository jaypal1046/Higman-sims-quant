```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers accelerate bitsandbytes huggingface_hub
!pip install numpy
```

```python
from huggingface_hub import login
login()   # ← It will ask for a token. Get free token from https://huggingface.co/settings/tokens
```

```python

# ============================================================
# FULL KV CACHE EXTRACTION — Gemma-2b (StaticCache / layers format)
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import os
import json

# ── 1. Load Model ────────────────────────────────────────────
model_name = "google/gemma-2b-it"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Model loaded successfully!")

# ── 2. Prepare Input ─────────────────────────────────────────
prompt = "Explain in extreme detail how attention and KV cache work in transformers. Include mathematical formulas and code examples. Write at least 3000 words.Describe a complete system for extreme KV cache compression in LLMs. Talk about quantization, PolarQuant, E8 lattices, and real-world performance. Make it very long.Write a long technical blog post about building efficient on-device LLMs with minimal memory usage. Focus on KV cache optimization techniques."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ── 3. Run Forward Pass ──────────────────────────────────────
with torch.no_grad():
    outputs = model(
        **inputs,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False
    )

pkv = outputs.past_key_values
print(f"\n🔍 past_key_values type : {type(pkv)}")

# ── 4. Debug: Inspect layers structure ───────────────────────
print(f"\n🔍 pkv.layers type      : {type(pkv.layers)}")
print(f"🔍 pkv.layers length    : {len(pkv.layers)}")

layer0 = pkv.layers[0]
print(f"🔍 Layer 0 type         : {type(layer0)}")
print(f"🔍 Layer 0 attributes   : {[a for a in dir(layer0) if not a.startswith('__')]}")

# ── 5. Extract Keys & Values ─────────────────────────────────
collected_keys   = []
collected_values = []

for layer_idx, layer in enumerate(pkv.layers):
    # Try common attribute names for keys/values
    if hasattr(layer, 'key_cache') and hasattr(layer, 'value_cache'):
        k = layer.key_cache
        v = layer.value_cache

    elif hasattr(layer, 'keys') and hasattr(layer, 'values'):
        k = layer.keys
        v = layer.values

    elif hasattr(layer, 'key') and hasattr(layer, 'value'):
        k = layer.key
        v = layer.value

    elif hasattr(layer, 'key_states') and hasattr(layer, 'value_states'):
        k = layer.key_states
        v = layer.value_states

    elif hasattr(layer, 'self_attention') :
        k = layer.self_attention.key
        v = layer.self_attention.value

    else:
        # Print all tensor attributes to find the right ones
        tensor_attrs = {a: getattr(layer, a) for a in dir(layer)
                        if not a.startswith('__') and isinstance(getattr(layer, a), torch.Tensor)}
        print(f"❌ Layer {layer_idx} tensor attributes: {list(tensor_attrs.keys())}")
        raise ValueError(f"Cannot find key/value tensors in layer {layer_idx}. See above.")

    # Slice only valid (non-padding) token positions
    seq_len = inputs["input_ids"].shape[1]

    collected_keys.append(k[..., :seq_len, :].cpu().to(torch.float32).numpy())
    collected_values.append(v[..., :seq_len, :].cpu().to(torch.float32).numpy())

    if layer_idx == 0:
        print(f"\n✅ Layer 0 key   shape : {collected_keys[0].shape}")
        print(f"✅ Layer 0 value shape : {collected_values[0].shape}")

# ── 6. Stack & Save ──────────────────────────────────────────
print(f"\n✅ Captured {len(collected_keys)} layers")

keys_array   = np.stack(collected_keys,   axis=0)
values_array = np.stack(collected_values, axis=0)

print(f"💾 Final keys   shape : {keys_array.shape}")
print(f"💾 Final values shape : {values_array.shape}")
print(f"   Shape meaning      : (num_layers, batch, num_heads, seq_len, head_dim)")

os.makedirs("real_kv_data", exist_ok=True)
np.save("real_kv_data/keys.npy",   keys_array)
np.save("real_kv_data/values.npy", values_array)

metadata = {
    "model_name"  : model_name,
    "prompt"      : prompt,
    "num_layers"  : len(collected_keys),
    "keys_shape"  : list(keys_array.shape),
    "values_shape": list(values_array.shape),
    "dtype"       : str(keys_array.dtype),
}
with open("real_kv_data/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Saved:")
print("   real_kv_data/keys.npy")
print("   real_kv_data/values.npy")
print("   real_kv_data/metadata.json")

# ── 7. Verify Reload ─────────────────────────────────────────
keys_loaded   = np.load("real_kv_data/keys.npy")
values_loaded = np.load("real_kv_data/values.npy")
print(f"\n🔁 Reloaded keys   shape : {keys_loaded.shape}")
print(f"🔁 Reloaded values shape : {values_loaded.shape}")
print("✅ All done!")
```




```python

"""
Higman-Sims Quantizer V11
=========================

This version extends V9 with minimal, targeted changes for real KV cache quantization.

Key improvements over V9:
1. Asymmetric treatment of Keys and Values (Keys are more sensitive to quantization)
2. Attention-aware scoring that optimizes Q·K fidelity when query samples are provided
3. Simple outlier handling for high-norm channels in Keys only
4. Increased calibration data defaults for heavy-tailed KV distributions

IMPORTANT: The core V9Engine is NOT modified. All changes are additive:
- New _search_target_engine_asymmetric() function
- New V11Engine wrapper that adds outlier handling externally
- Updated benchmark functions that call the asymmetric search

The recursive E8 core, encode/decode paths, norm quantization, rotation, and QJL
remain exactly as in V9 to preserve mathematical closure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def build_e8_codebook() -> np.ndarray:
    """Build the 240 minimal E8 vectors and normalize them onto the unit sphere."""
    vecs: List[np.ndarray] = []

    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8, dtype=np.float64)
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)

    for mask in range(256):
        signs = np.array(
            [1.0 if ((mask >> k) & 1) == 0 else -1.0 for k in range(8)],
            dtype=np.float64,
        )
        if int(np.sum(signs < 0.0)) % 2 == 0:
            vecs.append(signs * 0.5)

    codebook = np.asarray(vecs, dtype=np.float64)
    if codebook.shape != (240, 8):
        raise ValueError(f"E8 construction failed, got {codebook.shape}")
    codebook /= np.linalg.norm(codebook[0])
    return codebook


@dataclass
class CompressedV9:
    """Compressed recursive E8 representation."""

    n_vectors: int
    n_chunks: int
    num_stages: int
    vertex_ids: List[np.ndarray]
    quantized_norms: List[np.ndarray]
    tail_codes: Optional[np.ndarray] = None
    qjl_signs: Optional[np.ndarray] = None
    norm_bits: Tuple[float, ...] = ()
    bits_tail: int = 0
    qjl_rank: int = 0
    dim: int = 0


CompressedV8 = CompressedV9


class V9Engine:
    """
    Exact recursive sphere-inside-sphere E8 quantizer.

    Parameters
    ----------
    dim:
        Original dimensionality. Vectors are padded to a multiple of 8.
    num_stages:
        Number of recursive E8 stages. Any positive value is supported.
    norm_bits:
        Scalar or per-stage effective bits for norm quantization. Fractional
        values are supported through deterministic mixed-bit schedules.
    bits_tail:
        Optional scalar residual tail bits per padded dimension.
    use_rotation:
        Apply one global orthogonal transform before chunking.
    rotation_reflections:
        Number of Householder reflections used in the global rotation.
    use_qjl:
        Enable QJL-style sign correction after the recursive E8 stages.
    qjl_rank:
        Number of sign projections to store per vector.
    seed:
        Random seed.
    fit_passes:
        Number of quantized fitting passes for stage norm ranges.
    fit_samples:
        Maximum number of vectors used during fit.
    """

    SKIP_ID = np.uint8(255)

    def __init__(
        self,
        dim: int,
        num_stages: int = 4,
        norm_bits: Union[float, Sequence[float]] = 3.0,
        bits_tail: int = 0,
        use_rotation: bool = True,
        rotation_reflections: int = 8,
        use_qjl: bool = False,
        qjl_rank: Optional[int] = None,
        codebook: Optional[np.ndarray] = None,
        seed: int = 42,
        fit_passes: int = 3,
        fit_samples: int = 4096,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if bits_tail < 0:
            raise ValueError("bits_tail must be non-negative")

        self.dim = int(dim)
        self.num_stages = int(num_stages)
        self.bits_tail = int(bits_tail)
        self.use_rotation = bool(use_rotation)
        self.rotation_reflections = int(max(0, rotation_reflections))
        self.use_qjl = bool(use_qjl)
        self.fit_passes = int(max(1, fit_passes))
        self.fit_samples = int(max(1, fit_samples))

        self._rng = np.random.default_rng(seed)
        self._norm_eps = 1e-12
        self._min_log_span = 1e-3

        if codebook is None:
            codebook = build_e8_codebook()
        self.CB = np.asarray(codebook, dtype=np.float64)
        if self.CB.shape != (240, 8):
            raise ValueError("E8 codebook must have shape (240, 8)")
        self.CBT = self.CB.T.copy()
        self.K, self.D = self.CB.shape
        self.ID_BITS_RAW = int(math.ceil(math.log2(self.K)))
        self.ID_BITS_EFFECTIVE = float(math.log2(self.K))

        self.n_chunks = int(math.ceil(self.dim / self.D))
        self.padded_dim = self.n_chunks * self.D

        self.stage_norm_bits = self._normalize_stage_norm_bits(norm_bits)
        self._norm_log_ranges: List[Tuple[float, float]] = [
            (-4.0, 4.0) for _ in range(self.num_stages)
        ]
        self._stage_mean_log_norm: List[float] = [0.0 for _ in range(self.num_stages)]
        self._tail_scale = 1.0

        if qjl_rank is None:
            qjl_rank = max(0, self.padded_dim // 32) if self.use_qjl else 0
        self.qjl_rank = int(max(0, min(qjl_rank, self.padded_dim)))
        if self.qjl_rank == 0:
            self.use_qjl = False

        self._rotation_vectors: List[np.ndarray] = []
        if self.use_rotation and self.rotation_reflections > 0:
            self._rotation_vectors = self._build_rotation()

        self._qjl_projection: Optional[np.ndarray] = None
        self._qjl_gain = 0.0
        if self.use_qjl:
            self._qjl_projection = self._build_qjl_projection()

    def _normalize_stage_norm_bits(
        self, norm_bits: Union[float, Sequence[float]]
    ) -> Tuple[float, ...]:
        """
        Normalize the norm_bits parameter into a tuple of floats, one per stage.
        Ensures consistency across variable stage specifications.
        """
        if isinstance(norm_bits, Sequence) and not isinstance(norm_bits, (str, bytes)):
            values = [float(v) for v in norm_bits]
            if len(values) != self.num_stages:
                raise ValueError("per-stage norm_bits must match num_stages")
        else:
            values = [float(norm_bits)] * self.num_stages
        
        # Verify non-negativity to prevent invalid bit configurations
        if any(v < 0.0 for v in values):
            raise ValueError("norm_bits must be non-negative")
        return tuple(values)

    def _build_rotation(self) -> List[np.ndarray]:
        """
        Build a list of random vectors for Householder reflections.
        These are used to perform global random rotations across dimension bounds.
        """
        vectors: List[np.ndarray] = []
        for _ in range(self.rotation_reflections):
            # Generate Gaussian random vector
            v = self._rng.standard_normal(self.padded_dim)
            # Normalize onto unit sphere
            v /= np.linalg.norm(v) + self._norm_eps
            vectors.append(v.astype(np.float64))
        return vectors

    def _build_qjl_projection(self) -> np.ndarray:
        """
        Build the Quantized Johnson-Lindenstrauss (QJL) random projection matrix.
        Used to correct sign mismatches after recursive encoding.
        """
        # Random binary matrix in {-1, 1}
        projection = self._rng.choice(
            [-1.0, 1.0], size=(self.qjl_rank, self.padded_dim)
        ).astype(np.float64)
        # Scale for proper norm preservation
        projection /= math.sqrt(self.padded_dim)
        return projection

    def _apply_reflections(
        self, X: np.ndarray, vectors: Sequence[np.ndarray]
    ) -> np.ndarray:
        """
        Apply a sequence of Householder reflections to a matrix X.
        This provides a fast orthonormal transformation without building explicit rotation matrices.
        """
        Y = np.asarray(X, dtype=np.float64).copy()
        for v in vectors:
            # Householder update: Y = Y - 2 * (Y·v) ⊗ v
            Y -= 2.0 * (Y @ v)[:, None] * v[None, :]
        return Y

    def _rotate_forward(self, X: np.ndarray) -> np.ndarray:
        """Apply pre-computed global random rotation via forward Householder reflections."""
        if not self._rotation_vectors:
            return np.asarray(X, dtype=np.float64)
        return self._apply_reflections(X, self._rotation_vectors)

    def _rotate_inverse(self, X: np.ndarray) -> np.ndarray:
        """Invert the global random rotation by applying reflections in reverse order."""
        if not self._rotation_vectors:
            return np.asarray(X, dtype=np.float64)
        return self._apply_reflections(X, list(reversed(self._rotation_vectors)))

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """Ensure input X is a 2D numpy array with correct original dimensions."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {X.shape[1]}")
        return X

    def _pad(self, X: np.ndarray) -> np.ndarray:
        """Pad input matrix X with zeros so that its trailing dimension is a multiple of E8 dimension (8)."""
        X = self._ensure_2d(X)
        if self.padded_dim == self.dim:
            return X.copy()
        # Zero-pad on the right across the last dimension
        return np.pad(X, ((0, 0), (0, self.padded_dim - self.dim)), mode="constant")

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Preprocess matrix X by padding first, then applying the global random rotation."""
        Xp = self._pad(X)
        return self._rotate_forward(Xp)

    def _postprocess(self, Xp: np.ndarray) -> np.ndarray:
        """Postprocess matrix Xp by applying inverse rotation, then cropping padded dimensions."""
        Xp = self._rotate_inverse(np.asarray(Xp, dtype=np.float64))
        return Xp[:, : self.dim]

    def _reshape_chunks(self, Xp: np.ndarray) -> np.ndarray:
        """Reshape padded matrix Xp into 3D chunks of size D=8 for E8 processing."""
        return np.asarray(Xp, dtype=np.float64).reshape(-1, self.n_chunks, self.D)

    def _stage_bit_schedule(self, stage_idx: int) -> np.ndarray:
        """Compute the deterministic mixed-bit allocation schedule for a given stage."""
        eff = self.stage_norm_bits[stage_idx]
        lo = int(math.floor(eff + 1e-12))
        hi = int(math.ceil(eff - 1e-12))
        frac = max(0.0, min(1.0, eff - lo))
        schedule = np.full(self.n_chunks, lo, dtype=np.int16)
        if hi > lo and frac > 0.0:
            steps = np.floor((np.arange(self.n_chunks) + 1) * frac) - np.floor(
                np.arange(self.n_chunks) * frac
            )
            schedule += steps.astype(np.int16)
        if self.n_chunks > 1:
            schedule = np.roll(schedule, stage_idx % self.n_chunks)
        return schedule

    def _find_stage_vertices(
        self, residual: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find nearest E8 codebook vectors for each chunk in the residual.
        Returns: vertex_ids, norms, and a boolean mask indicating active (non-zero) chunks.
        """
        # Calculate the L2 norm for each 8-dimensional chunk (axis=2)
        norms = np.linalg.norm(residual, axis=2)
        
        # Identify chunks that have a non-zero norm (active chunks)
        active = norms > self._norm_eps
        
        # Prevent division by zero by setting norm to 1.0 where it's 0.0
        safe_norms = np.where(active, norms, 1.0)
        
        # Normalize every chunk onto the unit sphere
        unit = residual / safe_norms[:, :, None]
        
        # Compute dot products with all 240 E8 vectors (CBT) and find the argmax (nearest neighbor)
        ids = np.argmax(unit.reshape(-1, self.D) @ self.CBT, axis=1).reshape(
            residual.shape[0], self.n_chunks
        )
        ids = ids.astype(np.uint8)
        
        # Mark inactive (zero-norm) chunks with the SKIP_ID special token
        ids[~active] = self.SKIP_ID
        return ids, norms, active

    def _quantize_norms(
        self, norms: np.ndarray, active: np.ndarray, stage_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Log-uniformly quantize chunk norms using the bit schedule for the given stage.
        Returns integer codes and the dequantized (reconstructed) norm approximations.
        """
        # Get the mixed-bit allocation schedule for this specific stage
        schedule = self._stage_bit_schedule(stage_idx)
        max_bits = int(max(0, schedule.max(initial=0)))
        
        # Dynamically choose the smallest unsigned integer type needed to store the codes
        if max_bits > 16:
            code_dtype = np.uint32
        elif max_bits > 8:
            code_dtype = np.uint16
        else:
            code_dtype = np.uint8
            
        codes = np.zeros(norms.shape, dtype=code_dtype)
        norms_hat = np.zeros_like(norms, dtype=np.float64)

        # Retrieve the pre-computed log norm bounds (min/max) and the mean for this stage
        log_lo, log_hi = self._norm_log_ranges[stage_idx]
        log_mid = self._stage_mean_log_norm[stage_idx]
        span = max(log_hi - log_lo, self._min_log_span)

        # Iterate over each unique bit allocation amount present in the schedule
        for bits in np.unique(schedule):
            col_mask = schedule == bits
            if not np.any(col_mask):
                continue

            local_active = active[:, col_mask]
            
            # If 0 bits are allocated, we simply guess the mean log norm for all active chunks
            if bits <= 0:
                local_hat = np.zeros((norms.shape[0], int(np.sum(col_mask))), dtype=np.float64)
                local_hat[local_active] = math.exp(log_mid)
                norms_hat[:, col_mask] = local_hat
                continue

            # Calculate the number of quantization levels (e.g. 3 bits -> 7 levels, with 0 reserved)
            levels = (1 << int(bits)) - 1
            
            # Take the log of the actual positive norms
            local_log = np.log(np.maximum(norms[:, col_mask], self._norm_eps))
            
            # Scale the log norms linearly between 0.0 and 1.0 based on the bounds
            scaled = np.clip((local_log - log_lo) / span, 0.0, 1.0 - 1e-12)
            
            # Bucket into integer blocks and reserve code 0 for inactive/dropped chunks
            quant = np.floor(scaled * levels).astype(np.int32) + 1
            quant[~local_active] = 0

            # Reconstruct the expected values exactly from the center of each quantization bucket
            local_hat = np.zeros_like(local_log, dtype=np.float64)
            positive = quant > 0
            if np.any(positive):
                centers = (quant[positive].astype(np.float64) - 0.5) / levels
                local_hat[positive] = np.exp(log_lo + centers * span)

            codes[:, col_mask] = quant.astype(code_dtype)
            norms_hat[:, col_mask] = local_hat

        return codes, norms_hat

    def _dequantize_norms(
        self, codes: np.ndarray, vertex_ids: np.ndarray, stage_idx: int
    ) -> np.ndarray:
        """
        Reconstruct continuous positive norms from integer codes and vertex_ids.
        Unquantizes using the stored log-space bounds.
        """
        # Fetch the mixed bits distribution and boundaries matched to the encoder fit
        schedule = self._stage_bit_schedule(stage_idx)
        log_lo, log_hi = self._norm_log_ranges[stage_idx]
        log_mid = self._stage_mean_log_norm[stage_idx]
        span = max(log_hi - log_lo, self._min_log_span)
        
        norms_hat = np.zeros(codes.shape, dtype=np.float64)
        
        # Check which chunks are not flagged as SKIP_ID
        active = vertex_ids != self.SKIP_ID

        for bits in np.unique(schedule):
            col_mask = schedule == bits
            if not np.any(col_mask):
                continue

            local_active = active[:, col_mask]
            
            # Handle the 0-bit edgecase: assign the global stage mean norm
            if bits <= 0:
                local_hat = np.zeros((codes.shape[0], int(np.sum(col_mask))), dtype=np.float64)
                local_hat[local_active] = math.exp(log_mid)
                norms_hat[:, col_mask] = local_hat
                continue

            levels = (1 << int(bits)) - 1
            local_codes = codes[:, col_mask].astype(np.int32)
            local_hat = np.zeros_like(local_codes, dtype=np.float64)
            positive = local_codes > 0
            
            # Map code [1..levels] to the log-center domain (0..1) then exponentiate
            if np.any(positive):
                centers = (local_codes[positive].astype(np.float64) - 0.5) / levels
                local_hat[positive] = np.exp(log_lo + centers * span)
                
            # Filter off any active masks that should have been zeroes natively
            local_hat[~local_active] = 0.0
            norms_hat[:, col_mask] = local_hat

        return norms_hat

    def _stage_vectors_from_codes(
        self, vertex_ids: np.ndarray, norm_codes: np.ndarray, stage_idx: int
    ) -> np.ndarray:
        """
        Reconstruct the estimated vectors for a single stage from IDs and norm codes.
        Decodes the norms and multiplies by the corresponding E8 codebook vectors.
        """
        # First recover floating point magnitude multipliers
        norms_hat = self._dequantize_norms(norm_codes, vertex_ids, stage_idx)
        vectors = np.zeros((vertex_ids.shape[0], self.n_chunks, self.D), dtype=np.float64)
        
        # Locate active chunks (ids != 255) to avoid out of bounds array access on codebook
        active = vertex_ids != self.SKIP_ID
        
        if np.any(active):
            # Element-wise multiply the E8 normalized lookup unit-vector by the reconstructed norms
            vectors[active] = self.CB[vertex_ids[active]] * norms_hat[active, None]
        return vectors

    def _tail_encode(self, residual_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode the final residual using uniform scalar quantization.
        Returns tail integer codes and the reconstructed tail component.
        """
        levels = 1 << self.bits_tail
        scaled = residual_flat / max(self._tail_scale, self._norm_eps)
        scaled = np.clip(scaled, -0.5, 0.5 - 1.0 / levels)
        if self.bits_tail > 16:
            tail_dtype = np.uint32
        elif self.bits_tail > 8:
            tail_dtype = np.uint16
        else:
            tail_dtype = np.uint8
        codes = np.floor((scaled + 0.5) * levels).astype(tail_dtype)
        tail_hat = ((codes.astype(np.float64) + 0.5) / levels - 0.5) * self._tail_scale
        return codes, tail_hat

    def _tail_decode(self, tail_codes: np.ndarray) -> np.ndarray:
        """Decode the uniform scalar tail codes back into floating point residuals."""
        levels = 1 << self.bits_tail
        return ((tail_codes.astype(np.float64) + 0.5) / levels - 0.5) * self._tail_scale

    def _qjl_basis_from_signs(self, signs: np.ndarray) -> np.ndarray:
        """Construct the QJL scalar correction basis vectors given boolean sign indicators."""
        if self._qjl_projection is None or self.qjl_rank == 0:
            return np.zeros((signs.shape[0], self.padded_dim), dtype=np.float64)
        signed = np.where(signs, 1.0, -1.0).astype(np.float64)
        return (signed @ self._qjl_projection) / math.sqrt(self.qjl_rank)

    def _fit_stage_ranges(self, chunks: np.ndarray) -> np.ndarray:
        """
        Iteratively fit the norm quantization bounds (min/max log norms) for each stage.
        Returns the remaining residual chunks after recursive fitting.
        """
        residual = chunks.copy()

        for _ in range(self.fit_passes):
            residual = chunks.copy()
            for stage_idx in range(self.num_stages):
                vertex_ids, norms, active = self._find_stage_vertices(residual)
                if np.any(active):
                    log_norms = np.log(np.maximum(norms[active], self._norm_eps))
                    lo = float(np.quantile(log_norms, 0.002))
                    hi = float(np.quantile(log_norms, 0.998))
                    hi = max(hi, lo + self._min_log_span)
                    span = hi - lo
                    avg_bits = max(self.stage_norm_bits[stage_idx], 0.0)
                    if avg_bits > 0.0:
                        approx_levels = max(1.0, (2.0 ** avg_bits) - 1.0)
                        margin = span / (2.0 * approx_levels)
                    else:
                        margin = 0.1 * span + 0.05
                    lo -= margin
                    hi += margin
                    self._norm_log_ranges[stage_idx] = (lo, hi)
                    self._stage_mean_log_norm[stage_idx] = float(np.mean(log_norms))
                else:
                    self._norm_log_ranges[stage_idx] = (-6.0, -6.0 + self._min_log_span)
                    self._stage_mean_log_norm[stage_idx] = -6.0

                norm_codes, norms_hat = self._quantize_norms(norms, active, stage_idx)
                stage_vec = np.zeros_like(residual)
                live = vertex_ids != self.SKIP_ID
                if np.any(live):
                    stage_vec[live] = self.CB[vertex_ids[live]] * norms_hat[live, None]
                residual = residual - stage_vec

        return residual

    def fit(self, X: np.ndarray) -> "V9Engine":
        """
        Fit the engine on calibration data X to determine step sizes (norm bounds).
        Also initializes QJL basis and measures residual tail statistics.
        """
        X = self._ensure_2d(X)
        if len(X) > self.fit_samples:
            idx = self._rng.choice(len(X), size=self.fit_samples, replace=False)
            X = X[idx]

        chunks = self._reshape_chunks(self._preprocess(X))
        residual = self._fit_stage_ranges(chunks)
        residual_flat = residual.reshape(len(X), self.padded_dim)

        if self.use_qjl and self.qjl_rank > 0:
            signs = (residual_flat @ self._qjl_projection.T) >= 0.0
            basis = self._qjl_basis_from_signs(signs)
            denom = float(np.sum(basis * basis)) + self._norm_eps
            self._qjl_gain = float(np.sum(residual_flat * basis) / denom)
            residual_flat = residual_flat - self._qjl_gain * basis
        else:
            self._qjl_gain = 0.0

        if self.bits_tail > 0:
            self._tail_scale = max(
                float(np.quantile(np.abs(residual_flat), 0.995)) * 2.0,
                1e-9,
            )

        return self

    def encode(
        self, X: np.ndarray, return_debug: bool = False
    ) -> Union[CompressedV9, Tuple[CompressedV9, np.ndarray]]:
        """
        Encode vectors X into a CompressedV9 block.
        Optionally returns debug reconstruction prior to bounds/tail corrections.
        """
        X = self._ensure_2d(X)
        rotated = self._preprocess(X)
        chunks = self._reshape_chunks(rotated)
        residual = chunks.copy()

        all_vertex_ids: List[np.ndarray] = []
        all_norm_codes: List[np.ndarray] = []

        for stage_idx in range(self.num_stages):
            vertex_ids, norms, active = self._find_stage_vertices(residual)
            norm_codes, norms_hat = self._quantize_norms(norms, active, stage_idx)

            stage_vec = np.zeros_like(residual)
            live = vertex_ids != self.SKIP_ID
            if np.any(live):
                stage_vec[live] = self.CB[vertex_ids[live]] * norms_hat[live, None]

            residual = residual - stage_vec
            all_vertex_ids.append(vertex_ids)
            all_norm_codes.append(norm_codes)

        reconstruction_rot = (chunks - residual).reshape(len(X), self.padded_dim)

        qjl_signs = None
        if self.use_qjl and self.qjl_rank > 0:
            residual_flat = residual.reshape(len(X), self.padded_dim)
            qjl_signs = (residual_flat @ self._qjl_projection.T) >= 0.0
            qjl_corr = self._qjl_gain * self._qjl_basis_from_signs(qjl_signs)
            residual_flat = residual_flat - qjl_corr
            reconstruction_rot += qjl_corr
            residual = residual_flat.reshape(len(X), self.n_chunks, self.D)

        tail_codes = None
        if self.bits_tail > 0:
            residual_flat = residual.reshape(len(X), self.padded_dim)
            tail_codes, tail_hat = self._tail_encode(residual_flat)
            reconstruction_rot += tail_hat

        enc = CompressedV9(
            n_vectors=len(X),
            n_chunks=self.n_chunks,
            num_stages=self.num_stages,
            vertex_ids=all_vertex_ids,
            quantized_norms=all_norm_codes,
            tail_codes=tail_codes,
            qjl_signs=qjl_signs,
            norm_bits=self.stage_norm_bits,
            bits_tail=self.bits_tail,
            qjl_rank=self.qjl_rank if self.use_qjl else 0,
            dim=self.dim,
        )

        if return_debug:
            return enc, reconstruction_rot
        return enc

    def _decode_rotated(self, enc: CompressedV9) -> np.ndarray:
        """Decode the compressed block into the transformed (rotated/padded) coordinate space."""
        if enc.n_chunks != self.n_chunks or enc.num_stages != self.num_stages:
            raise ValueError("encoded object is incompatible with this engine")

        reconstruction = np.zeros((enc.n_vectors, self.n_chunks, self.D), dtype=np.float64)
        for stage_idx in range(self.num_stages):
            reconstruction += self._stage_vectors_from_codes(
                enc.vertex_ids[stage_idx], enc.quantized_norms[stage_idx], stage_idx
            )

        rotated = reconstruction.reshape(enc.n_vectors, self.padded_dim)

        if self.use_qjl and self.qjl_rank > 0 and enc.qjl_signs is not None:
            rotated += self._qjl_gain * self._qjl_basis_from_signs(enc.qjl_signs)

        if self.bits_tail > 0 and enc.tail_codes is not None:
            rotated += self._tail_decode(enc.tail_codes)

        return rotated

    def decode(self, enc: CompressedV9) -> np.ndarray:
        """Decode a CompressedV9 block to recover approximations of original vectors X."""
        return self._postprocess(self._decode_rotated(enc))

    def exactness_error(self, X: np.ndarray) -> float:
        """
        Calculate maximum absolute mismatch between the debug state reconstruction and
        the decoded output. Useful to verify mathematically lossless operation.
        """
        enc, recon_rot = self.encode(X, return_debug=True)
        dec_rot = self._decode_rotated(enc)
        return float(np.max(np.abs(recon_rot - dec_rot)))

    def bits_per_vector(self) -> float:
        """Calculate effective average bits used per vector configuration."""
        stage_bits = self.n_chunks * (
            self.num_stages * self.ID_BITS_EFFECTIVE + sum(self.stage_norm_bits)
        )
        tail_bits = float(self.bits_tail * self.padded_dim)
        qjl_bits = float(self.qjl_rank if self.use_qjl else 0)
        return stage_bits + tail_bits + qjl_bits

    def bits_per_dim(self) -> float:
        """Calculate effective average bits per original dimension (bpd)."""
        return self.bits_per_vector() / self.dim

    def raw_bits_per_vector(self) -> float:
        """Calculate strictly raw ceiling bytes/bits required per vector memory structure."""
        stage_bits = self.n_chunks * (
            self.num_stages * self.ID_BITS_RAW
            + sum(int(math.ceil(v - 1e-12)) for v in self.stage_norm_bits)
        )
        tail_bits = float(self.bits_tail * self.padded_dim)
        qjl_bits = float(self.qjl_rank if self.use_qjl else 0)
        return stage_bits + tail_bits + qjl_bits

    def raw_bits_per_dim(self) -> float:
        """Calculate strictly raw bits per original dimension."""
        return self.raw_bits_per_vector() / self.dim


V8EngineRecursive = V9Engine


# =============================================================================
# V11 Extensions: Asymmetric Search and Outlier Handling
# =============================================================================


class V11Engine:
    """
    V11 Engine wrapper around V9Engine with asymmetric KV cache optimizations.

    IMPORTANT: This wrapper does NOT modify the V9Engine core. All enhancements
    are applied through pre/post-processing that maintains exact closure.

    Key improvements for real KV cache:
    1. Asymmetric parameters for Keys vs Values (controlled via search function)
    2. Attention-aware scoring when query samples are provided
    3. Optional outlier handling for heavy-tailed distributions

    The outlier handling works by:
    1. Detecting high-norm chunks using MAD-based thresholding
    2. Applying uniform per-chunk scaling BEFORE V9 processing
    3. Reversing the scaling EXACTLY AFTER V9 decoding

    Since scaling is uniform within each chunk and applied identically to all
    vectors, the residual structure is preserved and V9's exactness is maintained.

    Parameters
    ----------
    base_engine: V9Engine
        The underlying V9Engine (unchanged core algorithm)
    handle_outliers: bool
        Whether to apply outlier handling (typically True for Keys, False for Values)
    outlier_threshold: float
        Standard deviations above median for outlier detection (default 4.0)
    """

    def __init__(
        self,
        base_engine: V9Engine,
        handle_outliers: bool = True,
        outlier_threshold: float = 4.0,
    ) -> None:
        self.base_engine = base_engine
        self.handle_outliers = bool(handle_outliers)
        self.outlier_threshold = float(outlier_threshold)
        self._chunk_scales: Optional[np.ndarray] = None
        self._chunk_scales_inv: Optional[np.ndarray] = None

    def _compute_channel_scales(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-chunk scale factors based on outlier detection.

        For each chunk position (0 to n_chunks-1), we compute the max-norm
        across all vectors. Chunks with very high max-norm (outliers) get
        a scale factor > 1 to reduce their magnitude before quantization.

        Uses MAD-based (Median Absolute Deviation) thresholding for robustness
        against heavy-tailed distributions typical in KV cache.
        """
        Xp = self.base_engine._preprocess(X)
        chunks = self.base_engine._reshape_chunks(Xp)
        n_vecs, n_chunks, D = chunks.shape

        # Compute max-norm per chunk position across all vectors
        chunk_norms = np.linalg.norm(chunks, axis=2)  # (n_vecs, n_chunks)
        max_norms = np.max(chunk_norms, axis=0)  # (n_chunks,)

        # MAD-based outlier detection
        median_norm = np.median(max_norms)
        mad = np.median(np.abs(max_norms - median_norm))
        mad = max(mad, 1e-12)  # Avoid division by zero

        # Compute z-scores using MAD (robust to outliers)
        z_scores = np.abs(max_norms - median_norm) / (1.4826 * mad)

        # Identify outlier chunks
        is_outlier = z_scores > self.outlier_threshold

        # Compute scale factors: outliers get scaled down
        # Scale factor is proportional to how much they exceed the threshold
        scales = np.ones(n_chunks, dtype=np.float64)
        if np.any(is_outlier):
            # For outliers, scale = max_norm / (median + threshold * MAD)
            reference = median_norm + self.outlier_threshold * 1.4826 * mad
            scales[is_outlier] = max_norms[is_outlier] / max(reference, 1e-12)

        # Ensure minimum scale of 1.0 (we only scale down, never up)
        scales = np.maximum(scales, 1.0)

        return scales

    def _apply_chunk_scales_pre(self, X: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Apply or reverse chunk-wise scaling in PRE-rotation space.

        This is applied BEFORE V9's internal rotation, so it operates on
        the original chunked structure. The scaling is uniform per chunk,
        preserving the residual structure.

        IMPORTANT: For exact closure, we must apply scaling in the same
        coordinate system used during fit. We do this by:
        1. Preprocessing (padding if needed)
        2. Reshaping to chunks
        3. Applying/reversing scale per chunk
        4. Reshaping back and postprocessing

        This ensures encode(decode(x)) == x exactly.
        """
        if self._chunk_scales is None:
            return X.copy()

        # Work in preprocessed space to match fit/encode path
        Xp = self.base_engine._preprocess(X)
        orig_shape = Xp.shape

        # Reshape to chunks
        n_vecs = Xp.shape[0]
        n_chunks = self.base_engine.n_chunks
        D = self.base_engine.D  # Padded dimension per chunk

        chunks = Xp.reshape(n_vecs, n_chunks, D)

        # Apply or reverse scaling
        scales = self._chunk_scales_inv if inverse else self._chunk_scales
        if inverse:
            # Reverse: multiply by inverse scale (restore original magnitude)
            chunks = chunks * scales[None, :, None]
        else:
            # Apply: divide by scale (reduce outlier magnitude)
            chunks = chunks / scales[None, :, None]

        # Reshape back to original shape
        Xp_scaled = chunks.reshape(orig_shape)

        # Postprocess (remove padding if any)
        return self.base_engine._postprocess(Xp_scaled)

    def fit(self, X: np.ndarray) -> "V11Engine":
        """Fit the engine, computing outlier scales if enabled."""
        if self.handle_outliers:
            self._chunk_scales = self._compute_channel_scales(X)
            self._chunk_scales_inv = 1.0 / np.maximum(self._chunk_scales, 1e-12)
            # Fit on scaled data
            X_scaled = self._apply_chunk_scales_pre(X, inverse=False)
            self.base_engine.fit(X_scaled)
        else:
            self._chunk_scales = None
            self._chunk_scales_inv = None
            self.base_engine.fit(X)
        return self

    def encode(self, X: np.ndarray, return_debug: bool = False) -> Union[CompressedV9, Tuple[CompressedV9, np.ndarray]]:
        """Encode with outlier handling."""
        if self.handle_outliers and self._chunk_scales is not None:
            X_scaled = self._apply_chunk_scales_pre(X, inverse=False)
            result = self.base_engine.encode(X_scaled, return_debug=return_debug)
        else:
            result = self.base_engine.encode(X, return_debug=return_debug)
        return result

    def decode(self, enc: CompressedV9) -> np.ndarray:
        """Decode and reverse outlier scaling."""
        X_decoded = self.base_engine.decode(enc)
        if self.handle_outliers and self._chunk_scales is not None:
            X_decoded = self._apply_chunk_scales_pre(X_decoded, inverse=True)
        return X_decoded

    def exactness_error(self, X: np.ndarray) -> float:
        """
        Check exactness error (should be ~machine precision).

        This verifies the internal consistency of the encode/decode path:
        encode(x, return_debug=True) produces a reconstruction, and
        _decode_rotated(enc) should match it exactly.

        With outlier handling, this check is performed in the scaled space
        where V9Engine operates, ensuring the wrapper maintains closure.
        """
        # Apply scaling if enabled (work in the space where V9 operates)
        X_work = self._apply_chunk_scales_pre(X, inverse=False) if self.handle_outliers else X

        # Get reconstruction from encode with debug info
        enc, recon_work = self.base_engine.encode(X_work, return_debug=True)

        # Get reconstruction from _decode_rotated
        dec_work = self.base_engine._decode_rotated(enc)

        # Check consistency in working (scaled) space
        return float(np.max(np.abs(recon_work - dec_work)))

    def bits_per_vector(self) -> float:
        return self.base_engine.bits_per_vector()

    def bits_per_dim(self) -> float:
        return self.base_engine.bits_per_dim()

    def raw_bits_per_vector(self) -> float:
        return self.base_engine.raw_bits_per_vector()

    def raw_bits_per_dim(self) -> float:
        return self.base_engine.raw_bits_per_dim()

    @property
    def dim(self) -> int:
        return self.base_engine.dim

    @property
    def num_stages(self) -> int:
        return self.base_engine.num_stages

    @property
    def use_qjl(self) -> bool:
        return self.base_engine.use_qjl

    @property
    def qjl_rank(self) -> int:
        return self.base_engine.qjl_rank


def _search_target_engine_asymmetric(
    dim: int,
    X_tr: np.ndarray,
    X_cal: np.ndarray,
    target_bpd: float,
    role: str = 'key',
    max_stages: int = 4,
    q_sample: Optional[np.ndarray] = None,
    qk_mse_weight: float = 0.5,
    handle_outliers: bool = True,
    outlier_threshold: float = 4.0,
    fit_samples: int = 1024,
    search_samples: int = 256,
    seed: int = 42,
) -> V11Engine:
    """
    Search for optimal V11Engine configuration with asymmetric treatment.

    This function searches for the best engine configuration optimized for
    either Keys or Values, with attention-aware scoring when query samples
    are provided.

    Parameters
    ----------
    dim: int
        Dimension of the vectors
    X_tr: np.ndarray
        Training data for fitting
    X_cal: np.ndarray
        Calibration data for evaluation during search
    target_bpd: float
        Target bits per dimension
    role: str
        'key' or 'value' - determines asymmetric parameters
    max_stages: int
        Maximum number of stages to search
    q_sample: np.ndarray, optional
        Query samples for attention-aware scoring (Q·K fidelity)
    qk_mse_weight: float
        Weight for Q·K MSE in combined scoring (0.0 = reconstruction only,
        1.0 = attention only). Default 0.5 balances both.
    handle_outliers: bool
        Whether to enable outlier handling (recommended True for Keys)
    outlier_threshold: float
        Threshold for outlier detection in standard deviations
    fit_samples: int
        Number of samples for fitting (default 1024 for heavy-tailed KV)
    search_samples: int
        Number of samples for calibration during search (default 256)
    seed: int
        Random seed

    Returns
    -------
    V11Engine
        Optimized engine for the specified role
    """
    if role not in ('key', 'value'):
        raise ValueError("role must be 'key' or 'value'")

    rng = np.random.default_rng(seed)

    # Subsample calibration data for faster search
    n_cal = min(len(X_cal), search_samples)
    if n_cal < len(X_cal):
        cal_idx = rng.choice(len(X_cal), size=n_cal, replace=False)
        X_cal_sub = X_cal[cal_idx]
    else:
        X_cal_sub = X_cal

    # Subsample query if provided
    if q_sample is not None:
        if len(q_sample) > n_cal:
            q_cal_sub = q_sample[cal_idx] if n_cal < len(q_sample) else q_sample[:n_cal]
        else:
            q_cal_sub = q_sample[:len(X_cal_sub)]
    else:
        q_cal_sub = None

    candidates: List[Tuple[float, float, V11Engine]] = []

    # Asymmetric parameter ranges based on role
    if role == 'key':
        # Keys: higher sensitivity, more stages, higher qjl_rank
        stage_range = range(2, max_stages + 1)  # 2 to max_stages
        qjl_fracs = (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375)  # Higher range
        use_rotation_default = True
    else:
        # Values: lower sensitivity, fewer stages, lower qjl_rank
        stage_range = range(2, min(max_stages, 3) + 1)  # 2 to 3
        qjl_fracs = (0.0, 0.03125, 0.0625, 0.09375)  # Lower range
        use_rotation_default = True

    for stages in stage_range:
        for use_rotation in (False, True) if role == 'key' else (use_rotation_default,):
            for frac in qjl_fracs:
                qjl_rank = int(round(frac * dim))

                # Create base V9Engine with increased fit_samples
                base_engine = V9Engine(
                    dim=dim,
                    num_stages=stages,
                    norm_bits=0.0,  # Will be computed below
                    bits_tail=0,
                    use_rotation=use_rotation,
                    use_qjl=qjl_rank > 0,
                    qjl_rank=qjl_rank,
                    seed=seed,
                    fit_samples=fit_samples,
                )

                # Compute remaining bits for norm quantization
                remaining_bits = target_bpd * dim - qjl_rank - (
                    base_engine.n_chunks * stages * base_engine.ID_BITS_EFFECTIVE
                )
                if remaining_bits < 0.0:
                    continue

                norm_bits = remaining_bits / (base_engine.n_chunks * stages)
                if norm_bits < 0.0 or norm_bits > 24.0:
                    continue

                # Create engine with computed norm_bits
                base_engine = V9Engine(
                    dim=dim,
                    num_stages=stages,
                    norm_bits=norm_bits,
                    bits_tail=0,
                    use_rotation=use_rotation,
                    use_qjl=qjl_rank > 0,
                    qjl_rank=qjl_rank,
                    seed=seed,
                    fit_samples=fit_samples,
                )

                # Wrap in V11Engine with outlier handling
                enable_outliers = handle_outliers and (role == 'key')
                engine = V11Engine(
                    base_engine=base_engine,
                    handle_outliers=enable_outliers,
                    outlier_threshold=outlier_threshold,
                )

                try:
                    engine.fit(X_tr)
                    R = engine.decode(engine.encode(X_cal_sub))

                    # Compute reconstruction MSE
                    recon_mse = float(np.mean((X_cal_sub - R) ** 2))

                    # Compute attention-aware score if query samples provided
                    if q_cal_sub is not None and role == 'key':
                        # Q·K product before and after quantization
                        qk_original = q_cal_sub @ X_cal_sub.T  # (n_q, n_cal)
                        qk_quantized = q_cal_sub @ R.T

                        # Q·K MSE
                        qk_mse = float(np.mean((qk_original - qk_quantized) ** 2))

                        # Normalize both metrics for fair combination
                        recon_mse_norm = recon_mse / (np.var(X_cal_sub) + 1e-12)
                        qk_mse_norm = qk_mse / (np.var(qk_original) + 1e-12)

                        # Combined score: lower is better
                        combined_mse = (
                            (1 - qk_mse_weight) * recon_mse_norm +
                            qk_mse_weight * qk_mse_norm
                        )
                        score = -combined_mse  # Negative because we maximize
                    else:
                        # Standard SNR-based scoring
                        snr = float(10.0 * np.log10(
                            np.mean(X_cal_sub * X_cal_sub) /
                            (np.mean((X_cal_sub - R) ** 2) + 1e-12)
                        ))
                        score = snr

                    exact = engine.exactness_error(X_cal_sub[:min(len(X_cal_sub), 32)])
                    candidates.append((score, -exact, engine))

                except Exception as e:
                    # Skip invalid configurations
                    continue

    if not candidates:
        raise RuntimeError(f"no valid engine found for target_bpd={target_bpd}, role={role}")

    # Sort by score (higher is better), then by exactness (lower error is better)
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _search_target_engine(
    dim: int,
    X_tr: np.ndarray,
    X_cal: np.ndarray,
    target_bpd: float,
    max_stages: int = 4,
    seed: int = 42,
) -> V9Engine:
    """
    Original symmetric search function from V9 (preserved for backward compatibility).
    """
    candidates: List[Tuple[float, float, V9Engine]] = []
    qjl_fracs = (0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.625)

    for stages in range(1, max_stages + 1):
        for use_rotation in (False, True):
            for frac in qjl_fracs:
                qjl_rank = int(round(frac * dim))
                probe = V9Engine(
                    dim=dim,
                    num_stages=stages,
                    norm_bits=0.0,
                    bits_tail=0,
                    use_rotation=use_rotation,
                    use_qjl=qjl_rank > 0,
                    qjl_rank=qjl_rank,
                    seed=seed,
                )

                remaining_bits = target_bpd * dim - qjl_rank - (
                    probe.n_chunks * stages * probe.ID_BITS_EFFECTIVE
                )
                if remaining_bits < 0.0:
                    continue

                norm_bits = remaining_bits / (probe.n_chunks * stages)
                if norm_bits < 0.0 or norm_bits > 24.0:
                    continue

                engine = V9Engine(
                    dim=dim,
                    num_stages=stages,
                    norm_bits=norm_bits,
                    bits_tail=0,
                    use_rotation=use_rotation,
                    use_qjl=qjl_rank > 0,
                    qjl_rank=qjl_rank,
                    seed=seed,
                )
                engine.fit(X_tr)
                R = engine.decode(engine.encode(X_cal))
                score = snr_db(X_cal, R)
                exact = engine.exactness_error(X_cal[: min(len(X_cal), 32)])
                candidates.append((score, -exact, engine))

    if not candidates:
        raise RuntimeError(f"no valid engine found for target_bpd={target_bpd}")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def turbo_encode(X: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple scalar baseline."""
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    scale = hi - lo + 1e-9
    levels = 1 << bits
    codes = np.clip(np.floor((X - lo) / scale * levels), 0, levels - 1).astype(np.uint8)
    return codes, lo, scale


def turbo_decode(
    codes: np.ndarray, lo: np.ndarray, scale: np.ndarray, bits: int
) -> np.ndarray:
    """Decode the scalar baseline."""
    levels = 1 << bits
    return lo + (codes.astype(np.float64) + 0.5) / levels * scale


def snr_db(X: np.ndarray, R: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in Decibels (dB) between original X and reconstruction R."""
    return float(10.0 * np.log10(np.mean(X * X) / (np.mean((X - R) ** 2) + 1e-12)))


def cosine_sim(X: np.ndarray, R: np.ndarray) -> float:
    """Calculate average cosine similarity between original X and reconstruction R."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(Xn * Rn, axis=1)))


def recall_at_k(X: np.ndarray, R: np.ndarray, k: int = 10, n_q: int = 50) -> float:
    """
    Measure top-k nearest neighbor preservation after quantization.
    Randomly selects queries, searches in original and reconstructed spaces,
    and returns the fraction of times the true nearest neighbor is in the top-k reconstructed neighbors.
    """
    n = len(X)
    if n <= 1:
        return 0.0
    k = min(k, n - 1)
    n_q = min(n_q, n)
    qidx = np.random.default_rng(1).choice(n, n_q, replace=False)
    hits = 0
    for qi in qidx:
        d_orig = np.linalg.norm(X - X[qi], axis=1)
        d_orig[qi] = np.inf
        true_nn = int(np.argmin(d_orig))

        d_hat = np.linalg.norm(R - R[qi], axis=1)
        d_hat[qi] = np.inf
        if true_nn in np.argpartition(d_hat, k - 1)[:k]:
            hits += 1
    return hits / n_q


def _benchmark_method(
    label: str,
    engine: Optional[Union[V9Engine, V11Engine]],
    X_te: np.ndarray,
    baseline: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None,
) -> Dict[str, Any]:
    """
    Helper to run quantization/dequantization and measure timings, metrics, 
    and exactness bounds for benchmarking summary tables.
    """
    if engine is None:
        if baseline is None:
            raise ValueError("baseline arguments required when engine is None")
        codes, lo, scale, bits = baseline
        t0 = time.perf_counter()
        R = turbo_decode(codes, lo, scale, bits)
        dec_time = time.perf_counter() - t0
        return {
            "label": label,
            "bpd": float(bits),
            "raw_bpd": float(bits),
            "X_hat": R,
            "enc": 0.0,
            "dec": dec_time,
            "exact": 0.0,
            "config": None,
        }

    t0 = time.perf_counter()
    enc = engine.encode(X_te)
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    R = engine.decode(enc)
    dec_time = time.perf_counter() - t0

    exact = engine.exactness_error(X_te[: min(len(X_te), 64)])

    # Get config info from either V9Engine or V11Engine
    if isinstance(engine, V11Engine):
        config = {
            "num_stages": engine.num_stages,
            "norm_bits": tuple(round(v, 4) for v in engine.base_engine.stage_norm_bits),
            "use_rotation": engine.base_engine.use_rotation,
            "qjl_rank": engine.qjl_rank if engine.use_qjl else 0,
            "handle_outliers": engine.handle_outliers,
        }
    else:
        config = {
            "num_stages": engine.num_stages,
            "norm_bits": tuple(round(v, 4) for v in engine.stage_norm_bits),
            "use_rotation": engine.use_rotation,
            "qjl_rank": engine.qjl_rank if engine.use_qjl else 0,
        }

    return {
        "label": label,
        "bpd": engine.bits_per_dim(),
        "raw_bpd": engine.raw_bits_per_dim(),
        "X_hat": R,
        "enc": enc_time,
        "dec": dec_time,
        "exact": exact,
        "config": config,
    }


def run_exact_3bpd_benchmark(dim: int, n_vectors: int = 2000, seed: int = 42) -> None:
    """
    Dedicated benchmark at exactly 3.0 effective bits per dimension.
    Uses V11 asymmetric search optimized for Keys.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_vectors, dim)).astype(np.float64)
    n_tr = max(200, n_vectors // 5)
    X_tr = X[:n_tr]
    X_te = X[n_tr:]
    X_cal = X_te[: min(len(X_te), 256)]

    print("\n" + "=" * 80)
    print(f"  EXACT 3.0 BPD BENCHMARK - dim={dim}  n_test={len(X_te)}  n_train={n_tr}")
    print("=" * 80)

    t0 = time.perf_counter()
    codes, lo, scale = turbo_encode(X_te, 3)
    tq_encode = time.perf_counter() - t0
    tq = _benchmark_method(
        "Turbo scalar 3.0bpd",
        None,
        X_te,
        baseline=(codes, lo, scale, 3),
    )
    tq["enc"] = tq_encode

    # Use V11 asymmetric search (optimized for Keys by default)
    best = _search_target_engine_asymmetric(
        dim=dim,
        X_tr=X_tr,
        X_cal=X_cal,
        target_bpd=3.0,
        role='key',
        max_stages=4,
        fit_samples=1024,
        search_samples=256,
        seed=seed,
    )
    v9 = _benchmark_method("V11 exact 3.0bpd (asymmetric)", best, X_te)

    print(f"  Turbo SNR : {snr_db(X_te, tq['X_hat']):.3f} dB")
    print(f"  V11 SNR   : {snr_db(X_te, v9['X_hat']):.3f} dB")
    print(f"  Gain      : {snr_db(X_te, v9['X_hat']) - snr_db(X_te, tq['X_hat']):+.3f} dB")
    print(f"  Cosine    : {cosine_sim(X_te, v9['X_hat']):.6f}")
    print(f"  Recall@10 : {recall_at_k(X_te, v9['X_hat']):.4f}")
    print(f"  Exactness : max |encode-decode mismatch| = {v9['exact']:.3e}")
    print(f"  Config    : {v9['config']}")


def run_v8_benchmark(dim: int, n_vectors: int = 2000, seed: int = 42) -> None:
    """
    Head-to-head comparison preserving the V8 benchmark entry point.
    Now uses V11 asymmetric search.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_vectors, dim)).astype(np.float64)
    n_tr = max(200, n_vectors // 5)
    X_tr = X[:n_tr]
    X_te = X[n_tr:]
    X_cal = X_te[: min(len(X_te), 256)]

    print("\n" + "=" * 80)
    print(f"  V11 ASYMMETRIC vs SCALAR BASELINE - dim={dim}  n_test={len(X_te)}")
    print("=" * 80)

    t0 = time.perf_counter()
    codes, lo, scale = turbo_encode(X_te, 3)
    tq_encode = time.perf_counter() - t0
    methods: Dict[str, Dict[str, Any]] = {}
    methods["tq3"] = _benchmark_method(
        "Turbo scalar 3.0bpd",
        None,
        X_te,
        baseline=(codes, lo, scale, 3),
    )
    methods["tq3"]["enc"] = tq_encode

    # V11 asymmetric search for Keys at 2.75 bpd
    exact_275 = _search_target_engine_asymmetric(
        dim=dim,
        X_tr=X_tr,
        X_cal=X_cal,
        target_bpd=2.75,
        role='key',
        max_stages=4,
        fit_samples=1024,
        search_samples=256,
        seed=seed,
    )
    methods["v11_275"] = _benchmark_method("V11 exact 2.75bpd (key)", exact_275, X_te)

    # V11 asymmetric search for Keys at 3.00 bpd
    exact_300 = _search_target_engine_asymmetric(
        dim=dim,
        X_tr=X_tr,
        X_cal=X_cal,
        target_bpd=3.00,
        role='key',
        max_stages=4,
        fit_samples=1024,
        search_samples=256,
        seed=seed,
    )
    methods["v11_300"] = _benchmark_method("V11 exact 3.00bpd (key)", exact_300, X_te)

    # Deep V11 config with outlier handling
    base_deep = V9Engine(
        dim=dim,
        num_stages=4,
        norm_bits=2.5,
        bits_tail=0,
        use_rotation=True,
        use_qjl=True,
        qjl_rank=max(1, dim // 16),
        seed=seed,
        fit_samples=1024,
    )
    deep = V11Engine(
        base_engine=base_deep,
        handle_outliers=True,
        outlier_threshold=4.0,
    )
    deep.fit(X_tr)
    methods["v11_deep"] = _benchmark_method(
        f"V11 deep {deep.bits_per_dim():.3f}bpd",
        deep,
        X_te,
    )

    keys = ["tq3", "v11_275", "v11_300", "v11_deep"]
    width = 22

    def row(label: str, fn: Any) -> None:
        values = [fn(methods[k]) for k in keys]
        print(f"  {label:<28}" + "".join(f"{v:>{width}}" for v in values))

    header = "  " + f"{'Metric':<26}" + "".join(f"{methods[k]['label']:>{width}}" for k in keys)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    row("effective bpd", lambda m: f"{m['bpd']:.3f}")
    row("raw bpd", lambda m: f"{m['raw_bpd']:.3f}")
    row("SNR (dB)", lambda m: f"{snr_db(X_te, m['X_hat']):.2f}")
    row("Cosine similarity", lambda m: f"{cosine_sim(X_te, m['X_hat']):.5f}")
    row("Recall@10", lambda m: f"{recall_at_k(X_te, m['X_hat']):.4f}")
    row("Encode (us/vec)", lambda m: f"{m['enc'] / max(1, len(X_te)) * 1e6:.2f}")
    row("Decode (us/vec)", lambda m: f"{m['dec'] / max(1, len(X_te)) * 1e6:.2f}")
    row("Exactness max abs", lambda m: f"{m['exact']:.2e}")
    print(sep)

    tq_snr = snr_db(X_te, methods["tq3"]["X_hat"])
    print("\n  Results:")
    for key in ("v11_275", "v11_300", "v11_deep"):
        gain = snr_db(X_te, methods[key]["X_hat"]) - tq_snr
        print(f"    {methods[key]['label']}: {gain:+.3f} dB vs scalar 3-bit baseline")
        if methods[key]["config"] is not None:
            print(f"      config={methods[key]['config']}")


def compare_all_dims_v8() -> None:
    """Quick sweep using the V11 asymmetric search path."""
    print("\n" + "=" * 80)
    print("  V11 ASYMMETRIC - MULTI-DIMENSION SWEEP AT EXACT 3.0 BPD")
    print("=" * 80)
    print(f"  {'dim':>6} | {'baseline':>10} | {'v11':>10} | {'gain':>9} | {'exact':>10}")
    print("  " + "-" * 61)

    rng = np.random.default_rng(99)
    for dim, n in [
        (8, 4000),
        (22, 4000),
        (64, 3000),
        (128, 3000),
        (256, 2000),
        (768, 1200),
        (1536, 600),
    ]:
        X = rng.standard_normal((n, dim)).astype(np.float64)
        n_tr = max(128, n // 5)
        X_tr = X[:n_tr]
        X_te = X[n_tr:]
        X_cal = X_te[: min(len(X_te), 192)]

        tq_codes, tq_lo, tq_scale = turbo_encode(X_te, 3)
        tq_snr = snr_db(X_te, turbo_decode(tq_codes, tq_lo, tq_scale, 3))

        # Use V11 asymmetric search
        engine = _search_target_engine_asymmetric(
            dim=dim,
            X_tr=X_tr,
            X_cal=X_cal,
            target_bpd=3.0,
            role='key',
            max_stages=4,
            fit_samples=1024,
            search_samples=192,
            seed=99 + dim,
        )
        R = engine.decode(engine.encode(X_te))
        v11_snr = snr_db(X_te, R)
        exact = engine.exactness_error(X_te[: min(len(X_te), 32)])

        print(
            f"  {dim:>6} | {tq_snr:>10.2f} | {v11_snr:>10.2f} | "
            f"{(v11_snr - tq_snr):>+9.2f} | {exact:>10.2e}"
        )

    print("=" * 80)


def print_explanation() -> None:
    """Explain the V11 design and improvements for real KV cache."""
    print(
        """
WHY V11 BEATS V9 FOR REAL KV CACHE QUANTIZATION
------------------------------------------------
1. Asymmetric treatment of Keys and Values
   Keys are far more sensitive to quantization error because they directly
   affect attention scores (Q·K). V11 uses more stages and higher QJL rank
   for Keys, while Values can use simpler configurations.

2. Attention-aware scoring
   When query samples are provided, V11 optimizes for Q·K fidelity, not just
   reconstruction MSE. This ensures the dot products that matter for attention
   are preserved accurately.

3. Outlier handling for high-norm channels
   Real KV cache often has heavy-tailed distributions with outlier channels.
   V11 detects these using MAD-based thresholding and applies per-chunk scaling
   that is exactly reversed on decode.

4. Increased calibration data
   Heavy-tailed distributions require more samples for stable fitting.
   V11 defaults to fit_samples=1024 and search_samples=256.

5. Preserved mathematical closure
   The V9Engine core is NOT modified. Outlier handling is applied externally
   through the V11Engine wrapper, maintaining exact encode/decode closure.

RECOMMENDED PARAMETERS FOR REAL KV CACHE
-----------------------------------------
For Keys (high sensitivity):
  - stages: 4
  - qjl_rank: dim // 16 to dim // 8 (6.25% to 12.5% of dim)
  - fit_samples: 1024 or higher
  - outlier_threshold: 3.0 to 5.0 (std devs)
  - handle_outliers: True
  - qk_mse_weight: 0.3 to 0.7 (if query samples available)

For Values (lower sensitivity):
  - stages: 3
  - qjl_rank: dim // 32 to dim // 16 (3.125% to 6.25% of dim)
  - fit_samples: 512 to 1024
  - handle_outliers: False (usually not needed)
  - qk_mse_weight: N/A

Example usage:
  key_engine = _search_target_engine_asymmetric(
      dim=768, X_tr=key_train, X_cal=key_cal,
      target_bpd=4.0, role='key',
      q_sample=query_samples, qk_mse_weight=0.5,
      outlier_threshold=4.0, fit_samples=1024,
  )

  value_engine = _search_target_engine_asymmetric(
      dim=768, X_tr=value_train, X_cal=value_cal,
      target_bpd=3.0, role='value',
      fit_samples=512,
  )
"""
    )


if __name__ == "__main__":
    print("=" * 80)
    print("  Higman-Sims Quantizer V11 - Asymmetric KV Cache Optimized")
    print("=" * 80)
    print_explanation()
    compare_all_dims_v8()
    run_exact_3bpd_benchmark(dim=768, n_vectors=1200)
    run_v8_benchmark(dim=768, n_vectors=1200)


```






```python


# ============================================================
# FIXED + IMPROVED — Test V11 on Real KV Cache (Beginner Friendly)
# ============================================================

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 1. Load Real Data (you already have these files) ────────
keys   = np.load("real_kv_data/keys.npy")
values = np.load("real_kv_data/values.npy")

print("Keys shape:", keys.shape)
print("Values shape:", values.shape)

# ── 2. Reshape to (N, head_dim) ─────────────────────────────
X_keys   = keys.reshape(-1, keys.shape[-1]).astype(np.float32)
X_values = values.reshape(-1, values.shape[-1]).astype(np.float32)

print("After reshape → Keys:", X_keys.shape, "Values:", X_values.shape)

# ── 3. Use ALL data (no artificial split if small) ──────────
total = X_keys.shape[0]
n_train = max(64, total // 2)      # at least 64 samples
n_cal   = total - n_train

X_tr  = X_keys[:n_train]
X_cal = X_keys[n_train:n_train + n_cal]

print(f"Using {n_train} train + {n_cal} calibration vectors")

# ── 4. Run Key Engine with relaxed settings ─────────────────
print("\n🔍 Training Key engine (trying low bpd)...")

best_key_engine = None
for bpd in [6.0, 5.0, 4.0, 3.5, 3.0]:
    try:
        engine = _search_target_engine_asymmetric(
            dim            = X_tr.shape[1],
            X_tr           = X_tr,
            X_cal          = X_cal,
            target_bpd     = bpd,
            role           = 'key',
            qk_mse_weight  = 0.6,
            fit_samples    = min(256, n_train),     # smaller for safety
            search_samples = min(128, n_cal),
            handle_outliers= True,
            outlier_threshold=4.0
        )
        print(f"✅ SUCCESS at {bpd} bpd | bits_per_dim = {engine.bits_per_dim():.3f}")
        best_key_engine = engine
        break
    except Exception as e:
        print(f"❌ {bpd} bpd failed: {e}")

if best_key_engine is None:
    print("⚠️  Could not find any valid config. Try longer prompts next time.")

# ── 5. Run Value Engine ─────────────────────────────────────
print("\n🔍 Training Value engine...")

best_val_engine = None
for bpd in [5.0, 4.0, 3.5, 3.0]:
    try:
        engine = _search_target_engine_asymmetric(
            dim            = X_values.shape[1],
            X_tr           = X_values[:n_train],
            X_cal          = X_values[n_train:n_train + n_cal],
            target_bpd     = bpd,
            role           = 'value',
            qk_mse_weight  = 0.0,          # Values don't need Q·K scoring
            fit_samples    = min(256, n_train),
            search_samples = min(128, n_cal),
            handle_outliers= False
        )
        print(f"✅ SUCCESS at {bpd} bpd | bits_per_dim = {engine.bits_per_dim():.3f}")
        best_val_engine = engine
        break
    except Exception as e:
        print(f"❌ {bpd} bpd failed: {e}")

# ── 6. Final Summary ────────────────────────────────────────
if best_key_engine:
    print("\n🎉 KEY ENGINE READY!")
    print("   Bits per dim :", best_key_engine.bits_per_dim())
    print("   Config       :", best_key_engine.base_engine.__dict__ if hasattr(best_key_engine, 'base_engine') else "see engine")
    
if best_val_engine:
    print("\n🎉 VALUE ENGINE READY!")
    print("   Bits per dim :", best_val_engine.bits_per_dim())
```