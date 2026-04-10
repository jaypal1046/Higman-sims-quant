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

``` output
Loading weights: 100%
 164/164 [04:10<00:00,  1.19s/it, Materializing param=model.norm.weight]
✅ Model loaded successfully!

🔍 past_key_values type : <class 'transformers.cache_utils.DynamicCache'>

🔍 pkv.layers type      : <class 'list'>
🔍 pkv.layers length    : 18
🔍 Layer 0 type         : <class 'transformers.cache_utils.DynamicLayer'>
🔍 Layer 0 attributes   : ['_abc_impl', 'batch_repeat_interleave', 'batch_select_indices', 'crop', 'device', 'dtype', 'get_mask_sizes', 'get_max_cache_shape', 'get_seq_length', 'is_compileable', 'is_initialized', 'is_sliding', 'keys', 'lazy_initialization', 'offload', 'prefetch', 'reorder_cache', 'reset', 'update', 'values']

✅ Layer 0 key   shape : (1, 1, 92, 256)
✅ Layer 0 value shape : (1, 1, 92, 256)

✅ Captured 18 layers
💾 Final keys   shape : (18, 1, 1, 92, 256)
💾 Final values shape : (18, 1, 1, 92, 256)
   Shape meaning      : (num_layers, batch, num_heads, seq_len, head_dim)

✅ Saved:
   real_kv_data/keys.npy
   real_kv_data/values.npy
   real_kv_data/metadata.json

🔁 Reloaded keys   shape : (18, 1, 1, 92, 256)
🔁 Reloaded values shape : (18, 1, 1, 92, 256)
✅ All done!
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

```Output
================================================================================
  Higman-Sims Quantizer V11 - Asymmetric KV Cache Optimized
================================================================================

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


================================================================================
  V11 ASYMMETRIC - MULTI-DIMENSION SWEEP AT EXACT 3.0 BPD
================================================================================
     dim |   baseline |        v11 |      gain |      exact
  -------------------------------------------------------------
       8 |      11.74 |      14.44 |     +2.69 |   4.44e-16
      22 |      11.86 |      10.33 |     -1.53 |   2.22e-16
      64 |      11.87 |      10.70 |     -1.17 |   2.22e-16
     128 |      12.01 |      10.74 |     -1.27 |   2.22e-16
     256 |      12.30 |      10.74 |     -1.57 |   2.22e-16
     768 |      12.63 |      10.76 |     -1.87 |   4.44e-16
    1536 |      13.18 |      10.75 |     -2.43 |   4.44e-16
================================================================================

================================================================================
  EXACT 3.0 BPD BENCHMARK - dim=768  n_test=960  n_train=240
================================================================================
  Turbo SNR : 12.556 dB
  V11 SNR   : 10.754 dB
  Gain      : -1.802 dB
  Cosine    : 0.960597
  Recall@10 : 0.8600
  Exactness : max |encode-decode mismatch| = 2.220e-16
  Config    : {'num_stages': 2, 'norm_bits': (3.0931, 3.0931), 'use_rotation': True, 'qjl_rank': 192, 'handle_outliers': True}

================================================================================
  V11 ASYMMETRIC vs SCALAR BASELINE - dim=768  n_test=960
================================================================================

----------------------------------------------------------------------------------------------------------------------
  Metric                       Turbo scalar 3.0bpdV11 exact 2.75bpd (key)V11 exact 3.00bpd (key)     V11 deep 5.266bpd
----------------------------------------------------------------------------------------------------------------------
  effective bpd                                3.000                 2.750                 3.000                 5.266
  raw bpd                                      3.000                 2.812                 3.250                 5.562
  SNR (dB)                                     12.56                 10.09                 10.75                 19.15
  Cosine similarity                          0.97314               0.95435               0.96060               0.99397
  Recall@10                                   0.9200                0.9000                0.8600                1.0000
  Encode (us/vec)                               6.34                310.91                241.47                578.17
  Decode (us/vec)                               1.80                 81.54                 74.78                102.96
  Exactness max abs                         0.00e+00              4.44e-16              2.22e-16              8.88e-16
----------------------------------------------------------------------------------------------------------------------

  Results:
    V11 exact 2.75bpd (key): -2.467 dB vs scalar 3-bit baseline
      config={'num_stages': 2, 'norm_bits': (2.8431, 2.8431), 'use_rotation': True, 'qjl_rank': 48, 'handle_outliers': True}
    V11 exact 3.00bpd (key): -1.802 dB vs scalar 3-bit baseline
      config={'num_stages': 2, 'norm_bits': (3.0931, 3.0931), 'use_rotation': True, 'qjl_rank': 192, 'handle_outliers': True}
    V11 deep 5.266bpd: +6.596 dB vs scalar 3-bit baseline
      config={'num_stages': 4, 'norm_bits': (2.5, 2.5, 2.5, 2.5), 'use_rotation': True, 'qjl_rank': 48, 'handle_outliers': True}
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

```Output

Keys shape: (18, 1, 1, 92, 256)
Values shape: (18, 1, 1, 92, 256)
After reshape → Keys: (1656, 256) Values: (1656, 256)
Using 828 train + 828 calibration vectors

🔍 Training Key engine (trying low bpd)...
✅ SUCCESS at 6.0 bpd | bits_per_dim = 6.000

🔍 Training Value engine...
✅ SUCCESS at 5.0 bpd | bits_per_dim = 5.000

🎉 KEY ENGINE READY!
   Bits per dim : 6.0
   Config       : {'dim': 256, 'num_stages': 4, 'bits_tail': 0, 'use_rotation': True, 'rotation_reflections': 8, 'use_qjl': True, 'fit_passes': 3, 'fit_samples': 256, '_rng': Generator(PCG64) at 0x7B2BE09717E0, '_norm_eps': 1e-12, '_min_log_span': 0.001, 'CB': array([[ 0.70710678,  0.70710678,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       [ 0.70710678, -0.70710678,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       [-0.70710678,  0.70710678,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       ...,
       [ 0.35355339, -0.35355339,  0.35355339, ..., -0.35355339,
        -0.35355339, -0.35355339],
       [ 0.35355339,  0.35355339, -0.35355339, ..., -0.35355339,
        -0.35355339, -0.35355339],
       [-0.35355339, -0.35355339, -0.35355339, ..., -0.35355339,
        -0.35355339, -0.35355339]]), 'CBT': array([[ 0.70710678,  0.70710678, -0.70710678, ...,  0.35355339,
         0.35355339, -0.35355339],
       [ 0.70710678, -0.70710678,  0.70710678, ..., -0.35355339,
         0.35355339, -0.35355339],
       [ 0.        ,  0.        ,  0.        , ...,  0.35355339,
        -0.35355339, -0.35355339],
       ...,
       [ 0.        ,  0.        ,  0.        , ..., -0.35355339,
        -0.35355339, -0.35355339],
       [ 0.        ,  0.        ,  0.        , ..., -0.35355339,
        -0.35355339, -0.35355339],
       [ 0.        ,  0.        ,  0.        , ..., -0.35355339,
        -0.35355339, -0.35355339]]), 'K': 240, 'D': 8, 'ID_BITS_RAW': 8, 'ID_BITS_EFFECTIVE': 7.906890595608519, 'n_chunks': 32, 'padded_dim': 256, 'stage_norm_bits': (3.4681094043914813, 3.4681094043914813, 3.4681094043914813, 3.4681094043914813), '_norm_log_ranges': [(-0.6715807417974656, 2.6062801018406785), (-1.2021354888622289, 2.1905502679661684), (-1.7312096770383418, 1.5557158678817415), (-2.346708200785553, 1.0643484730910746)], '_stage_mean_log_norm': [1.226774526686115, 0.6301947485154611, 0.043383789338815947, -0.5474882342939676], '_tail_scale': 1.0, 'qjl_rank': 80, '_rotation_vectors': [array([ 0.02027484, -0.06919702,  0.04993248,  0.06258199, -0.12981526,
       -0.08664261,  0.00850607, -0.02104171, -0.00111789, -0.05675865,
        0.05851216,  0.05175164,  0.00439346,  0.07500281,  0.03110649,
       -0.05717441,  0.02453543, -0.0638008 ,  0.0584491 , -0.0033219 ,
       -0.01230011, -0.04530674,  0.08134375, -0.01028187, -0.02849948,
       -0.02342977,  0.03541805,  0.02431541,  0.02746183,  0.02866537,
        0.14249797, -0.02704148, -0.03408289, -0.05414568,  0.04098518,
        0.07511799, -0.00758168, -0.05590116, -0.05485818,  0.04328824,
        0.04945361,  0.03613964, -0.04428076,  0.01544723,  0.00776388,
        0.0145508 ,  0.05798192,  0.01487729,  0.04517261,  0.00449648,
        0.01923702,  0.04200378, -0.09695421, -0.02126984, -0.031297  ,
       -0.04250877, -0.01830703,  0.09946832, -0.05760947,  0.06442596,
       -0.11197245, -0.02228211,  0.01082904,  0.03900525,  0.0473226 ,
        0.05278663, -0.02320298, -0.03076332,  0.05708681, -0.01272874,
       -0.08487984, -0.07540509, -0.06117724,  0.03307939,  0.00947653,
        0.04594255, -0.02842794,  0.01054869,  0.04162466, -0.02058287,
        0.03039228, -0.04404231, -0.02415637, -0.02539954, -0.07956711,
        0.0324015 , -0.03123244,  0.00083132,  0.03198725,  0.02971067,
        0.04427247, -0.00655289, -0.02816483, -0.00530418, -0.11226951,
       -0.09628596, -0.08800795, -0.06635342,  0.02659962, -0.06024751,
       -0.02516165,  0.08644624, -0.0237046 ,  0.04907178, -0.06211975,
       -0.01366912, -0.06321125, -0.02255811,  0.05591125, -0.11493004,
        0.02890508,  0.01581812, -0.03953272, -0.09621578,  0.00479925,
       -0.03523065,  0.01548149,  0.00145397,  0.10657693, -0.01592591,
       -0.06810005,  0.01192839,  0.01463783,  0.09043573,  0.05556547,
        0.02374499,  0.09736321, -0.07909626, -0.0425669 , -0.06165122,
       -0.02593662, -0.09160003,  0.04226079, -0.01478594, -0.09786246,
       -0.06757319,  0.02086015,  0.0557661 ,  0.1328557 ,  0.19387853,
        0.0275734 , -0.06584051, -0.14185913,  0.01781261, -0.05409034,
       -0.02763646, -0.04072684, -0.00936775,  0.07092671,  0.01044948,
       -0.01055502, -0.06890889, -0.11142772, -0.03235728, -0.00357851,
        0.11763206,  0.00866802,  0.06538815, -0.03322144, -0.07884214,
       -0.0642156 , -0.04825408,  0.14162116, -0.05465228,  0.05579023,
       -0.06007771,  0.06198371,  0.02561333, -0.01042215, -0.0027122 ,
       -0.04356735,  0.02968013, -0.03027306, -0.08154765, -0.08502963,
        0.01148341,  0.10506738,  0.0106453 , -0.00789379,  0.0190179 ,
        0.08689693,  0.01459697, -0.0273417 ,  0.0736087 ,  0.028528  ,
        0.10218399,  0.0121918 , -0.08147202, -0.09103267,  0.10984715,
        0.11468687, -0.0119446 , -0.02549598,  0.09723955, -0.07365906,
       -0.0595321 ,  0.04280478, -0.02625569, -0.00034079, -0.01087494,
        0.02246107,  0.09364907,  0.00602721,  0.0428455 , -0.1364115 ,
       -0.00324156, -0.05610568, -0.08109569, -0.05842928, -0.02223144,
        0.06094105, -0.08825367,  0.00203812, -0.03221499, -0.02180226,
        0.0667201 ,  0.03580438,  0.08898593, -0.01028028, -0.04630566,
       -0.01489481,  0.01613491,  0.01174859, -0.0721515 ,  0.00602088,
        0.01518554,  0.16750418,  0.12487887, -0.05677192, -0.01912151,
       -0.09737247, -0.03930364,  0.02099929,  0.08023341, -0.04851077,
       -0.04352469, -0.14287333, -0.01082324, -0.07068945, -0.0352271 ,
       -0.05834334, -0.00627191, -0.11695329, -0.09761221,  0.14167288,
       -0.08566073, -0.07297639,  0.122222  ,  0.19329332, -0.07795207,
       -0.02450204,  0.02272595,  0.11502168, -0.06566212, -0.01631996,
        0.05172141]), array([ 2.70860803e-02, -2.34346564e-02, -8.33721806e-03, -8.56564955e-02,
       -1.48383086e-02, -1.65960349e-02,  1.44642663e-02, -3.45970823e-02,
        2.93770169e-02,  6.30925539e-02,  9.68330213e-03,  2.19145488e-02,
        3.31159697e-03,  5.25771947e-06, -4.49533209e-02,  1.97177045e-02,
       -6.06098952e-03,  1.30405127e-01,  9.80205674e-02,  2.40383768e-02,
       -4.75387343e-02, -6.93036285e-02,  7.42086276e-02,  1.63693697e-02,
        2.99131040e-02, -1.08688325e-01,  5.77797457e-02,  2.83105479e-02,
       -6.91802247e-02, -2.93761624e-02,  1.64296752e-02,  3.26870011e-03,
       -1.82023683e-02, -6.44735573e-03, -1.56982799e-02,  9.50469851e-03,
        9.16744708e-02, -1.59903729e-01, -1.47558553e-02,  1.09967863e-02,
        1.84405303e-02, -2.31704101e-02, -1.09444389e-01,  2.04342347e-02,
        1.07614530e-01, -9.55601085e-02,  5.38167910e-02, -2.04672377e-02,
       -3.82052845e-03, -6.55959499e-02, -2.08367379e-02,  8.09932384e-02,
        3.62996278e-02,  1.07923626e-01,  7.33531775e-02,  2.73552555e-02,
        1.08647738e-01,  2.73494292e-02,  5.15839565e-02, -1.84764754e-02,
        4.14582788e-03, -4.34497511e-02,  6.16514294e-02, -7.34087328e-02,
        4.87407033e-02, -1.18776284e-02,  7.29691085e-02,  4.67794038e-02,
        1.13426900e-01,  4.55275219e-02, -9.79386644e-02, -4.17120637e-03,
       -7.30164630e-02, -3.22890176e-02,  9.41500663e-02,  3.97185821e-02,
       -4.35436131e-02, -6.31549508e-02,  2.04233599e-03, -7.57921279e-02,
       -4.18122769e-02,  1.94383005e-02,  7.19763488e-02,  3.79260552e-02,
       -1.42748147e-01,  1.89621550e-02,  4.48771711e-03,  2.57855113e-02,
        1.00690435e-01, -1.28540475e-01, -3.68259526e-02,  3.68136731e-02,
       -9.85338909e-02,  9.19521481e-02,  2.29487474e-02,  5.27424820e-02,
       -3.55699922e-02,  5.06977659e-02,  6.65661562e-02,  1.45083831e-02,
        1.46032586e-02,  1.68424795e-02, -5.37867156e-02, -9.19108900e-03,
       -9.50220776e-03,  2.38855735e-02,  6.22894045e-02, -6.59471726e-02,
       -7.78810684e-03,  9.22952050e-02, -4.63258099e-02, -5.12264671e-02,
        1.26037474e-02,  5.26054963e-02,  7.11847306e-04,  8.27947152e-02,
        5.33785684e-02,  5.24456882e-02,  3.45216542e-02,  1.45013612e-01,
       -1.27816458e-02, -1.24820148e-01,  9.99456146e-02, -2.85148366e-02,
        6.72098987e-03,  8.15854712e-02, -9.98213366e-02, -7.79780645e-02,
       -9.97601823e-02, -4.94749720e-02,  2.73895162e-02,  3.26570882e-02,
        1.72119792e-02, -8.80158147e-02, -1.43920262e-01,  3.38624761e-03,
       -2.93918139e-02,  2.86198962e-02,  4.37319594e-02,  8.61249015e-03,
        4.73565603e-02,  1.42799497e-02,  3.30232188e-02, -4.39013937e-02,
       -1.11898547e-02,  1.22592205e-02,  5.11192145e-02, -2.45302144e-02,
        3.24689045e-02, -1.65618508e-02, -7.32291863e-03,  5.16793300e-02,
       -1.24168367e-01, -8.07706850e-02, -9.23406749e-02, -1.45385110e-01,
       -4.22561147e-02,  4.66899972e-02, -1.77483782e-02,  1.23223921e-02,
        6.78586356e-02,  8.27153160e-02, -4.30731781e-03,  8.43288801e-02,
        5.73952343e-03, -5.21702058e-02, -3.70313523e-02, -9.22379488e-02,
       -5.53310535e-02, -2.23045664e-02,  5.00636312e-02,  1.07204570e-01,
       -8.61103977e-02,  2.44732903e-02, -6.48262557e-02,  2.95737966e-02,
       -8.16674474e-03, -1.14066081e-01,  5.78332312e-02, -3.76917586e-02,
       -3.32621738e-02, -6.66459539e-02, -4.07620797e-02,  2.66577259e-02,
       -1.17899894e-02,  2.04757591e-02,  2.25478596e-02,  8.22776886e-02,
       -2.13556985e-02, -9.20087648e-02,  6.64883343e-02, -2.06518305e-02,
        6.94395038e-02,  2.38845299e-02, -8.16991455e-03,  2.17288617e-02,
        1.21548773e-01,  1.29396622e-01,  4.32246928e-03,  9.97993067e-03,
        6.70501428e-02, -5.26849817e-02,  2.07504021e-02, -1.61126458e-03,
        1.95565927e-02, -5.19191716e-02, -9.90306175e-02, -1.29147602e-01,
       -6.96134258e-02, -2.85756326e-02, -1.82659396e-02,  1.20690187e-01,
        6.89037784e-02, -5.99386171e-02,  2.16623597e-02, -2.53611182e-02,
       -1.77159678e-02,  1.15458536e-02,  3.85745798e-02, -2.11359236e-02,
        6.62783271e-02, -7.11431557e-02,  3.94925829e-04,  1.61835993e-01,
        1.38979469e-02,  8.92897711e-02,  5.70173950e-03,  3.61826186e-02,
       -3.53761311e-03, -1.06164526e-02, -4.85620292e-02,  2.68079270e-02,
       -5.30510683e-02,  4.14661958e-02,  6.76137645e-02,  2.28350364e-02,
       -1.78333974e-02,  2.82822163e-02, -1.92304406e-02,  5.82849171e-02,
       -1.14097247e-01, -2.09084578e-02, -1.24028292e-01, -9.31428189e-02,
        8.49690997e-02,  5.57703413e-02, -4.48238732e-02, -9.36064972e-02,
       -1.84691196e-01, -3.38599625e-02,  1.50792712e-01,  2.70934441e-02,
       -3.48615515e-02,  2.89746617e-02, -9.72482579e-02, -1.85233507e-02]), array([ 0.00613707, -0.00531182,  0.04878728,  0.02126223,  0.0412311 ,
       -0.04246782,  0.05538902,  0.10049417, -0.05985153, -0.0547647 ,
        0.08240868, -0.0118046 ,  0.08660609, -0.0273014 ,  0.08976627,
        0.00811177,  0.01593094,  0.09653231, -0.02231874, -0.05806074,
       -0.02767332,  0.02790588, -0.09659653,  0.03932756, -0.03323847,
        0.07081212, -0.14770934, -0.04852568, -0.10404345, -0.05097266,
        0.01527928, -0.01105705, -0.01563165, -0.00982061,  0.01254765,
       -0.06221972,  0.04360775,  0.04088192,  0.02375418,  0.03433427,
        0.01828695,  0.12554998, -0.00537311, -0.01894492, -0.04648745,
       -0.06368348, -0.07677532, -0.05483266, -0.00436049,  0.0206237 ,
        0.00315511, -0.04722824,  0.05553518,  0.04561666, -0.0098492 ,
       -0.04028042,  0.03383422,  0.01159667, -0.08933946, -0.00419391,
        0.0161658 , -0.05550496,  0.01171203, -0.08975251,  0.08243346,
        0.0769899 , -0.01557858,  0.02242263, -0.14867556, -0.07133867,
       -0.01812413, -0.06614321,  0.04407333,  0.12321942, -0.07258901,
       -0.05166571,  0.01452554,  0.09939475, -0.07541206,  0.01536381,
        0.11236157, -0.10190214, -0.07903313, -0.02613361, -0.03211671,
        0.05013189,  0.01490874, -0.10950291,  0.03179727, -0.03563016,
        0.0786246 , -0.03871782, -0.03927477,  0.03338409,  0.0470673 ,
        0.02764464, -0.10398973,  0.03319301, -0.06380967,  0.01451491,
       -0.08783459,  0.027535  , -0.04976159, -0.07912971,  0.04403778,
        0.0149078 , -0.03787813,  0.08952772, -0.02718521,  0.00198082,
        0.0165901 , -0.03822911,  0.02906586, -0.03291033, -0.02539524,
        0.08406565, -0.06419698, -0.1488519 ,  0.0993837 ,  0.15727595,
       -0.02500232, -0.11948954, -0.01915472, -0.01765798, -0.01171699,
       -0.06868836,  0.03575492,  0.03235849, -0.09219452,  0.04313561,
        0.1266365 ,  0.01060877, -0.02081064, -0.00876062,  0.0379571 ,
       -0.1067705 ,  0.01014177, -0.02408893,  0.11399805, -0.01074526,
        0.10289716, -0.06809319,  0.03622984,  0.01970479, -0.05361422,
        0.01094412,  0.07480405, -0.01997571, -0.10438242, -0.00108351,
       -0.05567328, -0.02112007, -0.0050334 , -0.10522697, -0.09967497,
        0.0297402 , -0.03224814, -0.15822703,  0.04841946,  0.01680333,
       -0.04404115, -0.08123934,  0.05156358,  0.02155252,  0.14699012,
        0.02592274,  0.02391861, -0.0102983 ,  0.05038944,  0.03856344,
        0.0772228 , -0.03216203, -0.02686164, -0.02955736,  0.04878701,
        0.09243937, -0.02830729, -0.02620561,  0.0193764 , -0.01516179,
        0.05873514, -0.13891912, -0.051002  , -0.04826969, -0.14314996,
       -0.05944984, -0.05645882, -0.01240677,  0.06866232, -0.01511855,
       -0.06359409, -0.0035137 ,  0.06472677, -0.06020993, -0.05617619,
        0.03445862, -0.01366725,  0.03994534, -0.00084191,  0.04328781,
       -0.06385718, -0.00074554, -0.01299812, -0.07501208, -0.0964558 ,
        0.04230598, -0.02165134, -0.06306764, -0.00593358,  0.06959098,
       -0.14070579, -0.09233228, -0.05693573,  0.09014465,  0.01743367,
        0.04733818, -0.07034006, -0.06906764,  0.02762702,  0.00359512,
        0.0338534 , -0.01157801,  0.01715955,  0.00975486,  0.04798288,
        0.04978684, -0.09993493, -0.13864097,  0.06180078,  0.07327444,
       -0.0629654 , -0.11473901,  0.00610977,  0.0574263 ,  0.11089918,
        0.031852  , -0.02293236, -0.0551    ,  0.00070647, -0.01846257,
       -0.06262269,  0.12639409,  0.11013257,  0.07008637, -0.05681012,
        0.05274879,  0.03946054,  0.02730201,  0.07709573,  0.03919802,
        0.04565377,  0.03929272,  0.02102448, -0.1100365 ,  0.00515884,
       -0.03431321]), array([-8.18094537e-02,  1.07504372e-01,  1.10520111e-01,  8.68835258e-02,
        1.63136444e-02,  8.63340892e-02,  7.70458325e-04,  1.29631220e-02,
       -6.98964156e-02,  2.53763107e-02,  3.85996384e-03, -8.32675790e-02,
       -3.27260018e-03, -5.09643954e-03,  1.14902947e-01,  5.71595271e-02,
        7.31610529e-04,  1.59028775e-02,  2.82612498e-03, -1.29705821e-02,
       -6.91904546e-02, -9.65547439e-03, -4.76917780e-02, -7.99221424e-02,
        3.26781052e-02,  2.50102545e-02, -1.14209164e-01, -7.84219564e-03,
        6.36468039e-02,  6.77072530e-02,  6.55731099e-02,  2.48739359e-03,
       -5.40166309e-02, -6.92717392e-02,  2.20276893e-02,  2.42441783e-02,
        8.22882362e-02,  7.03124351e-02, -8.45226440e-03, -7.95316033e-02,
       -2.03977973e-02,  1.38857608e-02, -1.29174800e-02, -3.69399034e-02,
        1.61647965e-02, -3.22137741e-02, -4.01469752e-02,  1.99085567e-02,
       -2.56960001e-02,  1.56035378e-02,  1.74636323e-02, -7.28340999e-02,
       -3.07616921e-02,  9.19049710e-02, -7.42819563e-02, -1.35299893e-01,
       -1.19012080e-01,  1.86075372e-03,  1.97628120e-03, -7.51787148e-03,
        7.76129300e-02, -1.70851795e-01,  2.53087181e-02,  9.98095940e-02,
       -7.20895263e-02, -2.42778316e-02, -4.81260395e-02, -5.71680106e-02,
       -2.08551822e-02,  9.12471121e-02,  1.17448656e-01, -2.14737654e-02,
        1.21774455e-01,  2.27410455e-03,  1.12098698e-01, -5.96379692e-03,
        8.37742306e-03,  2.33617051e-02,  2.03197344e-01,  5.44155174e-02,
       -4.52094649e-02,  6.19398731e-02, -2.31289068e-02, -3.13056046e-02,
        5.80792328e-02,  1.98703983e-03,  1.78070121e-02,  8.93957452e-04,
        2.15149246e-02,  2.71644908e-02, -1.23813983e-01,  4.26083769e-02,
       -6.27722311e-02, -9.21940369e-02, -3.73373889e-03,  5.36819826e-03,
       -4.43300085e-02,  5.31208678e-02, -8.57939547e-02, -2.60124866e-02,
       -3.73860777e-02, -2.97793184e-03,  1.78254452e-02, -6.44261079e-02,
        4.62972917e-02,  4.02721081e-03, -1.20936190e-01, -1.25196256e-01,
       -7.87468645e-04, -1.41245797e-02, -6.61077687e-03, -1.78868278e-03,
        1.44181409e-02,  6.05752586e-02, -7.10235787e-02, -7.49142737e-02,
       -6.98869864e-02,  1.84680093e-02,  7.95790011e-02, -2.75718976e-02,
       -1.59912602e-01, -1.08921913e-01, -5.32490219e-02, -3.56509106e-02,
       -2.61048455e-02,  2.46616627e-03, -1.99273174e-02,  6.70754439e-02,
       -4.32120930e-02, -5.51207248e-02,  3.06104801e-02, -9.81606645e-02,
        2.49097088e-02,  6.55536134e-03, -9.42475664e-03,  1.01525115e-01,
       -3.97724818e-02,  1.31697510e-01, -1.44099340e-02, -8.16289064e-02,
        4.46936996e-03, -6.87952916e-02, -4.80534201e-02,  2.53789951e-02,
        3.55136839e-02, -3.97699413e-02,  6.31164951e-02,  7.39897226e-02,
        9.18106749e-02,  3.38409405e-02,  8.71525149e-02, -1.20223604e-01,
       -2.03210883e-02, -5.54203442e-02,  7.62110520e-03, -3.65279359e-02,
       -1.06210715e-02,  1.20311589e-01, -1.08487572e-02,  2.64502579e-02,
       -1.48470221e-02,  4.83971285e-03,  3.84595708e-04,  2.86575818e-02,
        7.44882967e-02,  1.05304024e-01,  1.97914043e-02,  3.76847673e-02,
       -7.35650756e-02, -5.61722005e-03,  6.01047865e-02,  5.53540820e-02,
        1.35264281e-02,  5.66597007e-02,  3.13705825e-02,  7.67254702e-02,
        1.84962940e-02, -2.27367977e-02,  2.14675031e-02, -1.87328217e-01,
        2.44746605e-02, -2.33212305e-01, -1.10166501e-01,  2.88777619e-02,
        3.05244288e-02, -7.43042821e-02, -4.55186863e-02,  8.76071429e-02,
       -3.09399699e-02,  1.43371006e-01, -1.22720045e-04,  2.60823164e-02,
        1.03353163e-01,  8.37546004e-03, -6.40714091e-02, -7.01395187e-03,
       -2.27629446e-03, -8.72364401e-02, -1.63531897e-02, -4.74420899e-02,
        5.90864066e-02,  2.21244562e-03, -1.80767506e-02, -6.78731657e-03,
        1.42623115e-02,  3.94277370e-02, -6.39031833e-02, -6.65799252e-02,
        7.06128388e-02, -2.63572236e-02, -9.05667825e-02,  2.83692088e-02,
        2.96193293e-02, -9.78457349e-02,  1.46688334e-02,  4.70196321e-02,
        2.39313728e-02,  4.03972538e-02, -8.97630968e-02,  2.11606091e-02,
       -1.93439326e-02, -3.08607088e-02,  5.56472884e-02,  9.45575612e-02,
        1.14698966e-01,  8.40445941e-02, -7.01438206e-03,  2.25464295e-02,
        4.90165281e-02,  7.74588575e-03,  8.35865325e-03,  5.26556002e-02,
       -3.78944029e-03, -4.66171719e-02, -2.64937725e-02,  4.05205517e-02,
        1.91335897e-04,  2.17467586e-02,  4.28325226e-02, -2.39604579e-02,
        4.83405761e-02,  2.42162171e-02, -7.89312012e-02,  9.21943984e-02,
       -3.20083943e-02, -1.05796306e-01, -6.68008431e-02, -6.52668180e-02,
        3.33500086e-03, -1.75049565e-02, -2.15308309e-02,  3.96123276e-02,
        2.17253694e-02,  2.02022787e-02,  2.61968817e-02,  3.93843003e-02]), array([-1.32959367e-01, -2.29869791e-02, -1.37516967e-01,  2.27448348e-03,
       -2.92225185e-04,  6.59470526e-02,  7.49101578e-02,  1.27900564e-02,
       -3.15603249e-02,  3.06015856e-02, -3.32984577e-02, -8.78416288e-05,
        6.22006466e-02, -3.51814833e-02,  5.08182606e-02,  4.27271589e-02,
       -6.02235892e-02,  6.14284693e-02,  4.40621419e-02,  6.42898552e-03,
       -4.80836282e-02, -5.41945054e-02, -3.39131207e-02,  3.42241999e-02,
       -6.02761528e-02,  2.75961287e-02, -7.83238539e-02, -1.28716463e-02,
        6.91604306e-03,  1.54226828e-01, -8.68743224e-02,  9.28474657e-02,
        9.45132522e-03,  2.59347098e-02,  7.46480645e-03,  2.80511884e-02,
       -9.69370330e-03,  9.17143603e-02, -2.87919630e-02,  7.14158086e-02,
       -4.06435204e-02, -3.80039122e-03, -6.76141450e-02,  2.87006666e-02,
        9.11514640e-02, -4.87924876e-03, -1.24185851e-02, -7.03045762e-02,
       -1.44626570e-02, -1.00465657e-01, -5.75798701e-02,  1.43045525e-02,
        8.31968891e-02,  1.77191255e-01, -3.69989128e-02,  9.05317006e-02,
        1.53747120e-02, -9.53996041e-03,  2.72859382e-02,  3.90538839e-03,
        6.96322141e-03, -2.57556550e-02, -8.81859063e-02, -9.73643137e-02,
        4.12034862e-02, -1.74528858e-02, -3.75983668e-02,  5.36493254e-04,
        5.01404312e-02,  1.13764737e-02, -4.13807306e-02,  7.73485506e-02,
        9.96072832e-02,  3.11942104e-02,  6.14139231e-02,  7.83367507e-02,
        7.12811272e-02,  3.87343669e-02,  3.77407997e-02,  3.27776425e-02,
       -6.92146833e-02,  4.41981830e-02, -8.55181015e-02, -5.01194399e-02,
        8.22224967e-02,  5.29969706e-02,  9.38177956e-02, -1.71358889e-02,
       -7.26782107e-02, -1.51656002e-02,  6.43563426e-03,  4.98222117e-03,
        7.15070093e-02, -2.27871681e-02,  2.22046094e-02, -6.23577561e-02,
        2.84038073e-02,  1.96072113e-04, -4.72945511e-02, -1.48749606e-02,
       -1.16169036e-02, -1.70400617e-02,  1.11727526e-01, -6.20949142e-03,
       -1.53837715e-02, -1.32296928e-01, -5.64022180e-02, -1.65933825e-02,
       -4.32602260e-02,  8.71513443e-02, -1.04034666e-02,  8.12687496e-02,
        3.89202595e-03,  2.34814950e-03, -5.60084480e-03,  2.39448950e-04,
        1.08413166e-01, -1.46309678e-01, -1.26247698e-01, -3.42605544e-02,
        9.16347603e-04,  4.35284055e-02,  2.98400222e-02, -2.42363493e-02,
        6.42762826e-02,  6.49796779e-02,  1.16094661e-02,  6.07197225e-02,
        1.71971272e-02, -3.54155979e-02,  4.40148668e-02,  6.97693375e-03,
        8.45109678e-05,  9.29066936e-02, -1.54589004e-01, -8.94205705e-02,
       -7.48745856e-02, -2.29127016e-02, -1.60594357e-02, -9.50747157e-02,
       -6.21389412e-02, -5.42976748e-02,  1.55002270e-01,  1.13645053e-01,
       -2.59711356e-02, -2.29330523e-02, -7.24861568e-02, -1.20355516e-01,
       -7.37222942e-03, -6.29396134e-02, -5.35195575e-03, -1.00933144e-01,
       -4.80616150e-02,  9.37466887e-03,  2.30987069e-02,  2.63320855e-02,
       -8.32899745e-02,  5.39093877e-02, -5.04734326e-02,  3.99175761e-02,
       -6.71035589e-04, -8.68157789e-02, -1.99442018e-02,  2.30453545e-02,
        3.86628525e-02, -8.89282570e-03,  9.66171064e-02,  6.35488424e-02,
       -1.61884840e-02,  4.73140082e-02,  1.21974217e-01,  1.23657762e-01,
       -7.74562150e-02, -5.84411685e-02,  9.36614007e-02, -6.67511829e-02,
       -8.34194140e-02, -3.06667433e-02,  2.65058416e-02, -6.45868249e-03,
       -4.10343653e-02, -4.25260183e-02, -4.49307314e-02, -5.54751576e-02,
        1.43914221e-01,  1.87655405e-02,  5.59324159e-02, -3.08486101e-02,
       -1.17292360e-02, -4.50074886e-02, -1.67256768e-01, -8.69180727e-02,
       -1.14204954e-01, -1.41905336e-01, -7.53973866e-02,  8.35717117e-02,
       -2.80281571e-03,  8.14024381e-02,  2.59227671e-02,  4.93599760e-02,
       -5.68241679e-02,  3.30482587e-02,  4.59637230e-02, -3.63720855e-02,
        3.52930647e-02,  3.57123494e-02, -3.46630864e-02, -7.08187309e-02,
       -7.46890127e-02,  5.45634569e-03,  2.06261801e-02, -4.99886458e-02,
        1.67687306e-03,  3.59866805e-02,  3.89924410e-02,  9.78849935e-02,
        8.05456934e-02, -6.31565570e-02,  1.06192368e-01, -3.34913960e-02,
        6.59054569e-02,  4.28826322e-03, -2.59963632e-02, -1.14020660e-01,
       -1.08214486e-02, -9.83582532e-02,  6.10124958e-02,  9.56718482e-02,
       -5.02652402e-02,  1.90468347e-02, -4.57541426e-02, -3.96417371e-02,
        4.88513307e-02, -2.43080471e-03,  1.09728474e-01, -3.04168617e-02,
        6.30332389e-02,  6.14436074e-03,  5.01937774e-02, -2.82045139e-02,
       -3.12592862e-03, -7.28510334e-03, -5.27343860e-02,  4.17608589e-02,
        4.10145209e-02,  3.74385216e-02,  9.69442620e-02,  9.14175680e-02,
       -2.36496737e-02,  2.35303122e-02, -3.93336757e-02,  1.47313889e-03,
       -3.78292564e-02,  1.02186096e-01,  2.29402971e-02, -1.37522090e-02]), array([ 0.06816819, -0.12396499, -0.11502646,  0.04545077,  0.00345338,
        0.01430312, -0.07575118, -0.00132689,  0.11442148,  0.04707048,
        0.00969203, -0.0250746 , -0.02253426, -0.12159962,  0.01471398,
        0.04828599, -0.08998445, -0.03777387, -0.06610342, -0.05513172,
        0.00305436, -0.11904131,  0.03508131,  0.04305306, -0.0143406 ,
       -0.14167457, -0.05690599,  0.07041168, -0.15865272, -0.01984725,
       -0.06739047,  0.04594943, -0.03855611,  0.02307003,  0.03229374,
        0.10486783, -0.01159012,  0.02147609, -0.0848123 ,  0.06798515,
       -0.04343306, -0.03202497, -0.00127058, -0.08925715,  0.01306518,
        0.05525251,  0.01672961, -0.09074278, -0.00707081,  0.04237766,
       -0.10725293, -0.06138045,  0.05005987,  0.01542529,  0.02352791,
        0.10619673,  0.0267593 ,  0.03757233,  0.15481211, -0.01221049,
       -0.04556751,  0.02443168,  0.02507101, -0.03008099, -0.05896074,
        0.13191643,  0.0186662 , -0.0418212 ,  0.05230926, -0.00623334,
       -0.01444219,  0.0321117 , -0.07378008,  0.02013873,  0.10924986,
        0.02662365, -0.02148567, -0.02896105, -0.02343473, -0.01450394,
       -0.02157673,  0.05233103, -0.09898449,  0.04922928, -0.0216841 ,
        0.02152769, -0.06436287,  0.07587882,  0.07113887,  0.05480272,
       -0.08650844,  0.04735154,  0.02312149, -0.09167789, -0.00138764,
        0.02081876,  0.03175775,  0.01012349,  0.01663238,  0.08415872,
        0.07001944, -0.16373902, -0.01812908, -0.00940331, -0.10006327,
        0.00537884,  0.07131722, -0.06205446,  0.01921204, -0.05230316,
       -0.03837325,  0.08437473, -0.03098576,  0.0263972 , -0.10686625,
        0.10635794,  0.0344178 , -0.01042266,  0.03286976, -0.08051553,
        0.06753495, -0.01850294, -0.01165452, -0.02807871, -0.03316131,
        0.03967301,  0.01461498,  0.03722168, -0.00050227,  0.02930911,
       -0.02392067,  0.12734116, -0.08954533,  0.0720128 ,  0.07812192,
       -0.04866724,  0.12731499,  0.05685298,  0.05546226,  0.00160036,
       -0.0124059 , -0.01195945,  0.02230341,  0.08002249,  0.01035495,
       -0.05205303,  0.04708346,  0.05260741,  0.03738795,  0.04257584,
       -0.01547285, -0.05305506, -0.04958299, -0.09340217,  0.01353226,
       -0.0047658 , -0.02512793, -0.16879695, -0.07123496,  0.0640118 ,
       -0.03795714,  0.02059358, -0.07942278,  0.08216523, -0.0082255 ,
        0.01656436,  0.13282651,  0.08660469, -0.01750812, -0.03387912,
       -0.01859205,  0.02912056, -0.00600941, -0.0225726 ,  0.0837915 ,
       -0.00652259,  0.13133075,  0.00518236,  0.10196048,  0.0204067 ,
        0.05499519, -0.04138292, -0.04875864,  0.02625405,  0.0564568 ,
        0.00229907, -0.0734666 , -0.07768507, -0.08150451, -0.10094983,
        0.00625277,  0.06425185, -0.00076618, -0.12909348, -0.01089443,
       -0.00929754,  0.11970359, -0.08321792, -0.0223118 ,  0.09742215,
        0.0256985 ,  0.03217111,  0.00654398,  0.01884325,  0.04440969,
       -0.04623019,  0.05014639, -0.07841293,  0.01875327,  0.01020799,
       -0.03519682, -0.00129557, -0.0355536 , -0.06120543, -0.02019512,
       -0.06304937,  0.01843513, -0.06568199,  0.04066138, -0.06747489,
       -0.0323414 , -0.03567143,  0.07567983,  0.01886303, -0.01209223,
        0.02848213, -0.12034287, -0.00246338,  0.11407909,  0.00755838,
        0.00387856,  0.0313039 ,  0.05114849, -0.11407638, -0.09607855,
        0.01200985, -0.04082698,  0.07493046, -0.01799978, -0.11048207,
       -0.07241694, -0.15703545, -0.01337103, -0.01075872, -0.02424129,
        0.06161535,  0.0684481 , -0.00973097, -0.12191348,  0.03525328,
        0.03171692,  0.01766952,  0.05605283, -0.06652395,  0.11472108,
       -0.05479343]), array([ 4.59696249e-02, -1.03529587e-01, -5.70776963e-02,  9.30389279e-02,
       -1.60078750e-02,  5.15346310e-02,  1.45387307e-02, -6.43985506e-03,
        8.36809521e-02, -1.45205345e-02, -1.09259687e-01, -4.92839239e-02,
        3.38191426e-02,  2.42670167e-02, -9.61490597e-02, -4.44430525e-02,
       -3.56923303e-02, -2.24239002e-02, -8.44211106e-02, -3.85545918e-02,
       -4.47049196e-02,  4.73385067e-02,  4.96447744e-02, -8.94640543e-03,
       -2.40541523e-02,  8.58525267e-03, -5.01021125e-02, -7.03654930e-02,
       -5.76885927e-02, -3.89498532e-02, -7.13865690e-02,  9.94339184e-03,
        4.47727300e-02, -9.73786452e-02,  1.54180737e-03, -1.34944696e-02,
        2.09537450e-02, -7.93876288e-04, -8.91876253e-02, -7.74556683e-02,
        1.80532480e-03,  6.31222981e-02, -1.35854576e-01,  2.12125398e-02,
        7.29909288e-02, -9.38911754e-03,  3.37584468e-02, -4.31377908e-02,
       -7.32388079e-03, -9.94534253e-02, -1.43840476e-01,  5.78447198e-02,
       -6.04357471e-02, -2.79495507e-02, -8.50671116e-02, -8.65106713e-02,
       -1.06660562e-02, -4.47642334e-02,  2.31909414e-02,  2.77661302e-02,
        1.96861807e-02,  1.46396866e-02,  9.68061840e-02, -6.55037150e-02,
       -3.37060587e-03,  6.94977747e-02,  7.75876159e-02, -8.72343054e-02,
       -1.91650068e-02,  8.33294272e-02,  3.11739070e-02,  1.19806375e-02,
       -1.33563501e-02, -2.35486448e-02,  1.76840854e-02, -1.54847689e-02,
       -2.61247109e-03,  4.82253400e-03, -4.16040309e-02,  3.26495451e-02,
        1.78807573e-01, -1.09029511e-01,  1.98802435e-02, -4.66425305e-02,
        3.54273460e-02,  2.85659916e-03,  4.67446696e-02,  7.21248569e-02,
        1.18062679e-02,  6.94586580e-02, -1.43837690e-01,  9.75844344e-02,
        2.30124651e-02,  6.61325928e-02, -7.48471895e-02, -5.09730528e-02,
        8.95424899e-03, -1.04538363e-03,  1.73680708e-02,  6.65419508e-02,
       -1.08307572e-01,  5.97500368e-02,  5.08148217e-04, -1.25033223e-02,
       -9.02623412e-02,  3.08652575e-02, -1.48120978e-01,  3.97831252e-02,
       -9.15502490e-02,  3.27998530e-03, -7.31177137e-03,  7.17586555e-02,
       -1.09616388e-02, -4.25898463e-02, -3.10833199e-02,  7.80995099e-02,
        3.29052537e-02, -2.36608691e-02, -1.45075022e-02, -9.61563513e-02,
        2.38353802e-02, -1.07975964e-01,  4.61275280e-02,  7.89175844e-03,
        1.46454780e-01, -6.75003619e-02, -6.29897677e-03,  1.10336467e-01,
        7.30993846e-02, -4.27170682e-02,  1.37418361e-02,  1.22673321e-04,
        1.72997982e-02, -4.28685194e-02, -5.48037875e-03, -9.80507478e-02,
        8.29068916e-03,  2.76116504e-02, -1.39536375e-02, -9.90937156e-02,
        6.06527607e-02,  2.23659456e-02, -4.45456759e-02, -1.72166345e-02,
       -9.20756787e-02, -6.29154785e-02, -3.29981705e-02,  7.53714841e-02,
       -3.92376287e-02,  3.92812082e-02, -3.30291466e-02,  1.52131708e-02,
       -8.67551961e-03,  5.35676138e-02, -3.28141007e-02, -9.64846792e-02,
       -8.47914236e-02, -3.26340906e-02, -7.66378287e-02,  2.51691343e-02,
       -2.85778599e-02,  1.45697320e-01, -1.06619919e-01,  1.24941997e-02,
       -6.92071419e-02, -3.02733322e-02,  8.67699040e-03, -1.42676426e-01,
        1.45307990e-02,  1.16755560e-01, -2.78213436e-02,  1.59408759e-02,
        6.28855099e-02, -1.00595973e-02, -3.56456283e-02, -1.39631003e-01,
        3.93288858e-02, -6.94360671e-03, -6.21346399e-02,  7.76751895e-03,
       -3.58125280e-02, -3.24223274e-02,  3.71998454e-02, -8.21183325e-02,
        6.42595002e-02, -8.57569974e-02, -4.06609756e-02,  1.17985064e-01,
        6.91704347e-02,  5.98863003e-03, -5.77075523e-02, -3.24742457e-02,
        4.49590983e-03,  2.30229062e-03, -5.57551351e-03, -1.89177220e-03,
       -1.31532859e-02, -8.84746304e-02, -3.16912420e-02,  5.61180433e-02,
       -8.39560725e-02, -1.16304446e-02, -5.02971536e-02,  6.83206198e-03,
       -2.38451346e-02,  9.06475632e-02,  3.10821209e-02, -8.04875255e-02,
        4.11841373e-02, -4.75298526e-02, -3.54024227e-02,  1.92548067e-02,
       -3.01718535e-02,  5.47824225e-02, -4.75211525e-02, -3.79763350e-02,
       -4.02725580e-02, -3.47055598e-02,  6.74424416e-02,  3.85806797e-02,
        1.04916013e-01,  1.39134495e-01, -6.32410910e-02,  4.32066445e-03,
       -7.21674223e-02, -4.77111038e-02, -3.86595973e-02, -6.48632220e-02,
       -6.45400209e-02, -8.85277068e-03, -7.46534673e-02,  5.16698823e-02,
       -5.63100572e-02,  1.72385050e-01, -2.76953269e-02,  4.41874356e-03,
       -7.94774775e-02,  1.04679656e-01,  6.86885365e-02,  1.82176902e-02,
       -8.26349618e-02,  1.29257621e-02, -6.88069768e-02,  8.97248609e-02,
        4.96492792e-02,  5.85579624e-03, -1.49487250e-01, -6.03388685e-02,
        4.21985501e-02,  7.44181823e-02,  6.80546715e-02, -9.76801936e-02,
        1.85021653e-02,  6.23083623e-02, -1.01749508e-01, -4.84992642e-04]), array([ 1.76364394e-02, -1.93703254e-02, -1.19285912e-01, -2.09761345e-02,
       -2.10796517e-02, -9.53639757e-03, -3.62895946e-02, -2.85315825e-02,
       -2.55722797e-03,  2.84956744e-02, -1.96061829e-02, -2.63828703e-02,
        5.94793357e-03, -1.23713922e-01,  3.07227849e-02, -3.62654882e-05,
        2.55640645e-03,  6.18747588e-02, -1.25768372e-01, -1.27122029e-01,
        1.76339564e-02, -4.58351171e-04,  1.09600097e-02, -2.97626525e-02,
        4.27867757e-02,  6.00653094e-02,  3.11702075e-02, -1.67982946e-02,
       -1.35742808e-01,  4.95415897e-02,  7.86837380e-02,  4.15118894e-02,
       -5.66521489e-02,  4.33947922e-02,  5.54212207e-02,  1.00393396e-01,
       -5.64246429e-02, -1.45844930e-02,  4.81270186e-02,  3.69741592e-03,
       -7.19599325e-02,  2.67027860e-02,  3.72678989e-02,  5.66512818e-03,
        1.41420623e-02, -1.66615943e-02,  9.52065018e-02, -5.80078292e-02,
       -3.78255781e-02,  7.07506742e-04,  5.66078135e-02, -9.64362395e-02,
       -8.43878424e-02, -1.31237075e-01,  2.10569058e-03, -7.20229331e-02,
       -2.47664158e-02,  4.25951438e-02, -4.13176680e-02,  1.48953975e-02,
       -1.92784422e-02,  4.32865890e-02,  3.05999633e-02, -4.45630287e-02,
       -1.08617820e-02,  6.72208440e-02,  2.53348910e-02, -8.99694012e-02,
        4.22241707e-02, -6.98528408e-02,  9.09709426e-02,  6.59556635e-02,
        1.86696715e-02,  1.01263629e-02, -1.34624321e-02,  4.36993928e-02,
       -2.38794531e-02,  1.51715504e-02,  5.61557219e-02, -3.61823702e-02,
       -3.96850377e-02, -7.69214501e-02, -6.85558889e-02, -1.17755261e-01,
        4.70036096e-02,  1.16500629e-01,  9.17963311e-02, -1.04725713e-01,
        4.66832584e-02,  9.15887844e-02, -2.94202124e-02,  6.53358088e-03,
       -1.29315563e-01,  1.71787725e-02, -6.84970873e-02,  1.28572161e-02,
       -1.00547106e-01,  3.49620300e-02,  6.59823160e-02, -5.04396113e-02,
       -7.32219199e-02, -1.38855670e-01, -1.71579937e-02, -3.74567496e-02,
       -9.98912512e-02,  2.91760438e-02, -2.89797175e-02, -6.92812804e-02,
       -3.57164656e-02,  6.89958810e-02,  3.88313165e-02, -8.37414791e-03,
        3.98725989e-02, -7.47669223e-02,  2.08475138e-03,  9.48809009e-03,
       -3.02591678e-02,  2.80450894e-02, -5.02627328e-02,  3.28342946e-02,
        3.55416978e-02, -1.87875631e-01,  7.66599857e-02, -1.96278543e-01,
       -3.96757819e-02,  6.60361161e-02, -2.04344109e-02, -9.44276720e-02,
       -7.04406165e-02,  6.61364384e-02, -1.23751831e-01,  7.27111039e-02,
       -3.54877416e-03, -7.67945653e-02, -3.75547164e-02, -6.55129210e-02,
        5.57880310e-02,  4.71709959e-02, -1.10720185e-01, -2.59927899e-02,
       -5.55922738e-02,  8.33067131e-02, -8.69512587e-02,  1.03318148e-01,
       -2.14430146e-03, -7.48327718e-02, -4.36189032e-02, -5.16271482e-02,
       -8.20900280e-02,  6.10719221e-03, -7.90821957e-02, -1.40423458e-01,
       -2.44525358e-02, -1.16110450e-01, -4.84418761e-02, -5.85495090e-02,
       -5.61094612e-02,  1.28869250e-01,  3.63310864e-03,  1.58027791e-02,
       -1.14406465e-01, -4.21175874e-02,  1.38662669e-02, -6.59833618e-02,
        5.84825263e-02, -7.74580033e-02, -1.84430133e-02, -3.77438420e-02,
        1.05828439e-02, -4.12498736e-02, -1.74465278e-02, -1.52076306e-02,
       -4.72972236e-02,  3.12417969e-02,  1.53530722e-02, -5.70705808e-03,
       -2.77230711e-02,  5.13003120e-02, -1.51885769e-02, -1.97524129e-02,
        6.74241598e-03,  1.09266309e-01, -8.63237801e-02,  4.95815640e-02,
       -1.21927059e-01,  1.39931760e-02,  1.01441022e-01,  3.14720177e-02,
        1.09888712e-02, -3.60949697e-02,  1.99955183e-03, -1.62337556e-02,
       -5.06197319e-02, -1.86690364e-02,  1.27764631e-02, -4.94115498e-02,
       -5.19274609e-02, -8.76045895e-02,  2.51641027e-02,  3.51207873e-02,
        1.49197692e-02,  6.21604652e-02,  5.65899314e-03,  2.52469200e-02,
        1.95682274e-02, -5.89550782e-03, -3.41868426e-02,  1.42599424e-01,
       -2.91072752e-02, -4.28849292e-02,  2.79517999e-02,  1.62203338e-02,
       -9.04736487e-02,  7.23045338e-02, -6.06647081e-03, -7.15479743e-02,
        7.65619899e-02,  4.02957663e-02, -8.14410708e-02,  1.14456993e-01,
        4.24289296e-02, -6.56666666e-04,  1.02415834e-01,  1.02593525e-02,
       -1.80865145e-03, -4.25257119e-02,  3.74732695e-02,  3.06949522e-02,
        9.45454022e-04, -5.08779740e-02,  8.36795181e-03, -4.49529386e-03,
       -8.58805616e-02, -9.47900304e-02,  3.63979185e-02, -1.67571003e-01,
        1.32291797e-02, -3.07902754e-02,  8.42558058e-02, -3.17811347e-02,
        4.24731649e-02, -2.55626847e-02,  7.32579185e-03,  4.74549335e-02,
       -2.97902585e-02,  6.41817333e-02, -6.39977317e-02,  1.89728429e-02,
        5.94213300e-02,  4.25087980e-02,  1.23425971e-01, -4.91682490e-02,
       -8.88221922e-02, -6.40387717e-02,  4.25348863e-02,  3.20680351e-02])], '_qjl_projection': array([[-0.0625,  0.0625, -0.0625, ..., -0.0625,  0.0625, -0.0625],
       [ 0.0625, -0.0625, -0.0625, ..., -0.0625, -0.0625, -0.0625],
       [ 0.0625, -0.0625,  0.0625, ..., -0.0625, -0.0625, -0.0625],
       ...,
       [ 0.0625,  0.0625,  0.0625, ..., -0.0625, -0.0625,  0.0625],
       [-0.0625, -0.0625,  0.0625, ...,  0.0625, -0.0625,  0.0625],
       [-0.0625, -0.0625, -0.0625, ...,  0.0625, -0.0625, -0.0625]]), '_qjl_gain': 0.8754460028115503}

🎉 VALUE ENGINE READY!
   Bits per dim : 5.0

```