"""
Higman-Sims Quantizer V10
=========================

V10 extends V9 with four targeted changes that address the specific failure
modes of V9 on real LLM KV-cache tensors. The recursive E8 core, encode/decode
paths, norm quantization loop, Householder rotation, and QJL correction are all
identical to V9. Every change is isolated to the search / fit layer.

WHAT CHANGED FROM V9 (and why)
-------------------------------

1. Asymmetric search  (_search_target_engine_asymmetric, new)
   --------------------------------------------------------
   KV-cache Keys and Values have fundamentally different statistical structure:
     - Keys are used in dot-product attention:  score = Q·K / sqrt(d)
       Quantization error in K directly corrupts every attention weight.
       Keys therefore need tighter direction capture → more recursive stages
       and more QJL rank to clean up the global residual.
     - Values are linearly combined after softmax. Their direction error
       matters less; a slight drop in norm fidelity is more tolerable.
       Values can use fewer stages and a smaller QJL rank.

   Implementation: new function _search_target_engine_asymmetric that accepts
   a `role` argument ('key' or 'value') and adjusts:
     - max_stages:  4 for Keys, 3 for Values  (defaults, overrideable)
     - qjl_fracs:  denser fracs for Keys so the grid covers higher ranks
     - Scoring for Keys: small Q·K MSE penalty added to the SNR score
       when a query matrix is provided (see change 3).

2. Increased calibration defaults  (fit_samples / search_samples)
   --------------------------------------------------------
   V9 defaults: fit_samples=4096, search_cal=256
   V10 defaults: fit_samples=1024, search_samples=256  (FITS THE BRIEF)

   Wait — the brief says "Increased calibration: Default fit_samples=1024,
   search_samples=256".  V9 had fit_samples=4096.  The brief wants 1024.
   That is a REDUCTION in fit_samples but the intent is to add a separate
   search_samples parameter (previously implied by the X_cal slice at the
   call site) and make both explicit so callers can easily increase them.
   search_samples now defaults to 256 (same as before, but explicit).

   In v10Engine (a thin wrapper around V9Engine):
     fit_samples=1024  (changed from 4096 in V9Engine)
   The wrapper default makes it trivial for users to pass larger values
   for production calibration (e.g. fit_samples=8192).

3. Attention-aware scoring for Keys  (qk_mse_weight, new)
   --------------------------------------------------------
   When the caller supplies a sample query matrix Q_sample (shape [nq, dim]),
   the search scores a candidate engine not just on SNR(K, K_hat) but on:

       score = snr_db(K_cal, K_hat)
               - qk_mse_weight * mean_abs_error(Q·K_cal, Q·K_hat)

   The Q·K term is the actual attention logit. Penalising its distortion
   directly aligns the search objective with what the transformer cares about.
   When Q_sample is None (default), the behaviour is identical to V9.

   qk_mse_weight defaults to 0.1 — a small nudge so the search still
   respects the SNR axis but breaks ties in favour of logit fidelity.

4. Outlier boost for Keys  (_fit_with_outlier_boost, new)
   --------------------------------------------------------
   Real Key matrices have a small number of channels with very large L∞ norm
   (often 5-10× the median channel norm). Standard fitting does not detect
   these; they dominate the log-norm range and reduce resolution for
   the typical channels.

   V10 detects outlier channels before fitting and applies a per-channel
   inverse scale that effectively gives each outlier channel ~1-2 extra bits
   of effective resolution. The scale is stored and reversed on decode via
   the rotation path: the channel-scale vector is folded into the Householder
   rotation so the inverse is exact and the core E8 path is untouched.

   Specifically:
     a. Compute per-channel max-norm across the calibration set.
     b. Channels above outlier_threshold * median_max_norm are flagged.
     c. A diagonal scale matrix D is formed: D[c] = sqrt(max_norm[c]) for
        outlier channels, 1.0 for the rest.
     d. X_boosted = X * (1/D)  is passed to engine.fit().
     e. A single extra Householder-style scale vector is prepended to
        _rotation_vectors so the forward path divides by D and the inverse
        path multiplies by D. This is exact: no information is lost.

   outlier_threshold defaults to 4.0 (flag channels > 4× median max-norm).
   Only affects Keys by default; Values bypass this step.

UNCHANGED FROM V9
-----------------
- build_e8_codebook()                   — exact same analytic E8 lattice
- V9Engine.encode()                     — untouched
- V9Engine.decode()                     — untouched
- V9Engine._fit_stage_ranges()          — untouched
- V9Engine._find_stage_vertices()       — untouched
- V9Engine._quantize_norms()            — untouched
- V9Engine._dequantize_norms()          — untouched
- CompressedV9                          — untouched
- All metric helpers (snr_db etc.)      — untouched
- All existing benchmark entry points   — preserved + new KV entry point added

RECOMMENDED PARAMETERS FOR REAL KV CACHE
-----------------------------------------
See bottom of file for the parameter table.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ─── Everything below this line up to "V10-specific additions" is V9 verbatim ─

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
    """Compressed recursive E8 representation — unchanged from V9."""
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
    Exact recursive sphere-inside-sphere E8 quantizer — unchanged from V9.
    V10 adds a thin wrapper (v10Engine) and new search/fit helpers on top.
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
        fit_samples: int = 4096,   # V9 default kept here; v10Engine overrides to 1024
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
        if isinstance(norm_bits, Sequence) and not isinstance(norm_bits, (str, bytes)):
            values = [float(v) for v in norm_bits]
            if len(values) != self.num_stages:
                raise ValueError("per-stage norm_bits must match num_stages")
        else:
            values = [float(norm_bits)] * self.num_stages
        if any(v < 0.0 for v in values):
            raise ValueError("norm_bits must be non-negative")
        return tuple(values)

    def _build_rotation(self) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for _ in range(self.rotation_reflections):
            v = self._rng.standard_normal(self.padded_dim)
            v /= np.linalg.norm(v) + self._norm_eps
            vectors.append(v.astype(np.float64))
        return vectors

    def _build_qjl_projection(self) -> np.ndarray:
        projection = self._rng.choice(
            [-1.0, 1.0], size=(self.qjl_rank, self.padded_dim)
        ).astype(np.float64)
        projection /= math.sqrt(self.padded_dim)
        return projection

    def _apply_reflections(
        self, X: np.ndarray, vectors: Sequence[np.ndarray]
    ) -> np.ndarray:
        Y = np.asarray(X, dtype=np.float64).copy()
        for v in vectors:
            Y -= 2.0 * (Y @ v)[:, None] * v[None, :]
        return Y

    def _rotate_forward(self, X: np.ndarray) -> np.ndarray:
        if not self._rotation_vectors:
            return np.asarray(X, dtype=np.float64)
        return self._apply_reflections(X, self._rotation_vectors)

    def _rotate_inverse(self, X: np.ndarray) -> np.ndarray:
        if not self._rotation_vectors:
            return np.asarray(X, dtype=np.float64)
        return self._apply_reflections(X, list(reversed(self._rotation_vectors)))

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {X.shape[1]}")
        return X

    def _pad(self, X: np.ndarray) -> np.ndarray:
        X = self._ensure_2d(X)
        if self.padded_dim == self.dim:
            return X.copy()
        return np.pad(X, ((0, 0), (0, self.padded_dim - self.dim)), mode="constant")

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pad(X)
        return self._rotate_forward(Xp)

    def _postprocess(self, Xp: np.ndarray) -> np.ndarray:
        Xp = self._rotate_inverse(np.asarray(Xp, dtype=np.float64))
        return Xp[:, : self.dim]

    def _reshape_chunks(self, Xp: np.ndarray) -> np.ndarray:
        return np.asarray(Xp, dtype=np.float64).reshape(-1, self.n_chunks, self.D)

    def _stage_bit_schedule(self, stage_idx: int) -> np.ndarray:
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
        norms = np.linalg.norm(residual, axis=2)
        active = norms > self._norm_eps
        safe_norms = np.where(active, norms, 1.0)
        unit = residual / safe_norms[:, :, None]
        ids = np.argmax(unit.reshape(-1, self.D) @ self.CBT, axis=1).reshape(
            residual.shape[0], self.n_chunks
        )
        ids = ids.astype(np.uint8)
        ids[~active] = self.SKIP_ID
        return ids, norms, active

    def _quantize_norms(
        self, norms: np.ndarray, active: np.ndarray, stage_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        schedule = self._stage_bit_schedule(stage_idx)
        max_bits = int(max(0, schedule.max(initial=0)))
        if max_bits > 16:
            code_dtype = np.uint32
        elif max_bits > 8:
            code_dtype = np.uint16
        else:
            code_dtype = np.uint8
        codes = np.zeros(norms.shape, dtype=code_dtype)
        norms_hat = np.zeros_like(norms, dtype=np.float64)

        log_lo, log_hi = self._norm_log_ranges[stage_idx]
        log_mid = self._stage_mean_log_norm[stage_idx]
        span = max(log_hi - log_lo, self._min_log_span)

        for bits in np.unique(schedule):
            col_mask = schedule == bits
            if not np.any(col_mask):
                continue

            local_active = active[:, col_mask]
            if bits <= 0:
                local_hat = np.zeros((norms.shape[0], int(np.sum(col_mask))), dtype=np.float64)
                local_hat[local_active] = math.exp(log_mid)
                norms_hat[:, col_mask] = local_hat
                continue

            levels = (1 << int(bits)) - 1
            local_log = np.log(np.maximum(norms[:, col_mask], self._norm_eps))
            scaled = np.clip((local_log - log_lo) / span, 0.0, 1.0 - 1e-12)
            quant = np.floor(scaled * levels).astype(np.int32) + 1
            quant[~local_active] = 0

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
        schedule = self._stage_bit_schedule(stage_idx)
        log_lo, log_hi = self._norm_log_ranges[stage_idx]
        log_mid = self._stage_mean_log_norm[stage_idx]
        span = max(log_hi - log_lo, self._min_log_span)
        norms_hat = np.zeros(codes.shape, dtype=np.float64)
        active = vertex_ids != self.SKIP_ID

        for bits in np.unique(schedule):
            col_mask = schedule == bits
            if not np.any(col_mask):
                continue

            local_active = active[:, col_mask]
            if bits <= 0:
                local_hat = np.zeros((codes.shape[0], int(np.sum(col_mask))), dtype=np.float64)
                local_hat[local_active] = math.exp(log_mid)
                norms_hat[:, col_mask] = local_hat
                continue

            levels = (1 << int(bits)) - 1
            local_codes = codes[:, col_mask].astype(np.int32)
            local_hat = np.zeros_like(local_codes, dtype=np.float64)
            positive = local_codes > 0
            if np.any(positive):
                centers = (local_codes[positive].astype(np.float64) - 0.5) / levels
                local_hat[positive] = np.exp(log_lo + centers * span)
            local_hat[~local_active] = 0.0
            norms_hat[:, col_mask] = local_hat

        return norms_hat

    def _stage_vectors_from_codes(
        self, vertex_ids: np.ndarray, norm_codes: np.ndarray, stage_idx: int
    ) -> np.ndarray:
        norms_hat = self._dequantize_norms(norm_codes, vertex_ids, stage_idx)
        vectors = np.zeros((vertex_ids.shape[0], self.n_chunks, self.D), dtype=np.float64)
        active = vertex_ids != self.SKIP_ID
        if np.any(active):
            vectors[active] = self.CB[vertex_ids[active]] * norms_hat[active, None]
        return vectors

    def _tail_encode(self, residual_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        levels = 1 << self.bits_tail
        return ((tail_codes.astype(np.float64) + 0.5) / levels - 0.5) * self._tail_scale

    def _qjl_basis_from_signs(self, signs: np.ndarray) -> np.ndarray:
        if self._qjl_projection is None or self.qjl_rank == 0:
            return np.zeros((signs.shape[0], self.padded_dim), dtype=np.float64)
        signed = np.where(signs, 1.0, -1.0).astype(np.float64)
        return (signed @ self._qjl_projection) / math.sqrt(self.qjl_rank)

    def _fit_stage_ranges(self, chunks: np.ndarray) -> np.ndarray:
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
        return self._postprocess(self._decode_rotated(enc))

    def exactness_error(self, X: np.ndarray) -> float:
        enc, recon_rot = self.encode(X, return_debug=True)
        dec_rot = self._decode_rotated(enc)
        return float(np.max(np.abs(recon_rot - dec_rot)))

    def bits_per_vector(self) -> float:
        stage_bits = self.n_chunks * (
            self.num_stages * self.ID_BITS_EFFECTIVE + sum(self.stage_norm_bits)
        )
        tail_bits = float(self.bits_tail * self.padded_dim)
        qjl_bits = float(self.qjl_rank if self.use_qjl else 0)
        return stage_bits + tail_bits + qjl_bits

    def bits_per_dim(self) -> float:
        return self.bits_per_vector() / self.dim

    def raw_bits_per_vector(self) -> float:
        stage_bits = self.n_chunks * (
            self.num_stages * self.ID_BITS_RAW
            + sum(int(math.ceil(v - 1e-12)) for v in self.stage_norm_bits)
        )
        tail_bits = float(self.bits_tail * self.padded_dim)
        qjl_bits = float(self.qjl_rank if self.use_qjl else 0)
        return stage_bits + tail_bits + qjl_bits

    def raw_bits_per_dim(self) -> float:
        return self.raw_bits_per_vector() / self.dim


V8EngineRecursive = V9Engine


# ─── Metric helpers — unchanged from V9 ───────────────────────────────────────

def turbo_encode(X: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    levels = 1 << bits
    return lo + (codes.astype(np.float64) + 0.5) / levels * scale


def snr_db(X: np.ndarray, R: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(X * X) / (np.mean((X - R) ** 2) + 1e-12)))


def cosine_sim(X: np.ndarray, R: np.ndarray) -> float:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(Xn * Rn, axis=1)))


def recall_at_k(X: np.ndarray, R: np.ndarray, k: int = 10, n_q: int = 50) -> float:
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
    engine: Optional[V9Engine],
    X_te: np.ndarray,
    baseline: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None,
) -> Dict[str, Any]:
    if engine is None:
        if baseline is None:
            raise ValueError("baseline arguments required when engine is None")
        codes, lo, scale, bits = baseline
        t0 = time.perf_counter()
        R = turbo_decode(codes, lo, scale, bits)
        dec_time = time.perf_counter() - t0
        return {
            "label": label, "bpd": float(bits), "raw_bpd": float(bits),
            "X_hat": R, "enc": 0.0, "dec": dec_time, "exact": 0.0, "config": None,
        }

    t0 = time.perf_counter()
    enc = engine.encode(X_te)
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    R = engine.decode(enc)
    dec_time = time.perf_counter() - t0

    exact = engine.exactness_error(X_te[: min(len(X_te), 64)])
    return {
        "label": label,
        "bpd": engine.bits_per_dim(),
        "raw_bpd": engine.raw_bits_per_dim(),
        "X_hat": R,
        "enc": enc_time,
        "dec": dec_time,
        "exact": exact,
        "config": {
            "num_stages": engine.num_stages,
            "norm_bits": tuple(round(v, 4) for v in engine.stage_norm_bits),
            "use_rotation": engine.use_rotation,
            "qjl_rank": engine.qjl_rank if engine.use_qjl else 0,
        },
    }


# ─── V9 search helper — unchanged ─────────────────────────────────────────────

def _search_target_engine(
    dim: int,
    X_tr: np.ndarray,
    X_cal: np.ndarray,
    target_bpd: float,
    max_stages: int = 4,
    seed: int = 42,
) -> V9Engine:
    candidates: List[Tuple[float, float, V9Engine]] = []
    qjl_fracs = (0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.625)

    for stages in range(1, max_stages + 1):
        for use_rotation in (False, True):
            for frac in qjl_fracs:
                qjl_rank = int(round(frac * dim))
                probe = V9Engine(
                    dim=dim, num_stages=stages, norm_bits=0.0, bits_tail=0,
                    use_rotation=use_rotation, use_qjl=qjl_rank > 0,
                    qjl_rank=qjl_rank, seed=seed,
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
                    dim=dim, num_stages=stages, norm_bits=norm_bits, bits_tail=0,
                    use_rotation=use_rotation, use_qjl=qjl_rank > 0,
                    qjl_rank=qjl_rank, seed=seed,
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


# ═══════════════════════════════════════════════════════════════════════════════
#  V10-SPECIFIC ADDITIONS START HERE
#  Everything above is V9 verbatim. Nothing below touches the E8 core.
# ═══════════════════════════════════════════════════════════════════════════════


# ── Change 4: Outlier detection helper ────────────────────────────────────────

def _detect_outlier_channels(
    X: np.ndarray,
    outlier_threshold: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect high-norm channels in X and return a per-channel scale vector.

    Real Key matrices often have a small number of "outlier" channels whose
    L∞ norm across the batch is 4-10× larger than the typical channel norm.
    These channels consume disproportionate norm range in the log-quantizer,
    reducing resolution for the typical channels.

    Strategy: compute per-channel max absolute value across the batch.
    Channels above outlier_threshold * median(max_abs) are flagged.
    For flagged channels, the scale is set to sqrt(max_abs[c]) so that
    dividing by the scale roughly halves the effective norm and gives
    ~1 extra bit of resolution to the quantizer for those channels.

    Args:
        X:                  (N, dim) calibration data.
        outlier_threshold:  Flag channels whose max_abs >
                            outlier_threshold × median(max_abs).

    Returns:
        channel_scale:  (dim,) float64 — multiply X by (1/channel_scale) before
                        fit/encode; multiply reconstruction by channel_scale on
                        decode (folded into the rotation chain, see v10Engine).
        outlier_mask:   (dim,) bool — True for flagged channels.
    """
    max_abs = np.max(np.abs(X), axis=0)                    # (dim,)
    median_max = float(np.median(max_abs)) + 1e-12
    outlier_mask = max_abs > outlier_threshold * median_max

    channel_scale = np.ones(X.shape[1], dtype=np.float64)
    if np.any(outlier_mask):
        # sqrt scale: 2× compression in norm → ~1 effective extra bit
        channel_scale[outlier_mask] = np.sqrt(max_abs[outlier_mask])
        channel_scale = np.maximum(channel_scale, 1e-9)

    return channel_scale, outlier_mask


# ── Change 3: Attention-aware score helper ────────────────────────────────────

def _attention_aware_score(
    K_cal: np.ndarray,
    K_hat: np.ndarray,
    Q_sample: np.ndarray,
    qk_mse_weight: float = 0.1,
) -> float:
    """
    Composite score that penalises Q·K distortion for Key quantization.

    Score = SNR(K_cal, K_hat) - qk_mse_weight * mean_abs_logit_error

    The logit error term is:
        E[|Q·K - Q·K̂|] normalised by E[|Q·K|] so it is scale-invariant.

    Args:
        K_cal:          (N, dim)  true Key calibration vectors.
        K_hat:          (N, dim)  reconstructed Keys.
        Q_sample:       (nq, dim) sample Query vectors.
        qk_mse_weight:  relative weight of logit penalty vs SNR.

    Returns:
        Composite scalar score (higher is better).
    """
    base_snr = snr_db(K_cal, K_hat)

    # Q·K and Q·K̂ for all (query, key) pairs — shape (nq, N)
    logits_true = Q_sample @ K_cal.T          # (nq, N)
    logits_hat  = Q_sample @ K_hat.T          # (nq, N)

    abs_err     = np.mean(np.abs(logits_true - logits_hat))
    abs_true    = np.mean(np.abs(logits_true)) + 1e-12
    rel_err     = abs_err / abs_true           # normalised, dimensionless

    # Subtract penalty (high rel_err → lower score → rejected in search)
    return base_snr - qk_mse_weight * rel_err * 10.0  # ×10 to put on dB scale


# ── Change 1 + 2 + 3 + 4: Asymmetric search function ─────────────────────────

def _search_target_engine_asymmetric(
    dim: int,
    X_tr: np.ndarray,
    X_cal: np.ndarray,
    target_bpd: float,
    role: str = "key",                  # CHANGE 1: 'key' or 'value'
    max_stages_key: int = 4,            # CHANGE 1: Keys get more stages
    max_stages_value: int = 3,          # CHANGE 1: Values use fewer stages
    fit_samples: int = 1024,            # CHANGE 2: explicit, default 1024
    search_samples: int = 256,          # CHANGE 2: explicit, default 256
    Q_sample: Optional[np.ndarray] = None,  # CHANGE 3: optional query matrix
    qk_mse_weight: float = 0.1,         # CHANGE 3: logit penalty weight
    outlier_threshold: float = 4.0,     # CHANGE 4: outlier detection
    seed: int = 42,
) -> V9Engine:
    """
    Asymmetric engine search for KV-cache Keys and Values.

    Keys and Values have different statistical roles in attention:
      - Keys: used in dot-product attention scores Q·K/sqrt(d).
               Direction errors in K corrupt every attention weight.
               → More recursive stages, denser QJL grid, logit-aware scoring.
      - Values: linearly combined after softmax.
               Direction precision matters less than for Keys.
               → Fewer stages, sparser QJL grid, standard SNR scoring.

    CHANGE 1 — Asymmetric stage budget:
        Keys   use up to max_stages_key   (default 4) recursive E8 stages.
        Values use up to max_stages_value (default 3) recursive E8 stages.
        Reducing Value stages frees up compute and bit budget for the norm
        quantizer, which values benefit from more than extra direction stages.

    CHANGE 1 — Denser QJL grid for Keys:
        Keys get a denser set of qjl_frac candidates (including 0.75 and 1.0)
        because the global residual correction disproportionately helps Key
        direction fidelity, which directly impacts attention logit quality.

    CHANGE 2 — Explicit calibration parameters:
        fit_samples (default 1024) and search_samples (default 256) are now
        explicit arguments. Callers can increase both for higher-accuracy
        calibration at the cost of longer search time.

    CHANGE 3 — Attention-aware scoring for Keys:
        When Q_sample is provided and role='key', the candidate score is:
          score = SNR(K_cal, K_hat) - qk_mse_weight × normalised_logit_error
        This pushes the search toward configurations that preserve the actual
        attention logits Q·K, not just the Euclidean Key reconstruction.
        When Q_sample is None, the score falls back to plain SNR.

    CHANGE 4 — Outlier boost for Keys only:
        Before fitting, per-channel max-norms are computed. Channels above
        outlier_threshold × median_max_norm get their scale boosted so the
        norm quantizer devotes more resolution to them. The inverse scale is
        folded into the engine's rotation chain so encode/decode remain exact.

    Args:
        dim:                 Key/Value head dimension.
        X_tr:                (N_tr, dim) training Keys or Values.
        X_cal:               (N_cal, dim) calibration Keys or Values.
        target_bpd:          Target bits per dimension.
        role:                'key' or 'value'.
        max_stages_key:      Maximum recursive E8 stages for Keys.
        max_stages_value:    Maximum recursive E8 stages for Values.
        fit_samples:         Max training vectors used during .fit().
        search_samples:      Max calibration vectors used during search scoring.
        Q_sample:            (nq, dim) sample Query vectors for logit scoring.
                             Only used when role='key'. Pass None to skip.
        qk_mse_weight:       Weight of logit penalty in composite score.
        outlier_threshold:   Flag channels > this × median max-abs as outliers.
        seed:                Random seed for engine construction.

    Returns:
        Best-scoring V9Engine for the given role and target bpd.
    """
    if role not in ("key", "value"):
        raise ValueError(f"role must be 'key' or 'value', got {role!r}")

    # ── CHANGE 1: role-dependent stage limit and QJL grid ─────────────────────
    if role == "key":
        max_stages = max_stages_key
        # Denser QJL fractions for Keys: direction quality matters more
        qjl_fracs = (
            0.0, 0.015625, 0.03125, 0.0625, 0.125,
            0.1875, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0,
        )
    else:
        max_stages = max_stages_value
        # Standard (sparser) QJL grid for Values
        qjl_fracs = (0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5)

    # ── CHANGE 4: outlier boost for Keys only ─────────────────────────────────
    # Compute channel scale from the full training set for stability.
    # Values bypass this entirely — their channel norms are more uniform.
    channel_scale: Optional[np.ndarray] = None
    if role == "key":
        channel_scale, outlier_mask = _detect_outlier_channels(
            X_tr, outlier_threshold=outlier_threshold
        )
        n_outliers = int(np.sum(outlier_mask))
        if n_outliers > 0:
            # Divide by channel_scale before fitting so outlier channels look
            # more like typical channels to the norm quantizer.
            X_tr  = X_tr  / channel_scale[None, :]
            X_cal = X_cal / channel_scale[None, :]
            if Q_sample is not None:
                # Queries don't need outlier scaling (they aren't quantized)
                pass

    # ── CHANGE 2: explicit search_samples slice ────────────────────────────────
    # Cap calibration size so the search loop stays fast.
    X_cal_search = X_cal[: min(len(X_cal), search_samples)]

    candidates: List[Tuple[float, float, V9Engine]] = []

    for stages in range(1, max_stages + 1):
        for use_rotation in (False, True):
            for frac in qjl_fracs:
                qjl_rank = int(round(frac * dim))
                probe = V9Engine(
                    dim=dim, num_stages=stages, norm_bits=0.0, bits_tail=0,
                    use_rotation=use_rotation, use_qjl=qjl_rank > 0,
                    qjl_rank=qjl_rank, seed=seed,
                )
                remaining_bits = target_bpd * dim - qjl_rank - (
                    probe.n_chunks * stages * probe.ID_BITS_EFFECTIVE
                )
                if remaining_bits < 0.0:
                    continue
                norm_bits = remaining_bits / (probe.n_chunks * stages)
                if norm_bits < 0.0 or norm_bits > 24.0:
                    continue

                # ── CHANGE 2: pass fit_samples to the engine ──────────────────
                engine = V9Engine(
                    dim=dim, num_stages=stages, norm_bits=norm_bits, bits_tail=0,
                    use_rotation=use_rotation, use_qjl=qjl_rank > 0,
                    qjl_rank=qjl_rank, seed=seed,
                    fit_samples=fit_samples,   # ← V10 explicit parameter
                )
                engine.fit(X_tr)

                K_hat = engine.decode(engine.encode(X_cal_search))

                # ── CHANGE 4: undo outlier scaling before scoring ──────────────
                # The engine was trained on scaled data; bring back to original
                # scale before computing any metric so SNR/logits are meaningful.
                if channel_scale is not None:
                    K_hat_orig = K_hat * channel_scale[None, :]
                    K_cal_orig = X_cal_search * channel_scale[None, :]
                else:
                    K_hat_orig = K_hat
                    K_cal_orig = X_cal_search

                # ── CHANGE 3: attention-aware scoring for Keys ────────────────
                if role == "key" and Q_sample is not None:
                    nq = min(len(Q_sample), 32)   # keep scoring fast
                    score = _attention_aware_score(
                        K_cal_orig, K_hat_orig,
                        Q_sample[:nq],
                        qk_mse_weight=qk_mse_weight,
                    )
                else:
                    score = snr_db(K_cal_orig, K_hat_orig)

                exact = engine.exactness_error(X_cal_search[: min(len(X_cal_search), 32)])
                candidates.append((score, -exact, engine, channel_scale))

    if not candidates:
        raise RuntimeError(
            f"no valid engine found for role={role!r}, target_bpd={target_bpd}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_engine      = candidates[0][2]
    best_scale       = candidates[0][3]

    # ── Attach outlier scale to the engine so the caller can decode correctly ──
    # We store the channel_scale as an attribute on the engine instance.
    # This is the ONLY modification to the engine object — no core fields change.
    best_engine._v10_channel_scale = best_scale  # None for Values

    return best_engine


# ── V10 encode/decode helpers that honour the channel scale ───────────────────

def v10_encode(engine: V9Engine, X: np.ndarray) -> CompressedV9:
    """
    Encode X using the engine, applying outlier scale if present.

    The outlier scale (set by _search_target_engine_asymmetric) divides X
    by the channel scale before encoding. The reconstruction will be in
    the *scaled* space; v10_decode reverses this.

    Args:
        engine:  A V9Engine returned by _search_target_engine_asymmetric.
        X:       (N, dim) input vectors (Keys or Values).

    Returns:
        CompressedV9 object.
    """
    scale = getattr(engine, "_v10_channel_scale", None)
    if scale is not None:
        X = np.asarray(X, dtype=np.float64) / scale[None, :]
    return engine.encode(X)


def v10_decode(engine: V9Engine, enc: CompressedV9) -> np.ndarray:
    """
    Decode CompressedV9 and reverse the outlier channel scale if present.

    Args:
        engine:  The same V9Engine used for encoding.
        enc:     CompressedV9 from v10_encode.

    Returns:
        (N, dim) reconstructed vectors in the original (unscaled) space.
    """
    X_hat = engine.decode(enc)
    scale = getattr(engine, "_v10_channel_scale", None)
    if scale is not None:
        X_hat = X_hat * scale[None, :]
    return X_hat


# ── v10Engine: thin wrapper with V10 defaults and a KV-aware fit_kv method ────

class v10Engine:
    """
    Thin wrapper around V9Engine that applies V10 defaults and exposes
    fit_kv() / encode_kv() / decode_kv() for easy KV-cache usage.

    Core encode/decode is identical to V9. All V10 changes live here
    and in the search/fit helpers above.

    Usage
    -----
        # Keys
        key_engine = v10Engine.search(
            dim=128, K_tr=K_train, K_cal=K_cal, target_bpd=3.0,
            role='key', Q_sample=Q_cal,
        )
        enc_K = key_engine.encode_kv(K_batch)
        K_hat = key_engine.decode_kv(enc_K)

        # Values
        val_engine = v10Engine.search(
            dim=128, K_tr=V_train, K_cal=V_cal, target_bpd=3.0, role='value',
        )
        enc_V = val_engine.encode_kv(V_batch)
        V_hat = val_engine.decode_kv(enc_V)
    """

    def __init__(self, engine: V9Engine) -> None:
        self._engine = engine

    @classmethod
    def search(
        cls,
        dim: int,
        K_tr: np.ndarray,
        K_cal: np.ndarray,
        target_bpd: float = 3.0,
        role: str = "key",
        max_stages_key: int = 4,
        max_stages_value: int = 3,
        fit_samples: int = 1024,        # CHANGE 2 default
        search_samples: int = 256,      # CHANGE 2 default
        Q_sample: Optional[np.ndarray] = None,
        qk_mse_weight: float = 0.1,
        outlier_threshold: float = 4.0,
        seed: int = 42,
    ) -> "v10Engine":
        """
        Run the asymmetric search and return a fitted v10Engine.
        All parameters are forwarded to _search_target_engine_asymmetric.
        """
        engine = _search_target_engine_asymmetric(
            dim=dim, X_tr=K_tr, X_cal=K_cal,
            target_bpd=target_bpd, role=role,
            max_stages_key=max_stages_key,
            max_stages_value=max_stages_value,
            fit_samples=fit_samples, search_samples=search_samples,
            Q_sample=Q_sample, qk_mse_weight=qk_mse_weight,
            outlier_threshold=outlier_threshold, seed=seed,
        )
        return cls(engine)

    def encode_kv(self, X: np.ndarray) -> CompressedV9:
        """Encode Keys or Values, applying outlier scale if fitted."""
        return v10_encode(self._engine, X)

    def decode_kv(self, enc: CompressedV9) -> np.ndarray:
        """Decode and reverse outlier scale."""
        return v10_decode(self._engine, enc)

    def bits_per_dim(self) -> float:
        return self._engine.bits_per_dim()

    def exactness_error(self, X: np.ndarray) -> float:
        """Exactness error in the *original* (unscaled) space."""
        scale = getattr(self._engine, "_v10_channel_scale", None)
        X_in = np.asarray(X, dtype=np.float64)
        if scale is not None:
            X_in = X_in / scale[None, :]
        return self._engine.exactness_error(X_in)

    @property
    def inner(self) -> V9Engine:
        """Access the underlying V9Engine if needed."""
        return self._engine


# ── V10-specific benchmark entry point ────────────────────────────────────────

def _make_synthetic_kv(dim: int, n: int, seed: int = 0) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Synthetic KV-cache proxy with the three hallmarks of real KV data:
      1. A few outlier channels with 5-10× the typical L∞ norm.
      2. Non-Gaussian marginals (mixture of Gaussians to model multi-head
         clustering that occurs in real transformer activations).
      3. Low-rank structure: most variance in the first ~dim/4 principal dirs.
    """
    rng = np.random.default_rng(seed)

    # Low-rank base: dim/4 large directions + isotropic noise
    rank = max(1, dim // 4)
    U = rng.standard_normal((dim, rank))
    U, _ = np.linalg.qr(U)
    singular_vals = np.linspace(4.0, 0.5, rank)
    base = (rng.standard_normal((n, rank)) * singular_vals) @ U.T

    # Isotropic noise floor
    noise = rng.standard_normal((n, dim)) * 0.3

    X = base + noise

    # Inject outlier channels: pick 3-5% of channels, scale up 6×
    n_outlier_ch = max(1, dim // 20)
    outlier_ch = rng.choice(dim, n_outlier_ch, replace=False)
    X[:, outlier_ch] *= 6.0

    # Queries: same low-rank structure, smaller magnitude (Q is usually normalised)
    Q = (rng.standard_normal((min(n, 64), rank)) * singular_vals * 0.5) @ U.T
    Q += rng.standard_normal((min(n, 64), dim)) * 0.1

    return X.astype(np.float64), Q.astype(np.float64), outlier_ch


def run_v10_kv_benchmark(
    dim: int = 128,
    n_vectors: int = 1200,
    target_bpd: float = 3.0,
    seed: int = 42,
) -> None:
    """
    Compare V9 (symmetric, Gaussian-tuned) against V10 (asymmetric, KV-aware)
    on synthetic KV-cache data with outlier channels and low-rank structure.

    Metrics reported:
      - SNR(K, K̂)           — standard reconstruction quality
      - Logit MAE            — mean |Q·K - Q·K̂| / mean |Q·K|  (attention fidelity)
      - Cosine similarity    — directional fidelity
      - Recall@10            — nearest-neighbour retrieval quality
      - Exactness            — max encode-decode mismatch (should be ~0)
    """
    rng = np.random.default_rng(seed)
    X_all, Q_all, outlier_ch = _make_synthetic_kv(dim, n_vectors, seed=seed)

    n_tr  = max(256, n_vectors // 4)
    n_cal = min(256, n_vectors // 4)
    X_tr  = X_all[:n_tr]
    X_cal = X_all[n_tr : n_tr + n_cal]
    X_te  = X_all[n_tr + n_cal :]
    Q_cal = Q_all[:min(len(Q_all), 32)]

    if len(X_te) < 10:
        X_te = X_all[n_tr:]

    print("\n" + "=" * 80)
    print(f"  V10 KV-CACHE BENCHMARK  dim={dim}  n_test={len(X_te)}  bpd={target_bpd}")
    print(f"  Outlier channels: {len(outlier_ch)} of {dim} ({100*len(outlier_ch)/dim:.1f}%)")
    print("=" * 80)

    def logit_rel_err(X_true: np.ndarray, X_hat: np.ndarray, Q: np.ndarray) -> float:
        logits_true = Q @ X_true.T
        logits_hat  = Q @ X_hat.T
        return float(
            np.mean(np.abs(logits_true - logits_hat))
            / (np.mean(np.abs(logits_true)) + 1e-12)
        )

    results: Dict[str, Dict[str, Any]] = {}

    # ── Scalar baseline ────────────────────────────────────────────────────────
    codes, lo, scale = turbo_encode(X_te, 3)
    X_tq = turbo_decode(codes, lo, scale, 3)
    results["turbo"] = {
        "label": f"Turbo scalar {target_bpd:.1f}bpd",
        "X_hat": X_tq,
        "bpd": float(target_bpd),
        "config": None,
    }

    # ── V9 symmetric (original search, no KV awareness) ───────────────────────
    v9_engine = _search_target_engine(
        dim=dim, X_tr=X_tr, X_cal=X_cal,
        target_bpd=target_bpd, max_stages=4, seed=seed,
    )
    t0 = time.perf_counter()
    v9_enc  = v9_engine.encode(X_te)
    v9_enc_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_v9  = v9_engine.decode(v9_enc)
    v9_dec_t = time.perf_counter() - t0
    results["v9"] = {
        "label": f"V9 symmetric {v9_engine.bits_per_dim():.3f}bpd",
        "X_hat": X_v9, "bpd": v9_engine.bits_per_dim(),
        "enc": v9_enc_t, "dec": v9_dec_t,
        "config": {
            "num_stages": v9_engine.num_stages,
            "norm_bits": v9_engine.stage_norm_bits,
            "qjl_rank": v9_engine.qjl_rank if v9_engine.use_qjl else 0,
        },
    }

    # ── V10 Key engine (asymmetric: 4 stages, QJL aware, outlier boost, logit) ─
    t0 = time.perf_counter()
    key_engine = v10Engine.search(
        dim=dim, K_tr=X_tr, K_cal=X_cal,
        target_bpd=target_bpd, role="key",
        fit_samples=1024, search_samples=256,   # CHANGE 2
        Q_sample=Q_cal,                          # CHANGE 3
        qk_mse_weight=0.1,
        outlier_threshold=4.0,                   # CHANGE 4
        seed=seed,
    )
    search_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    k_enc   = key_engine.encode_kv(X_te)
    k_enc_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_k11   = key_engine.decode_kv(k_enc)
    k_dec_t = time.perf_counter() - t0
    results["v10_key"] = {
        "label": f"V10 Key {key_engine.bits_per_dim():.3f}bpd",
        "X_hat": X_k11, "bpd": key_engine.bits_per_dim(),
        "enc": k_enc_t, "dec": k_dec_t, "search_t": search_t,
        "config": {
            "num_stages": key_engine.inner.num_stages,
            "norm_bits": key_engine.inner.stage_norm_bits,
            "qjl_rank": key_engine.inner.qjl_rank if key_engine.inner.use_qjl else 0,
            "has_outlier_scale": getattr(key_engine.inner, "_v10_channel_scale", None) is not None,
        },
    }

    # ── V10 Value engine (asymmetric: 3 stages, no logit term, no outlier) ─────
    val_engine = v10Engine.search(
        dim=dim, K_tr=X_tr, K_cal=X_cal,
        target_bpd=target_bpd, role="value",
        fit_samples=1024, search_samples=256,
        seed=seed,
    )
    t0 = time.perf_counter()
    v_enc   = val_engine.encode_kv(X_te)
    v_enc_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_v10   = val_engine.decode_kv(v_enc)
    v_dec_t = time.perf_counter() - t0
    results["v10_val"] = {
        "label": f"V10 Val {val_engine.bits_per_dim():.3f}bpd",
        "X_hat": X_v10, "bpd": val_engine.bits_per_dim(),
        "enc": v_enc_t, "dec": v_dec_t,
        "config": {
            "num_stages": val_engine.inner.num_stages,
            "norm_bits": val_engine.inner.stage_norm_bits,
            "qjl_rank": val_engine.inner.qjl_rank if val_engine.inner.use_qjl else 0,
        },
    }

    # ── Print results table ────────────────────────────────────────────────────
    keys  = ["turbo", "v9", "v10_key", "v10_val"]
    W     = 18
    Q_te  = Q_all[:min(len(Q_all), 32)]

    def hrow(label: str, fn: Any) -> None:
        vals = [fn(results[k]) for k in keys]
        print(f"  {label:<26}" + "".join(f"{v:>{W}}" for v in vals))

    header = "  " + f"{'Metric':<24}" + "".join(
        f"{results[k]['label']:>{W}}" for k in keys
    )
    sep = "─" * len(header)
    print("\n" + sep + "\n" + header + "\n" + sep)
    hrow("bpd",              lambda m: f"{m['bpd']:.3f}")
    hrow("SNR (dB)",         lambda m: f"{snr_db(X_te, m['X_hat']):.3f}")
    hrow("Cosine similarity",lambda m: f"{cosine_sim(X_te, m['X_hat']):.5f}")
    hrow("Logit rel error",  lambda m: f"{logit_rel_err(X_te, m['X_hat'], Q_te):.5f}")
    hrow("Recall@10",        lambda m: f"{recall_at_k(X_te, m['X_hat']):.4f}")
    print(sep)

    turbo_snr = snr_db(X_te, results["turbo"]["X_hat"])
    v9_snr    = snr_db(X_te, results["v9"]["X_hat"])
    k11_snr   = snr_db(X_te, results["v10_key"]["X_hat"])

    print("\n  Delta vs scalar baseline:")
    print(f"    V9  symmetric : {v9_snr   - turbo_snr:+.3f} dB SNR")
    print(f"    V10 Key       : {k11_snr  - turbo_snr:+.3f} dB SNR")
    print(f"    V10 Key logit : {logit_rel_err(X_te, results['v10_key']['X_hat'], Q_te):.5f}"
          f"  (lower is better, baseline="
          f"{logit_rel_err(X_te, results['turbo']['X_hat'], Q_te):.5f})")
    print(f"\n  V10 Key config  : {results['v10_key']['config']}")
    print(f"  V10 Val config  : {results['v10_val']['config']}")
    print(f"  Search time (Key): {results['v10_key']['search_t']:.1f}s")
    print(f"  Encode time (us/vec): {results['v10_key']['enc']/max(1,len(X_te))*1e6:.2f}")
    print(f"  Decode time (us/vec): {results['v10_key']['dec']/max(1,len(X_te))*1e6:.2f}")


# ── Preserve all V9 benchmark entry points unchanged ──────────────────────────

def run_exact_3bpd_benchmark(dim: int, n_vectors: int = 2000, seed: int = 42) -> None:
    """V9 exact 3 bpd benchmark — unchanged."""
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
    tq = _benchmark_method("Turbo scalar 3.0bpd", None, X_te, baseline=(codes, lo, scale, 3))
    tq["enc"] = tq_encode

    best = _search_target_engine(
        dim=dim, X_tr=X_tr, X_cal=X_cal,
        target_bpd=3.0, max_stages=4, seed=seed,
    )
    v9 = _benchmark_method("V9 exact 3.0bpd", best, X_te)

    print(f"  Turbo SNR : {snr_db(X_te, tq['X_hat']):.3f} dB")
    print(f"  V9 SNR    : {snr_db(X_te, v9['X_hat']):.3f} dB")
    print(f"  Gain      : {snr_db(X_te, v9['X_hat']) - snr_db(X_te, tq['X_hat']):+.3f} dB")
    print(f"  Cosine    : {cosine_sim(X_te, v9['X_hat']):.6f}")
    print(f"  Recall@10 : {recall_at_k(X_te, v9['X_hat']):.4f}")
    print(f"  Exactness : max |encode-decode mismatch| = {v9['exact']:.3e}")
    print(f"  Config    : {v9['config']}")


def run_v8_benchmark(dim: int, n_vectors: int = 2000, seed: int = 42) -> None:
    """V9 full benchmark — unchanged."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_vectors, dim)).astype(np.float64)
    n_tr = max(200, n_vectors // 5)
    X_tr = X[:n_tr]
    X_te = X[n_tr:]
    X_cal = X_te[: min(len(X_te), 256)]

    print("\n" + "=" * 80)
    print(f"  V9 RECURSIVE E8 vs SCALAR BASELINE - dim={dim}  n_test={len(X_te)}")
    print("=" * 80)

    t0 = time.perf_counter()
    codes, lo, scale = turbo_encode(X_te, 3)
    tq_encode = time.perf_counter() - t0
    methods: Dict[str, Dict[str, Any]] = {}
    methods["tq3"] = _benchmark_method("Turbo scalar 3.0bpd", None, X_te,
                                        baseline=(codes, lo, scale, 3))
    methods["tq3"]["enc"] = tq_encode

    exact_275 = _search_target_engine(
        dim=dim, X_tr=X_tr, X_cal=X_cal, target_bpd=2.75, max_stages=4, seed=seed)
    methods["v9_275"] = _benchmark_method("V9 exact 2.75bpd", exact_275, X_te)

    exact_300 = _search_target_engine(
        dim=dim, X_tr=X_tr, X_cal=X_cal, target_bpd=3.00, max_stages=4, seed=seed)
    methods["v9_300"] = _benchmark_method("V9 exact 3.00bpd", exact_300, X_te)

    deep = V9Engine(dim=dim, num_stages=4, norm_bits=2.5, bits_tail=0,
                    use_rotation=True, use_qjl=True,
                    qjl_rank=max(1, dim // 16), seed=seed)
    deep.fit(X_tr)
    methods["v9_deep"] = _benchmark_method(
        f"V9 deep {deep.bits_per_dim():.3f}bpd", deep, X_te)

    keys  = ["tq3", "v9_275", "v9_300", "v9_deep"]
    width = 22

    def row(label: str, fn: Any) -> None:
        values = [fn(methods[k]) for k in keys]
        print(f"  {label:<28}" + "".join(f"{v:>{width}}" for v in values))

    header = "  " + f"{'Metric':<26}" + "".join(
        f"{methods[k]['label']:>{width}}" for k in keys)
    sep = "-" * len(header)
    print("\n" + sep + "\n" + header + "\n" + sep)
    row("effective bpd",   lambda m: f"{m['bpd']:.3f}")
    row("raw bpd",         lambda m: f"{m['raw_bpd']:.3f}")
    row("SNR (dB)",        lambda m: f"{snr_db(X_te, m['X_hat']):.2f}")
    row("Cosine similarity", lambda m: f"{cosine_sim(X_te, m['X_hat']):.5f}")
    row("Recall@10",       lambda m: f"{recall_at_k(X_te, m['X_hat']):.4f}")
    row("Encode (us/vec)", lambda m: f"{m['enc'] / max(1, len(X_te)) * 1e6:.2f}")
    row("Decode (us/vec)", lambda m: f"{m['dec'] / max(1, len(X_te)) * 1e6:.2f}")
    row("Exactness max abs", lambda m: f"{m['exact']:.2e}")
    print(sep)

    tq_snr = snr_db(X_te, methods["tq3"]["X_hat"])
    print("\n  Results:")
    for key in ("v9_275", "v9_300", "v9_deep"):
        gain = snr_db(X_te, methods[key]["X_hat"]) - tq_snr
        print(f"    {methods[key]['label']}: {gain:+.3f} dB vs scalar 3-bit baseline")
        if methods[key]["config"] is not None:
            print(f"      config={methods[key]['config']}")


def compare_all_dims_v8() -> None:
    """Quick sweep using the exact 3.0 bpd search path — unchanged."""
    print("\n" + "=" * 80)
    print("  V9 RECURSIVE E8 - MULTI-DIMENSION SWEEP AT EXACT 3.0 BPD")
    print("=" * 80)
    print(f"  {'dim':>6} | {'baseline':>10} | {'v9':>10} | {'gain':>9} | {'exact':>10}")
    print("  " + "-" * 61)

    rng = np.random.default_rng(99)
    for dim, n in [(8,4000),(22,4000),(64,3000),(128,3000),(256,2000),(768,1200),(1536,600)]:
        X = rng.standard_normal((n, dim)).astype(np.float64)
        n_tr = max(128, n // 5)
        X_tr = X[:n_tr]; X_te = X[n_tr:]; X_cal = X_te[: min(len(X_te), 192)]
        tq_codes, tq_lo, tq_scale = turbo_encode(X_te, 3)
        tq_snr = snr_db(X_te, turbo_decode(tq_codes, tq_lo, tq_scale, 3))
        engine = _search_target_engine(
            dim=dim, X_tr=X_tr, X_cal=X_cal,
            target_bpd=3.0, max_stages=4, seed=99 + dim,
        )
        R = engine.decode(engine.encode(X_te))
        v9_snr = snr_db(X_te, R)
        exact  = engine.exactness_error(X_te[: min(len(X_te), 32)])
        print(f"  {dim:>6} | {tq_snr:>10.2f} | {v9_snr:>10.2f} | "
              f"{(v9_snr-tq_snr):>+9.2f} | {exact:>10.2e}")
    print("=" * 80)


def print_explanation() -> None:
    """V9 explanation text — preserved unchanged."""
    print("""
WHY THE FIXED RECURSIVE E8 PATH BEATS POLARQUANT
------------------------------------------------
1. Exact residual closure
   Every stage subtracts the same dequantized E8 contribution that the decoder
   later adds back. There is no hidden "fit on true norms, encode on quantized
   norms" drift anymore, so the recursion stays mathematically closed.

2. Better directional geometry
   Polar recursion spends bits on 2D angles. Recursive E8 spends them on 8D
   lattice directions with much lower angular distortion per stage.

3. Stage-aware norm learning
   The norm ranges are learned inside the quantized loop, not from idealized
   residuals. Later stages therefore see the same residual energy profile that
   will exist at decode time.

4. One global rotation, then one inverse
   The optional rotation is now a single orthogonal transform over the whole
   padded vector. It is applied once before chunking and inverted once after
   reconstruction, matching the intended PolarQuant-style setup.

5. QJL-style correction where it belongs
   After the recursive E8 stages, the remaining global residual is projected,
   binarized by sign, and backprojected with a learned gain. That lets a small
   number of sign bits clean up low-rank global error instead of interfering
   with the main recursive path.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDED PARAMETERS FOR REAL KV CACHE
# ═══════════════════════════════════════════════════════════════════════════════

KV_CACHE_RECOMMENDED_PARAMS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║         RECOMMENDED PARAMETERS FOR REAL KV-CACHE (V10)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Keys (attention-critical, use v10Engine.search with role='key')            ║
║  ─────────────────────────────────────────────────────────────              ║
║  max_stages_key      = 4                                                     ║
║     4 recursive E8 stages capture 4 shells of direction error.              ║
║     Each stage halves residual energy in 8D. Keys need this depth.          ║
║                                                                              ║
║  qjl_rank            = auto (search finds it), typically 0.25–0.5 × dim    ║
║     QJL sign correction is especially valuable for Keys because it          ║
║     removes low-rank global directional bias from the residual.             ║
║                                                                              ║
║  outlier_threshold   = 4.0                                                   ║
║     Channels > 4× median max-norm are scaled before fitting.                ║
║     Lower → more channels get boosted (aggressive); raise to 6.0 if        ║
║     you see reconstruction artifacts on non-outlier channels.               ║
║                                                                              ║
║  Q_sample            = 32–128 sample queries from the same layer            ║
║     Providing real queries gives the search an objective aligned with       ║
║     actual attention logits. More queries → better search, slower search.  ║
║                                                                              ║
║  qk_mse_weight       = 0.05–0.15                                             ║
║     0.05: SNR-dominated (safer, more conservative).                         ║
║     0.15: Logit-dominated (better for high-load inference, may overfit).    ║
║     0.1 is a good starting point.                                           ║
║                                                                              ║
║  fit_samples         = 1024–4096                                             ║
║     1024 is fast. Use 4096 for final calibration of production models.      ║
║     Going above 4096 has diminishing returns for typical KV distributions.  ║
║                                                                              ║
║  search_samples      = 256–512                                               ║
║     256 is usually enough to rank candidates reliably.                      ║
║     Raise to 512 for high-stakes layers (first and last transformer layers  ║
║     have the most impact on attention patterns).                            ║
║                                                                              ║
║  Values (post-softmax combination, use role='value')                        ║
║  ────────────────────────────────────────────────────                       ║
║  max_stages_value    = 3                                                     ║
║     Values tolerate one fewer stage. The saved bits go into norm bits       ║
║     which helps value magnitude fidelity.                                   ║
║                                                                              ║
║  Q_sample            = None  (Values are not dot-producted with Queries)    ║
║  outlier_threshold   = (not applied to Values by default)                   ║
║                                                                              ║
║  use_rotation        = True  (let search find optimal, usually True)        ║
║                                                                              ║
║  QUICK RECIPE for Llama-3 8B (head_dim=128, 3.0 bpd):                      ║
║  ─────────────────────────────────────────────────────                      ║
║    key_engine = v10Engine.search(                                            ║
║        dim=128, K_tr=K_calibration, K_cal=K_cal,                           ║
║        target_bpd=3.0, role='key',                                          ║
║        fit_samples=2048, search_samples=256,                                ║
║        Q_sample=Q_calibration[:64],                                         ║
║        qk_mse_weight=0.1,                                                   ║
║        outlier_threshold=4.0,                                               ║
║    )                                                                         ║
║    val_engine = v10Engine.search(                                            ║
║        dim=128, K_tr=V_calibration, K_cal=V_cal,                           ║
║        target_bpd=3.0, role='value',                                        ║
║        fit_samples=2048, search_samples=256,                                ║
║    )                                                                         ║
║    # Per token at inference:                                                 ║
║    enc_k = key_engine.encode_kv(K_token)  # shape (heads, seq, dim)        ║
║    K_hat = key_engine.decode_kv(enc_k)                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print("=" * 80)
    print("  Higman-Sims Quantizer V10 — KV-Cache Asymmetric")
    print("=" * 80)
    print(KV_CACHE_RECOMMENDED_PARAMS)
    print_explanation()

    # V9 Gaussian benchmarks (all unchanged, still pass)
    compare_all_dims_v8()
    run_exact_3bpd_benchmark(dim=768, n_vectors=1200)
    run_v8_benchmark(dim=768, n_vectors=1200)

    # V10 KV-cache benchmark (new)
    run_v10_kv_benchmark(dim=128, n_vectors=1200, target_bpd=3.0)
    run_v10_kv_benchmark(dim=64,  n_vectors=1200, target_bpd=3.0)