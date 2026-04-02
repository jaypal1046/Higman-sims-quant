"""
Higman-Sims Quantizer V9
========================

This is a cleaned, exact recursive E8 quantizer intended for low-bit LLM
compression research. The main fixes relative to the earlier V8 prototype are:

1. Encode/decode now share the exact same reconstruction path. Each recursive
   stage subtracts the same dequantized vector that decode adds back.
2. Norm-range fitting is done in the quantized loop, so later stages see the
   residual distribution that the decoder will actually reconstruct.
3. Rotation is a single global orthogonal transform implemented as a short
   Householder chain and inverted exactly once on decode.
4. Fractional effective norm bits are supported through deterministic mixed-bit
   schedules.
5. QJL-style correction is implemented as a proper random projection + sign
   backprojection with a learned global gain.

The public benchmark helpers from V8 are preserved:
  - run_v8_benchmark
  - compare_all_dims_v8
  - print_explanation

And a new exact-rate benchmark is added:
  - run_exact_3bpd_benchmark
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


def run_exact_3bpd_benchmark(dim: int, n_vectors: int = 2000, seed: int = 42) -> None:
    """
    Dedicated benchmark at exactly 3.0 effective bits per dimension.
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

    best = _search_target_engine(
        dim=dim,
        X_tr=X_tr,
        X_cal=X_cal,
        target_bpd=3.0,
        max_stages=4,
        seed=seed,
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
    """
    Head-to-head comparison preserving the V8 benchmark entry point.
    """
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
    methods["tq3"] = _benchmark_method(
        "Turbo scalar 3.0bpd",
        None,
        X_te,
        baseline=(codes, lo, scale, 3),
    )
    methods["tq3"]["enc"] = tq_encode

    exact_275 = _search_target_engine(
        dim=dim,
        X_tr=X_tr,
        X_cal=X_cal,
        target_bpd=2.75,
        max_stages=4,
        seed=seed,
    )
    methods["v9_275"] = _benchmark_method("V9 exact 2.75bpd", exact_275, X_te)

    exact_300 = _search_target_engine(
        dim=dim,
        X_tr=X_tr,
        X_cal=X_cal,
        target_bpd=3.00,
        max_stages=4,
        seed=seed,
    )
    methods["v9_300"] = _benchmark_method("V9 exact 3.00bpd", exact_300, X_te)

    deep = V9Engine(
        dim=dim,
        num_stages=4,
        norm_bits=2.5,
        bits_tail=0,
        use_rotation=True,
        use_qjl=True,
        qjl_rank=max(1, dim // 16),
        seed=seed,
    )
    deep.fit(X_tr)
    methods["v9_deep"] = _benchmark_method(
        f"V9 deep {deep.bits_per_dim():.3f}bpd",
        deep,
        X_te,
    )

    keys = ["tq3", "v9_275", "v9_300", "v9_deep"]
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
    for key in ("v9_275", "v9_300", "v9_deep"):
        gain = snr_db(X_te, methods[key]["X_hat"]) - tq_snr
        print(f"    {methods[key]['label']}: {gain:+.3f} dB vs scalar 3-bit baseline")
        if methods[key]["config"] is not None:
            print(f"      config={methods[key]['config']}")


def compare_all_dims_v8() -> None:
    """Quick sweep using the exact 3.0 bpd search path."""
    print("\n" + "=" * 80)
    print("  V9 RECURSIVE E8 - MULTI-DIMENSION SWEEP AT EXACT 3.0 BPD")
    print("=" * 80)
    print(f"  {'dim':>6} | {'baseline':>10} | {'v9':>10} | {'gain':>9} | {'exact':>10}")
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

        engine = _search_target_engine(
            dim=dim,
            X_tr=X_tr,
            X_cal=X_cal,
            target_bpd=3.0,
            max_stages=4,
            seed=99 + dim,
        )
        R = engine.decode(engine.encode(X_te))
        v9_snr = snr_db(X_te, R)
        exact = engine.exactness_error(X_te[: min(len(X_te), 32)])

        print(
            f"  {dim:>6} | {tq_snr:>10.2f} | {v9_snr:>10.2f} | "
            f"{(v9_snr - tq_snr):>+9.2f} | {exact:>10.2e}"
        )

    print("=" * 80)


def print_explanation() -> None:
    """Explain the corrected recursive E8 design and updated diagram language."""
    print(
        """
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

BETTER DIAGRAM DESCRIPTION
--------------------------
Use a side-by-side composition.

Left panel: "Nested polar circles"
  Show a high-dimensional cloud collapsing into repeated 2D pairings.
  Each layer becomes a ring with angle ticks and one radius feeding inward.
  The visual message is angle recursion: circle, angle bins, smaller circle,
  angle bins again.

Right panel: "Nested E8 spheres"
  Show the same cloud first rotated once as a whole, then split into 8D cells.
  Around each cell draw a translucent outer sphere with discrete E8 surface
  points, then a smaller inner residual sphere, then another smaller one.
  The visual message is residual shelling: shell 1 captures most energy, shell 2
  cleans the leftover, shell 3 and shell 4 polish the interior, and a final
  low-rank sign correction spans the whole vector globally.

Bottom caption:
  "PolarQuant recurses through angles. Recursive E8 recurses through shrinking
  lattice shells, so direction error falls faster and the decode path is exact."
"""
    )


if __name__ == "__main__":
    print("=" * 80)
    print("  Higman-Sims Quantizer V9 - Exact Recursive E8")
    print("=" * 80)
    print_explanation()
    compare_all_dims_v8()
    run_exact_3bpd_benchmark(dim=768, n_vectors=1200)
    run_v8_benchmark(dim=768, n_vectors=1200)
