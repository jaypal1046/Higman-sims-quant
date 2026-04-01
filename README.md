# Higman-Sims Implicit Spectral Quantizer

Research prototype for hybrid vector quantization built around the Higman-Sims graph, its 22-dimensional spectral embedding, and a scalar residual stage.

The main reference implementation for this README is [src/higman_sims_quant.py](src/higman_sims_quant.py). Later files such as `v2` through `v7` are exploratory variants and should be treated as experiments rather than settled results.

## Project Status

- This repository is a research prototype, not a production codec.
- The original `higman_sims_quant.py` implementation uses an implicit Higman-Sims codebook plus scalar residual quantization.
- The repository contains local benchmark comparisons against a simple in-repo scalar baseline sometimes labeled `TurboQuant` in older scripts.
- The repository should not be described as proving that it beats Google's official TurboQuant paper or system.
- Most numbers in this repository come from synthetic Gaussian benchmarks unless a file explicitly says otherwise.

## Why This Approach Is Interesting

This project is still a prototype, but it does have some real strengths:

- The coarse codebook is analytic in the original HS prototype, so there is no learned centroid table to serialize with each compressed payload.
- Decode is simple: one row lookup per 22D chunk, residual dequantization, then addition.
- The Higman-Sims embedding is highly symmetric, which makes it a clean and mathematically structured coarse quantizer to study.
- The design is easy to inspect end to end because the codebook comes from explicit combinatorics rather than opaque training.
- On the repository's local synthetic benchmarks, the original HS hybrid can outperform a simple scalar baseline on reconstruction quality at the same residual setting.
- The repository is useful for exploring quality-first quantization ideas before moving to harder benchmarks such as real embeddings, retrieval, or system-level inference workloads.

## Quick Start

### 1. Requirements

- Python 3.10 or higher
- `pip`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install the minimal dependencies directly:

```bash
pip install numpy scipy
```

### 3. Run the original HS benchmark

From the repository root:

```bash
python .\src\higman_sims_quant.py
```

If you `cd` into `src`, you can also run:

```bash
python .\higman_sims_quant.py
```

Expected behavior:

- the script builds the Golay-derived HS embedding once
- runs a decode-scaling check
- prints benchmark tables for several dimensions

Exact timing and quality numbers will vary by machine, Python version, and random seed.

### 4. Use it in your own code

From the repository root:

```python
from src.higman_sims_quant import HybridQuantizer
import numpy as np

qz = HybridQuantizer(dim=768, bits_residual=2, verbose=True)

vectors = np.random.randn(100, 768).astype(np.float64)

compressed = qz.encode(vectors)
reconstructed = qz.decode(compressed)

mse = np.mean((vectors - reconstructed) ** 2)
print(f"MSE: {mse:.4f}")
print(f"Bits per vector: {qz.bits_per_vector()}")
print(f"Compression ratio: {qz.compression_ratio():.2f}x")
print(f"Overhead: {qz.overhead_bits()} bits")
```

### 5. Use the raw 22D quantizer directly

```python
from src.higman_sims_quant import HSQuantizer
import numpy as np

qz = HSQuantizer(bits_residual=2, verbose=True)

x = np.random.randn(50, 22).astype(np.float64)

ids, residual_codes = qz.encode(x)
x_hat = qz.decode(ids, residual_codes)

print(f"Vertex ID range: {ids.min()}..{ids.max()}")
print(f"Bits per 22D vector: {qz.bits_per_vector()}")
```

## What This Project Is

This project explores a hybrid quantization idea:

1. Use a fixed analytic 22D codebook derived from the Higman-Sims graph as a coarse quantizer.
2. Quantize only the residual error with a small scalar code.

The codebook is implicit rather than learned. Encoder and decoder both reconstruct it from the same mathematical construction, so there is no transmitted learned-codebook payload in the original HS prototype.

This repository is about:

- analytic codebooks
- graph- and design-based geometry
- compact reconstruction
- bitrate accounting
- decode simplicity

This repository is not yet a validated claim against Google's official TurboQuant implementation.

## High-Level Idea

Instead of quantizing each dimension independently from the start, the HS prototype first snaps a 22D input vector to the nearest point in a fixed 100-point spherical code derived from the Higman-Sims graph. The residual error is then quantized with a scalar residual code.

The motivation is simple:

- if the coarse codebook already captures a useful part of the vector geometry
- the remaining residual is smaller
- and the scalar stage has less work to do

## Technical Overview

### Step 1: Extended Golay code

The construction starts from the binary extended Golay code `[24, 12, 8]`.

- It has `2^12 = 4096` codewords.
- Exactly `759` of them have Hamming weight `8`.
- Those weight-8 codewords are the octads of the large Witt design `S(5, 8, 24)`.

In the implementation this is generated directly from a fixed systematic generator matrix.

### Step 2: Steiner system `S(3, 6, 22)`

Fix two points in the `S(5, 8, 24)` design, then restrict to octads containing both of them. Removing those fixed points yields `77` six-element blocks on the remaining `22` points.

That gives the Steiner system `S(3, 6, 22)` used by the next stage of the construction.

### Step 3: Higman-Sims graph

From the `S(3, 6, 22)` blocks, the code builds the Higman-Sims graph with `100` vertices:

- `1` special vertex
- `22` point vertices
- `77` block vertices

The resulting graph is the strongly regular graph `srg(100, 22, 0, 6)`.

### Step 4: Spectral embedding

The Higman-Sims adjacency matrix has three eigenvalues:

- `22` with multiplicity `1`
- `2` with multiplicity `77`
- `-8` with multiplicity `22`

The implementation uses the `-8` eigenspace to build a `100 x 22` embedding matrix. Each row becomes one coarse codebook vector.

This gives a highly symmetric 22D spherical configuration that is useful as an implicit coarse codebook.

### Step 5: Encoding a 22D vector

For one 22D input vector `x`:

1. Compute all scores `x @ V.T`.
2. Pick the nearest HS vertex ID.
3. Use that row of `V` as the coarse approximation.
4. Compute the residual `x - coarse`.
5. Quantize the residual with `bits_residual` bits per dimension.

The output is:

- one 7-bit vertex ID
- plus residual codes

### Step 6: Decoding

Decoding a 22D chunk consists of:

1. one row lookup from the fixed `100 x 22` embedding matrix
2. residual dequantization
3. vector addition

For higher dimensions, the implementation pads to a multiple of `22`, splits into chunks, and applies the same process per chunk.

## What "O(1) decode" Means Here

In this repository, `O(1)` decode means constant-time lookup with respect to the codebook size for one 22D chunk:

- decode does not search all 100 vertices
- it simply reads the row identified by the stored ID

Decode still scales linearly with:

- batch size
- number of chunks
- total dimension

So for full vectors, decode is constant per chunk and linear in the number of chunks.

## Original HS Prototype Numbers

For the original `higman_sims_quant.py` prototype with `bits_residual = 2`:

| Dimension | Bits/vector | Compression ratio |
| --- | ---: | ---: |
| 22 | 51 | 13.80x |
| 64 | 153 | 13.39x |
| 768 | 1785 | 13.77x |
| 4096 | 9537 | 13.74x |

These are properties of the original HS chunked codec in this repository.

## Benchmark Interpretation

The original script prints a comparison against a simple scalar baseline that older files sometimes call `TurboQuant`.

The correct way to read that comparison is:

- it is a local benchmark inside this repository
- it is useful for internal iteration
- it is not the same thing as reproducing Google's official TurboQuant system

So a safe summary is:

- the original HS prototype can look better than a simple local scalar baseline on some synthetic reconstruction metrics
- that does not by itself establish a win over Google's published method

## Why the Codebook Overhead Is Called "Implicit"

In the original HS prototype, the codebook is not learned from data and is not serialized with the compressed payload. Instead, both encoder and decoder rebuild it from the same deterministic construction:

Golay generator matrix -> octads -> `S(3, 6, 22)` blocks -> HS adjacency matrix -> eigendecomposition -> 22D embedding

That is why the original implementation describes the coarse codebook as implicit.

## Project Structure

```text
src/
  higman_sims_quant.py       Original HS implicit spectral quantizer
  higman_sims_quant_v2.py    Follow-up experiment
  higman_sims_quant_v3.py    Equal-bitrate diagnostics
  higman_sims_quant_v4.py    Direction-first variant
  higman_sims_quant_v5.PY    Historical experiment with known quality issues
  higman_sims_quant_v6.py    E8-based reference prototype
  higman_sims_quant_v7.py    Latest norm-split E8 research prototype

tests/
  *.test.py                  Construction and graph checks

docs/
  *.md / *.html             Historical notes and benchmark artifacts
```

## Main Classes

### `HSQuantizer`

Core 22D quantizer.

- builds the HS embedding
- encodes `(N, 22)` arrays into IDs plus residual codes
- decodes by row lookup plus residual reconstruction

### `HybridQuantizer`

Wrapper for arbitrary dimension.

- pads to a multiple of 22
- chunks the input
- applies `HSQuantizer` to each chunk

### `CompressedVector`

Named tuple holding:

- `ids`
- `codes`
- original dimension
- batch size

## Known Limitations

- The main benchmark is synthetic Gaussian data.
- The original encoder still uses a direct nearest-vertex search over all 100 HS vertices.
- Chunking high-dimensional vectors into independent 22D blocks ignores cross-chunk correlation.
- The residual stage is still a simple scalar quantizer.
- Later experimental versions in this repository show that bitrate accounting must be handled very carefully.

## Suggested Next Steps

- evaluate on real embeddings or geometry data
- test retrieval metrics and application-level metrics
- improve search speed on the coarse codebook
- improve the residual stage
- compare against a closer reproduction of the official TurboQuant paper if a paper-level claim is needed

## Public Release Checklist

If you are deciding whether to make the repository public, use [docs/PUBLIC_RELEASE_CHECKLIST.md](docs/PUBLIC_RELEASE_CHECKLIST.md).

## Legal

- License: [LICENSE](LICENSE)
- Disclaimer: [DISCLAIMER.md](DISCLAIMER.md)
- Terms and Conditions: [TERMS_AND_CONDITIONS.md](TERMS_AND_CONDITIONS.md)

## References

1. D. G. Higman and C. C. Sims, "A simple group of order 44,352,000."
2. A. E. Brouwer, A. M. Cohen, and A. Neumaier, *Distance-Regular Graphs*.
3. P. Delsarte, "An algebraic approach to the association schemes of coding theory."
4. J. H. Conway and N. J. A. Sloane, *Sphere Packings, Lattices and Groups*.
5. Google TurboQuant paper and blog posts, for context only; this repository does not currently claim an official reproduction or benchmark win over that system.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
