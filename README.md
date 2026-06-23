# Lattice-RSN for KV Cache Quantization

Lattice-RSN is a research prototype for KV-cache quantization built around E8 lattice projection, recursive residual refinement, and several normalization strategies explored across `v12` through `v19`.

The project is interesting because it treats low-bitrate and high-fidelity regimes as different problems instead of forcing one codec design to cover everything. The current repository contains working quantizers, validation scripts, and exploratory integration work for transformer inference.

## Current Status

- This is an active research prototype, not a production codec.
- The core quantizers run and reproduce the local reconstruction results in this repo.
- Preliminary end-to-end cache evaluation is now available for short teacher-forced autoregressive runs.
- Hardware latency, serialized codec size, and broad model-level benchmarking are still incomplete.

## What Is In The Repo

- `src/core/v12.py` to `src/core/v19.py`
  Different quantization variants with different bitrate and fidelity trade-offs.
- `src/research/llm_eval.py`
  End-to-end autoregressive evaluation with quantized `past_key_values`.
- `src/engine/`
  Early runtime and integration experiments for PyTorch, Triton, and Hugging Face layer replacement.
- `tests/`
  Validation scripts, benchmarks, and compatibility checks.
- `docs/research/`
  Research notes and architecture summaries.

## Main Idea

The project explores three linked ideas:

1. Partition activations into 8D chunks and project them onto the E8 lattice.
2. Normalize either globally or locally before quantization, depending on bitrate regime.
3. Add recursive residual refinement so that high-bitrate modes can drive reconstruction error very low.

The earlier `v12` path emphasizes aggressive compression with limited metadata. The later `v16` to `v19` paths explore more local normalization and staged refinement to improve fidelity.

## What The Current Evidence Supports

Local results in this repository suggest that the approach is promising:

- Reconstruction-focused tests reach very high SNR in the higher-bitrate variants.
- The PyTorch E8 kernel matches the NumPy reference in parity checks.
- The end-to-end evaluator in `src/research/llm_eval.py` now quantizes the KV cache inside an actual autoregressive loop.

Example short-run results from the current local GPT-2 evaluation:

- `v19`, 96 tokens, teacher-forced streaming:
  baseline PPL `50.188924`, quantized PPL `50.184533`, top-1 agreement `98.95%`
- `v19`, 64 tokens, teacher-forced streaming:
  baseline PPL `80.140889`, quantized PPL `80.011448`, top-1 agreement `100.00%`
- `v16`, 64 tokens, teacher-forced streaming:
  baseline PPL `80.140889`, quantized PPL `80.140813`, top-1 agreement `100.00%`

These runs are useful because they show that the quantized-cache path can operate end to end without obvious degradation in short tests. They should still be treated as preliminary rather than definitive.

## What Is Not Yet Proven

The repository does not yet prove that this is the best KV-cache quantization method.

Open work includes:

- evaluation across more models
- longer contexts
- standardized datasets
- direct comparison against strong baselines under the same protocol
- actual serialized bitrate instead of entropy-style estimates alone
- production latency measurements with Triton or another fused runtime

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run A Core Reconstruction Check

```bash
python src/core/v16.py
python src/core/v19.py
```

### Run The Kernel Parity Check

```bash
python tests/integration/validation_parity.py
```

### Run End-To-End Cache Evaluation

```bash
python src/research/llm_eval.py --model openai-community/gpt2 --engine v19 --target-bpd 2.0 --max-stages 6 --max-length 256 --calibration-length 64 --device cpu
```

## Recommended Framing

The most accurate way to describe the project today is:

- a working research prototype
- promising early end-to-end results
- strong reconstruction behavior in local experiments
- not yet a settled production system
- not yet proven to be best in class

## Notes

- Some older scripts and class names still use more dramatic internal naming from earlier iterations. The public-facing docs are being revised toward more neutral language.
- The TeX manuscript under `paper/` may still contain older wording if it is being edited separately.

Last updated: June 23, 2026.
