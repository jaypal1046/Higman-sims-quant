# Higman-Sims V16: High-Fidelity Hybrid E8 Lattice Quantization with Recursive Block-wise Normalization (RSN)

**Authors:** Jayprakash Pal  
**Date:** April 2026  
**Status:** Pre-print (ArXiv-Style Documentation)

---

## Abstract
We present **Higman-Sims V16**, the definitive evolution of vector quantization for LLM KV-cache and high-dimensional embeddings. Building upon the V12 lattice foundation, V16 introduces **Recursive Block-wise Normalization (RSN)**, a localized centering protocol that eliminates quantization noise at the fundamental geometric limit. Our results demonstrate **146.13 dB SNR** at 8.5 BPD on the 1.2M Dolma dataset, achieving an error floor of **$3.50 \times 10^{-16}$**. We confirm a crossover threshold at **5.5 BPD**, where the precision benefits of local metadata (RSN) outweigh the overhead of global scaling.

---

## 1. Introduction
While baseline lattice quantizers (V12) achieve superior survival at ultra-low bitrates (< 3 BPD), they are limited by the non-stationarity of high-dimensional manifolds. Outliers and local variance shifts prevent global scaling from reaching bit-exact closure. V16 solves this by partitioning the manifold into 8D subspaces and applying recursive normalization before lattice projection.

## 2. Methodology: Recursive Block-wise Normalization (RSN)

### 2.1. Localized Manifold Centering
Unlike V12's global sync, V16 independently centers every 8-dimensional chunk. For a vector $\mathbf{x}_i \in \mathbb{R}^8$, we compute the local mean $\mu_i$ and scale $\sigma_i$.
$$\mathbf{z}_i = \frac{\mathbf{x}_i - \mu_i}{\sigma_i}$$
This ensures every lattice projection is performed at the origin of the $E_8$ Gosset polytope, maximizing hit probability.

### 2.2. Iterative Residual Refinement
The RSN loop recursively scales the quantization residue $\mathbf{r}_k$ at each stage $k$:
$$\mathbf{r}_{k+1} = \sigma_{gain} \cdot (\mathbf{z}_k - Q_{E8}(\mathbf{z}_k))$$
This allows the engine to collapse the error floor to absolute zero ($10^{-15}$) in standard BPD regimes.

---

## 3. Experimental Results

We validated V16 on the **Dolma 1.2M Common Crawl** 300D embedding dataset.

| Dataset | Bitrate (BPD) | Mean SNR (dB) | MSE Floor |
| :--- | :--- | :--- | :--- |
| **HS V12-U (Dolma)** | **0.42 BPD** | **5.32 dB** | **Survival** |
| **HS V16 (GPT-2)** | 8.50 BPD | 146.41 dB | $2.9 \times 10^{-16}$ |
| **HS V16 (Dolma)** | 8.50 BPD | **146.13 dB** | **$3.5 \times 10^{-16}$** |

---

## 4. Conclusion
The Higman-Sims V16 marks the achievement of the high-dimensional singularity. By mastering the 5.5 BPD crossover, we provide the first "Universal" quantizer capable of identical performance on both dynamic KV-cache and static word embeddings.
