import sys
import os
import torch
import numpy as np
import time

# Add project root to path
sys.path.append(os.getcwd())

from src.higman_sims_hybrid import HigmanSims_Hybrid

def load_dolma_sample(file_path, n_samples=2000):
    print(f"--- Loading {n_samples} vectors from Dolma Dataset ---")
    X = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            parts = line.split()
            # First part is word, rest are vectors
            vec = [float(x) for x in parts[1:]]
            X.append(vec)
    
    X = torch.tensor(X, dtype=torch.float32)
    print(f"Shape: {X.shape}")
    return X

def calculate_snr(original, reconstructed):
    noise = original - reconstructed
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def run_dolma_god_mode():
    data_path = r"c:\Jay\_Plugin\Higman sims quant\data\glove\dolma_300_2024_1.2M.100_combined.txt"
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    # Load data
    X = load_dolma_sample(data_path, n_samples=2048) # Use power of 2 for better alignment
    
    # Initialize Hybrid in God-Mode (V16 @ 8.5 BPD)
    print("\n--- Initializing GOD-MODE Hybrid (V16) ---")
    hybrid = HigmanSims_Hybrid(dim=X.shape[1], target_bpd=8.5)
    
    # Fit & Quantize
    print("Quantizing...")
    start_time = time.time()
    hybrid.fit(X)
    codes = hybrid.encode(X)
    X_hat = hybrid.decode(codes)
    elapsed = time.time() - start_time
    
    # Analytics
    snr = calculate_snr(X, X_hat)
    exact_bpd, theoretical_bpd = hybrid.measure_efficiency(codes)
    
    print("\n" + "="*50)
    print("GOD-MODE DOLMA BENCHMARK RESULTS")
    print("="*50)
    print(f"Data Source: Dolma 300D Embeddings")
    print(f"Samples: {X.shape[0]}")
    print(f"Target BPD: 8.5")
    print("-" * 50)
    print(f"Final SNR: {snr:.2f} dB")
    print(f"Measured BPD: {exact_bpd:.2f}")
    print(f"Theoretical Entropy: {theoretical_bpd:.2f} BPD")
    print(f"Status: {'BIT-EXACT SINGULARITY' if snr > 100 else 'LOSSY'}")
    print(f"Time Taken: {elapsed:.2f}s")
    print("="*50)

if __name__ == "__main__":
    run_dolma_god_mode()
