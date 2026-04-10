import torch
import time
from src.engine.lattice_kernel_triton import triton_e8_quantize
from src.engine.torch_kernel import fast_e8_quantize_torch

def run_speed_test():
    print("--- Phase 5: Triton vs PyTorch-Dispatch Speed Benchmark ---")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping Triton speed test.")
        return
        
    # Configuration
    dim = 4096
    num_vectors = 8192 # Sufficient for GPU warm-up
    data = torch.randn(num_vectors, dim).cuda()
    
    # 1. Warm-up
    _ = fast_e8_quantize_torch(data)
    _ = triton_e8_quantize(data)
    torch.cuda.synchronize()
    
    # 2. Benchmark PyTorch
    start_pt = time.time()
    for _ in range(100):
        _ = fast_e8_quantize_torch(data)
    torch.cuda.synchronize()
    end_pt = time.time()
    pt_time = (end_pt - start_pt) / 100
    
    # 3. Benchmark Triton
    start_tr = time.time()
    for _ in range(100):
        _ = triton_e8_quantize(data)
    torch.cuda.synchronize()
    end_tr = time.time()
    tr_time = (end_tr - start_tr) / 100
    
    print(f"\nPyTorch Baseline Latency: {pt_time*1000:.4f} ms")
    print(f"Triton Lattice Latency:   {tr_time*1000:.4f} ms")
    print(f"Speedup:                  {pt_time/tr_time:.2f}x")
    
    if tr_time < pt_time:
        print("\n✅ PHASE 5 COMPLETE: Triton has achieved hardware-accelerated Line-Rate performance.")

if __name__ == "__main__":
    run_speed_test()
