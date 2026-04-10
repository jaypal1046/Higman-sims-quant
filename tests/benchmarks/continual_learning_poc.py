import torch
import numpy as np
from src.engine.lattice_memory import LatticeMemory

def run_poc():
    print("--- Phase 2 POC: Continual Learning & Selective Forgetting ---")
    dim = 128
    mem = LatticeMemory(dim=dim, capacity=10)
    
    # 1. Store Base Knowledge
    print("\n1. Injecting Knowledge...")
    k1 = torch.randn(dim)
    v1 = torch.tensor([1.0, 2.0, 3.0] + [0.0]*(dim-3)) # "Fact A"
    mem.store(k1, v1)
    
    k2 = torch.randn(dim)
    v2 = torch.tensor([9.0, 8.0, 7.0] + [0.0]*(dim-3)) # "Fact B"
    mem.store(k2, v2)
    
    # 2. Verify Retrieval
    print("\n2. Verifying Retrieval...")
    r1 = mem.retrieve(k1)
    r2 = mem.retrieve(k2)
    
    match1 = torch.allclose(v1, r1, atol=1e-5)
    match2 = torch.allclose(v2, r2, atol=1e-5)
    print(f"Retrieval Fact A: {'SUCCESS' if match1 else 'FAILED'}")
    print(f"Retrieval Fact B: {'SUCCESS' if match2 else 'FAILED'}")
    
    # 3. Selective Forgetting (Unlearning Fact A)
    print("\n3. Applied Forget Syndrome to Fact A (Slot 0)...")
    mem.forget(0)
    
    # 4. Final verification
    print("\n4. Final State Verification...")
    r1_after = mem.retrieve(k1) # Should fail or return Fact B (nearest remaining)
    r2_after = mem.retrieve(k2) # Should still succeed
    
    match1_after = torch.allclose(v1, r1_after, atol=1e-5)
    match2_after = torch.allclose(v2, r2_after, atol=1e-5)
    
    print(f"Retrieval Fact A (Post-Forget): {'STILL EXISTS' if match1_after else 'ERASED SUCCESS'}")
    print(f"Retrieval Fact B (Integrity):   {'STABLE' if match2_after else 'CORRUPTED'}")
    
    if not match1_after and match2_after:
        print("\n✅ PHASE 2 COMPLETE: Lattice Memory demonstrated Online Learning and Selective Forgetting.")

if __name__ == "__main__":
    run_poc()
