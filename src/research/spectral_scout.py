import torch

class SpectralScout:
    """
    Spectral Scout: Randomized SVD for Real-Time Error Isolation.
    ==========================================================
    Identifies the 'Structural Ground' of a KV cache or hidden state.
    """
    @staticmethod
    def find_spectral_ground(x, rank=64, energy_threshold=0.9):
        """
        Identify high-importance singular vectors using Randomized SVD.
        """
        device = x.device
        dtype = x.dtype
        n, d = x.shape
        
        # 1. Generate random projection matrix
        omega = torch.randn(d, rank + 10, device=device, dtype=dtype)
        
        # 2. Project input to low-rank subspace
        y = x @ omega
        
        # 3. Orthonormalize (QR decomposition)
        q, _ = torch.linalg.qr(y)
        
        # 4. Project back and get singular values
        b = q.T @ x
        u_tilde, s, v_t = torch.linalg.svd(b, full_matrices=False)
        
        # 5. Extract the 'Ground' mask
        # We find the cumulative energy and threshold it
        cum_energy = torch.cumsum(s**2, dim=0)
        total_energy = cum_energy[-1]
        ground_cutoff_idx = torch.where(cum_energy / total_energy >= energy_threshold)[0]
        
        if len(ground_cutoff_idx) > 0:
            cutoff = ground_cutoff_idx[0].item()
        else:
            cutoff = rank
            
        # Return the top V_t vectors representing the 'Ground'
        ground_vectors = v_t[:cutoff+1]
        
        return ground_vectors, s[:cutoff+1]

if __name__ == "__main__":
    # Test on a Mock Activation (Low-Rank Signal + Noise)
    print("--- Spectral Scout: Randomized SVD Test ---")
    dim = 4096
    
    # Generate Low-Rank Ground (Rank 16)
    ground_source = torch.randn(dim, 16)
    signal = torch.randn(100, 16) @ ground_source.T
    noise = torch.randn(100, dim) * 0.1
    X = signal + noise
    
    # Run Scout
    vectors, values = SpectralScout.find_spectral_ground(X, rank=32)
    print(f"Captured Ground Rank: {vectors.shape[0]}")
    print(f"Top Singular Values:  {values[:5].tolist()}")
    print("Spectral Isolation SUCCESS.")
