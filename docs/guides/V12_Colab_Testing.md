# V12 THE-UNTOUCHABLE Colab Execution Suite

Follow these steps to extract realistic LLM KV Cache data on Google Colab, compress it using the V12 Untouchable Quantizer, and capture the exact SNR metrics for our research paper.

### Step 1: Install Dependencies
Run this in the first Colab cell:
```python
!pip install torch torchvision torchaudio transformers accelerate huggingface_hub numpy
```

### Step 2: Extract Real KV Cache from Gemma-2B
Run this cell to load Gemma and pull legitimate multi-layer Keys and Values representations.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

model_name = "google/gemma-2b"
# Log in if gated: from huggingface_hub import login; login()

print("Downloading Model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map="cpu"
)

prompt = "Explain in extreme detail how the E8 lattice and Higman-Sims geometries can compress vector systems mathematically. Provide proofs."
inputs = tokenizer(prompt, return_tensors="pt")

print("Extracting KV Cache...")
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

# Aggregate Cache: DynamicCache format -> outputs.past_key_values.key_cache
keys_list = [layer_cache[0] for layer_cache in outputs.past_key_values]
values_list = [layer_cache[1] for layer_cache in outputs.past_key_values]

# Flatten arrays for quantization: (samples, dimension)
keys_arr = torch.cat(keys_list, dim=0).cpu().numpy().astype(np.float32)
values_arr = torch.cat(values_list, dim=0).cpu().numpy().astype(np.float32)

head_dim = keys_arr.shape[-1]
X_keys = keys_arr.reshape(-1, head_dim)
X_values = values_arr.reshape(-1, head_dim)

print(f"✅ Extracted Keys shape: {X_keys.shape}")
print(f"✅ Extracted Values shape: {X_values.shape}")
```

### Step 3: Define The V12 Untouchable Algorithm
Run the complete V12 quantization logic in the next cell.
```python
import math
import numpy as np

def build_e8():
    v = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    x = np.zeros(8)
                    x[i], x[j] = si, sj
                    v.append(x)
    for m in range(256):
        s = np.array([0.5 if ((m >> k) & 1) == 0 else -0.5 for k in range(8)])
        if np.sum(s < 0) % 2 == 0: v.append(s)
    return np.array(v) / math.sqrt(2.0)

class Untouchable_Core:
    """Extreme Fidelity Lattice Sync (8-Stage E8)."""
    def __init__(self, dim, stages=8):
        self.dim = int(dim)
        self.stages = int(stages)
        self.nch = int(math.ceil(dim/8))
        self.pw = self.nch * 8
        self.CB8 = build_e8()
        self.CB8T = self.CB8.T.copy()
        self.m = None
        self.md = None
        self.rngs = []

    def fit(self, X):
        self.m = np.median(X, axis=0)
        self.md = np.median(np.abs(X - self.m), axis=0) + 1e-12
        Xn = (X - self.m) / self.md
        res = np.pad(Xn, ((0,0), (0, self.pw - self.dim))).reshape(-1, self.nch, 8)
        self.rngs = []
        for s in range(self.stages):
            idx = np.argmax(res @ self.CB8T, axis=2)
            dots = np.sum(res * self.CB8[idx], axis=2)
            lo, hi = float(np.quantile(dots, 0.001)), float(np.quantile(dots, 0.999))
            self.rngs.append((lo, hi))
            sp = max(hi - lo, 1e-6)
            lvls = 15  # 4-bit representation
            q = np.floor(np.clip((dots - lo) / sp, 0, 0.9999) * lvls).astype(int) + 1
            h = lo + (q.astype(float) - 0.5) / lvls * sp
            res -= self.CB8[idx] * (h[..., None])

    def encode(self, X):
        nv = len(X)
        Xn = (X - self.m) / self.md
        res = np.pad(Xn, ((0,0), (0, self.pw - self.dim))).reshape(-1, self.nch, 8)
        a_idx, a_q = [], []
        for s in range(self.stages):
            idx = np.argmax(res @ self.CB8T, axis=2)
            dots = np.sum(res * self.CB8[idx], axis=2)
            lo, hi = self.rngs[s]
            sp = max(hi - lo, 1e-6)
            lvls = 15
            q = np.floor(np.clip((dots - lo) / sp, 0, 0.9999) * lvls).astype(int) + 1
            h = lo + (q.astype(float) - 0.5) / lvls * sp
            res -= self.CB8[idx] * (h[..., None])
            a_idx.append(idx)
            a_q.append(q)
        return a_idx, a_q

    def decode(self, co):
        idx, qs = co
        nv = idx[0].shape[0]
        res = np.zeros((nv, self.nch, 8))
        for s in reversed(range(self.stages)):
            lo, hi = self.rngs[s]
            sp = hi - lo
            lvls = 15
            h = lo + (qs[s].astype(float) - 0.5) / lvls * sp
            res += self.CB8[idx[s]] * (h[..., None])
        return res.reshape(nv, -1)[:, :self.dim] * self.md + self.m
```

### Step 4: Run the Benchmark & Report Back!
Run this final cell to process the KV Cache.
```python
print("\n=== RUNNING THE-UNTOUCHABLE V12 ON REAL GEMMA KV CACHE ===")

eng_keys = Untouchable_Core(dim=head_dim, stages=8)
print("Fitting Keys...")
eng_keys.fit(X_keys[:5000])
print("Encoding Keys...")
c_k = eng_keys.encode(X_keys)
d_k = eng_keys.decode(c_k)
snr_k = 10 * np.log10(np.mean(X_keys**2) / (np.mean((X_keys - d_k)**2) + 1e-12))
print(f"🎯 KEYS SNR: {snr_k:.2f} dB at 12.0 BPD")

eng_values = Untouchable_Core(dim=head_dim, stages=8)
print("Fitting Values...")
eng_values.fit(X_values[:5000])
print("Encoding Values...")
c_v = eng_values.encode(X_values)
d_v = eng_values.decode(c_v)
snr_v = 10 * np.log10(np.mean(X_values**2) / (np.mean((X_values - d_v)**2) + 1e-12))
print(f"🎯 VALUES SNR: {snr_v:.2f} dB at 12.0 BPD")

print("\n[!] Please copy these metrics back to your local repository chat to bake them into the research paper!")
```
