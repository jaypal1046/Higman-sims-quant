| Method | Eff. bpd | K SNR (dB) | K Cosine | V SNR (dB) | V Cosine | Max |err| | Encode ms | Decode ms | PPL | Delta PPL % | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP cache | nan | inf | 1.00000 | inf | 1.00000 | 0.0000e+00 | 0.00 | 0.00 | 1.1531 | +0.000 | Uncompressed reference |
| Scalar 3-bit | 3.000 | 24.47 | 0.98717 | 12.51 | 0.96870 | 3.2121e+01 | 86.53 | 55.91 | 3.1409 | +172.399 | Per-dimension uniform 3-bit |
| V9 exact 3.0 bpd | 3.000 | 11.17 | 0.95907 | 10.20 | 0.95786 | 1.2707e+02 | 6742.39 | 570.05 | 17.1536 | +1387.660 | Recursive E8 at searched exact-rate config |
