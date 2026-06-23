# Higman-Sims Quantizer V12

V12 is the lower-bitrate side of this research project. It emphasizes sparse refinement and limited metadata so that more of the bit budget can go directly to the signal.

## What V12 Is Good For

- studying aggressive compression behavior
- comparing global and local normalization choices
- testing retrieval-oriented reconstruction ideas

## What The Repo Shows

The local V12 experiments suggest that this design can remain usable in regimes where heavier metadata would be too expensive. The evidence is strongest on reconstruction and retrieval-style checks, not on broad end-to-end LLM benchmarking.

## Important Caveat

V12 should be described as a research prototype. It is interesting because of its trade-off profile, not because it has already been proven superior in every setting.
