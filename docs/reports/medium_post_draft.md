# I Built An E8-Lattice KV-Cache Quantization Prototype. Here Is What Looks Promising And What Is Still Unproven.

Large language models keep getting longer context windows, but the KV cache still scales linearly with sequence length. That makes memory one of the first real bottlenecks you hit when you try to push context length without moving to much larger hardware.

Over the last few months I have been building a research prototype for KV-cache quantization around a different idea than the usual scalar or low-bit packing approach. Instead of treating each value independently, I quantize small 8-dimensional chunks with E8 lattice structure, then add staged normalization and residual refinement on top.

I want to share the project in a careful way, because I do not think it is honest to call it "the best" solution yet. What I can say is that the prototype is now working end to end, and the early evidence is good enough that I think the approach deserves attention.

GitHub repo:
[https://github.com/jaypal1046/Higman-sims-quant](https://github.com/jaypal1046/Higman-sims-quant)

![Local rate-distortion curve from the repository experiments. Upload `assets/figure_pareto_frontier.png` manually when publishing to Medium.](../../assets/figure_pareto_frontier.png)

*Suggested caption: A local Pareto-style view of bitrate versus distortion from the repository experiments. I treat this as encouraging evidence, not a universal benchmark.*

Local file in this repo:
`../../assets/figure_pareto_frontier.png`

## What I Built

The repository explores several related quantizers rather than one single final design.

- `V12` focuses on low-bitrate behavior and tries to spend as little as possible on metadata.
- `V16` explores stronger local normalization so the lattice sees a cleaner local distribution.
- `V19` pushes toward very low bitrate again, but with a denser base layer plus sparse refinement.

The common thread is simple:

1. Split activations into 8D chunks.
2. Project them onto an E8 lattice.
3. Normalize either globally or locally depending on the regime.
4. Use recursive residual refinement to recover more detail when the bitrate budget allows it.

That framing turned out to be useful, because low-bitrate and high-fidelity settings do not behave like the same problem. A design that looks strong at 8 BPD can fall apart at 2 BPD, and a design that survives at 2 BPD may leave quality on the table at higher bitrates.

One small piece of the implementation looks like this:

```python
def quantize_tensor(tensor, engine):
    flat = tensor.detach().cpu().numpy().reshape(-1, tensor.shape[-1]).astype(np.float64)
    codes = engine.encode(flat)
    decoded = engine.decode(codes)
    return torch.from_numpy(decoded).reshape(tensor.shape).to(
        device=tensor.device,
        dtype=tensor.dtype,
    )
```

That snippet is not the whole method, but it captures the basic pattern: flatten a cache tensor, run the lattice codec, decode it, and put it back into the same shape and dtype so it can continue through the model.

## What Is Actually Working Today

The first thing I needed to know was whether the math worked at all in code. At this point, the answer is yes.

- The core quantizers run locally and reproduce the reconstruction numbers in the repo.
- The PyTorch E8 kernel matches the NumPy reference implementation in parity checks.
- The project now has a real end-to-end autoregressive evaluation path where quantized `past_key_values` are fed back into the next decoding step.

That last point matters the most. Before that, I had good reconstruction metrics and good cache-level proxies, but not a true end-to-end path.

![Residual energy decays across quantization stages. Upload `assets/figure_residual_decay.png` manually when publishing to Medium.](../../assets/figure_residual_decay.png)

*Suggested caption: Residual energy drops across quantization stages in the local experiments, which is one reason staged refinement is worth exploring here.*

Local file in this repo:
`../../assets/figure_residual_decay.png`

## The Most Important Upgrade: End-To-End Cache Evaluation

I recently replaced the old "projected PPL" style check with a real teacher-forced autoregressive evaluation loop. The model runs one token at a time, the cache is quantized after each step, and the quantized cache is used in the next forward pass.

This is the core idea in code:

```python
for position in range(1, input_ids.shape[1]):
    outputs = model(
        input_ids=current_input,
        past_key_values=past,
        use_cache=True,
    )

    logits = outputs.logits[:, -1, :]
    past = outputs.past_key_values

    if quantizer is not None:
        past = quantizer.quantize_cache(past)

    current_input = input_ids[:, position : position + 1]
```

That is still not the final benchmark I want, but it is much more meaningful than inferring model quality from SNR alone.

On short GPT-2 runs, the results were encouraging:

- With `V19` on a 96-token run, baseline streaming perplexity was `50.188924` and quantized-cache perplexity was `50.184533`, with `98.95%` top-1 agreement.
- With `V19` on a 64-token run, baseline perplexity was `80.140889` and quantized-cache perplexity was `80.011448`, with `100%` top-1 agreement.
- With `V16` on a 64-token run, baseline perplexity was `80.140889` and quantized-cache perplexity was `80.140813`, also with `100%` top-1 agreement.

I do not interpret tiny improvements on short runs as "quantization helps perplexity." That would be over-reading noise. What I do take from it is that the cache-quantized decoding loop is now functioning without obvious damage in these short tests.

## Why I Think The Direction Is Interesting

What keeps me interested is not one flashy number. It is the shape of the trade-offs.

The project suggests that:

- metadata cost matters a lot more than it first appears
- local normalization can be very useful once bitrate is high enough
- the low-bitrate and high-fidelity regimes probably need different designs
- reconstruction-focused lattice methods can transfer into actual autoregressive cache use without immediately breaking the model

A simplified version of the layerwise cache fitting path looks like this:

```python
for layer_idx in range(get_layer_count(cache)):
    keys, values = get_layer_kv(cache, layer_idx)
    head_dim = keys.shape[-1]

    key_engine = build_engine(engine_name, head_dim, target_bpd, max_stages)
    value_engine = build_engine(engine_name, head_dim, target_bpd, max_stages)

    key_engine.fit(keys.detach().cpu().numpy().reshape(-1, head_dim).astype(np.float64))
    value_engine.fit(values.detach().cpu().numpy().reshape(-1, head_dim).astype(np.float64))
```

That is one of the reasons I see this as more than a pure toy. The logic is no longer just reconstructing vectors in isolation. It is starting to behave like an actual cache-aware evaluation setup.

## What This Does Not Prove Yet

This is the part I want to be very clear about.

The current results do **not** prove that this is the best KV-cache quantization method.

There is still a lot left to do:

- evaluate across more models
- test much longer contexts
- compare against stronger baselines under one consistent protocol
- measure real serialized storage, not only entropy-style estimates
- measure actual runtime overhead with Triton or another fused kernel path
- test on standard corpora instead of only hand-picked short runs

In other words, I think the project has moved from "interesting idea" to "credible research prototype," but not yet to "settled result."

## The Honest Claim I Am Comfortable Making

I am comfortable saying the following:

This E8-lattice KV-cache quantization prototype works, the end-to-end path now exists, and the early evidence suggests the approach is worth broader benchmarking.

I am **not** comfortable saying:

- it is already best in class
- it is production-ready
- it has been proven across long-context inference

That distinction matters. I would rather be slightly conservative and let stronger benchmarks earn the bigger claim later.

## What Comes Next

My next goal is straightforward:

1. run the end-to-end evaluator on more models
2. move to longer sequences
3. compare against stronger baselines
4. tighten the runtime path so the memory savings can be evaluated alongside latency

If those hold up, then the story gets much stronger. If they do not, that is useful too, because it will tell me which parts of the design are real and which parts only looked good in reconstruction space.

For now, I think this is a promising line of work, and I am happy that it has crossed the point where it can be tested honestly end to end.

If you want to look through the code, the repository is here:
[https://github.com/jaypal1046/Higman-sims-quant](https://github.com/jaypal1046/Higman-sims-quant)

---

## Publishing Note

If you post this on Medium, upload the images manually from these exact files:

- relative to this draft:
  `../../assets/figure_pareto_frontier.png`
- relative to this draft:
  `../../assets/figure_residual_decay.png`
- absolute path:
  `C:\Jay\_Plugin\Higman sims quant\assets\figure_pareto_frontier.png`
- absolute path:
  `C:\Jay\_Plugin\Higman sims quant\assets\figure_residual_decay.png`

The relative links in this draft are useful for local preview, but Medium will need the image files uploaded directly.
