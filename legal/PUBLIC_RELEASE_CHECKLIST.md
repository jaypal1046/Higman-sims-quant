# Public Release Checklist

Use this checklist before changing the repository from private to public.

## Claims and Positioning

- [ ] Remove or rewrite any statement that says the project "beats Google," "beats TurboQuant," "wins," or "state of the art" unless it is still fully supported.
- [ ] Make sure the README, benchmark output, and source-file headers all describe the project as a research prototype.
- [ ] Make sure comparisons to TurboQuant are clearly labeled as either:
  - local scalar baseline comparisons, or
  - official reproductions with matching methodology.
- [ ] Make sure bitrate accounting is explained honestly anywhere benchmark claims appear.

## Documentation

- [ ] Confirm that the README points to the correct entry-point file.
- [ ] Confirm that install instructions work from a clean checkout.
- [ ] Confirm that the README explains what the project does well without overstating benchmark conclusions.
- [ ] Confirm that the README also states current limitations.
- [ ] Confirm that the license, disclaimer, and terms files are present and linked.

## Code and Repository Cleanup

- [ ] Remove dead files, misleading artifacts, or temporary benchmark outputs that should not be public.
- [ ] Decide which version is the main public prototype (`higman_sims_quant.py`, `v6`, `v7`, or another file).
- [ ] Mark older versions clearly as historical or experimental if they remain in the repository.
- [ ] Check for broken imports, path assumptions, and Windows-only or machine-specific behavior.
- [ ] Make sure there are no secrets, local tokens, private notes, or sensitive paths committed.

## Benchmark Hygiene

- [ ] Re-run the main benchmark from a clean environment.
- [ ] Verify that the printed bitrate, overhead, and timing numbers match the actual implementation.
- [ ] Verify that any "expected output" in the README is described as sample output, not guaranteed output.
- [ ] Separate synthetic-benchmark claims from real-data claims.
- [ ] If comparing to Google or another paper, document the comparison method and caveats clearly.

## Public Presentation

- [ ] Choose a short repository description for GitHub.
- [ ] Add topics/tags that describe the repository accurately.
- [ ] Decide whether the repository should be public now or after one more benchmark/documentation pass.
- [ ] Be comfortable with others cloning, benchmarking, criticizing, or forking the code as-is.

## Recommended Minimum Bar for Going Public

If you want a simple release threshold, make the repo public only after all of these are true:

- [ ] README is honest and readable
- [ ] no unsupported "beat Google" claims remain
- [ ] entry point is clear
- [ ] legal docs are in place
- [ ] benchmark output is internally consistent
- [ ] you are comfortable defending the current state publicly
