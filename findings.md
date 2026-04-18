# Colab A100 — Qwen2-1.5B — 15-trial comparison

The configuration space is bimodal: every feasible config lands in either
a "slow" cluster (~1950-2030 ms, ~50 tok/s goodput) or a "fast" cluster
(~430-450 ms, ~215-230 tok/s goodput), with no in-between. Across both
runs the split is explained entirely by one binary knob: `enforce_eager`.
Every slow trial had `enforce_eager=true`; every fast trial had
`enforce_eager=false`. TBA-TPE's explore phase (trials 1-6) happened to
sample `enforce_eager=true` six times in a row, so its best-so-far
latency stayed ~1940 ms until the phase handoff at trial 7 — at which
point every remaining trial (7-15) landed in the fast cluster. Random
search found the fast cluster earlier (trial 4) but kept gambling:
trials 7, 8, 10, 15 were back in the slow cluster, and trial 13 crashed.
Final best-goodput numbers are close (TBA-TPE 230 tok/s vs Random ~224
tok/s), so the real story isn't peak goodput — it's that TBA-TPE is
*consistently* fast after trial 6 while random is still rolling dice at
trial 15.

**Limitations.** These are results from a single model (Qwen2-1.5B), a
single GPU (Colab A100 40GB), and only 15 trials per optimizer.
Feasibility on Colab is noisy: we've seen 9/15-14/15 feasibility on
repeat runs of the same optimizer, and a single-seed comparison can't
tell us whether TBA-TPE's explore phase "unluckily" missed the fast knob
or whether this is typical. With only one dominant knob (`enforce_eager`)
the search problem collapses to "find the binary flip," which isn't
representative of deployments with stronger interactions between
batching, KV cache sizing, and prefill settings. We also have no
statistical significance testing, no wall-clock cost accounting (TBA-TPE
spent 6 trials on slow configs before handoff), and no cross-GPU or
cross-model validation. Treat these plots as an illustrative first data
point, not a claim of general superiority.
