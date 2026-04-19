# Qwen2-1.5B on Colab A100 — benchmark findings

This document reports findings under two measurement conditions: the
corrected concurrent load-generator harness (primary result, used in the
paper) and the original sequential harness (retained as a replication
baseline). Both datasets and their per-trial JSONLs are published in
this repository under `results/multiseed_concurrent/` and
`results/multiseed/` respectively.

## Concurrent harness — paper primary result (5 seeds × 2 optimizers × 15 trials = 150 trials)

Requests dispatched through the fixed `LoadGenerator` with
`asyncio.Semaphore`-capped concurrent dispatch (commit d4cbc15
onwards). `batch_wall_ms` in each JSONL confirms the new dispatch
behavior: a slow-cluster trial with 5 requests at ~2500 ms individual
latency shows ~4000 ms batch-wall (requests overlap), not ~12500 ms
(fully serialized).

| Metric | Random (n=5) | TBA-TPE (n=5) | Mann-Whitney p |
|---|---|---|---|
| Fast-cluster trials / 15 | 7.40 ± 2.51 | 10.20 ± 1.10 | **0.014** |
| Post-hit consistency | 0.539 ± 0.224 | 0.876 ± 0.123 | **0.010** |
| Best latency (ms) | 470.52 ± 10.00 | 465.69 ± 2.26 | 0.84 (tied on mean) |
| Feasibility | 75 / 75 | 75 / 75 | — |
| Crashes | 0 | 0 | — |

The consistency advantage (more trials in the fast cluster; more stable
post-hit behavior) replicates and if anything is sharper than under the
sequential harness. **The new finding is on best-latency variance:**
TBA-TPE's best-latency standard deviation is **4.42× tighter** than
Random's (2.26 ms vs 10.00 ms across 5 seeds) even though the means are
statistically indistinguishable (p=0.84). With real concurrent load the
fast cluster has visible per-config variation — contention effects and
scheduling jitter inside vLLM — and TBA-TPE's TPE-exploit phase returns
to near-identical configurations across seeds while Random keeps picking
new points. The consistency advantage is therefore not just "lands in
the fast cluster more often" but also "lands on the same best config
across seeds," which matters more when the measurement itself is noisy.

## Earlier sequential-harness run — retained as replication baseline (5 seeds × 2 optimizers × 15 trials = 150 trials)

Original multi-seed run. Requests dispatched through an inline curl
loop, effectively serial: "5 req/s for 10 s" was 10 sequential requests,
not overlapping load. Retained because the paper's consistency claims
replicate across both measurement conditions; the variance-on-best
finding appears only under concurrent dispatch, and that contrast is the
evidence that concurrent load changes what we can measure, not just how
fast we measure it.

| Metric | Random (n=5) | TBA-TPE (n=5) | Mann-Whitney p |
|---|---|---|---|
| Fast-cluster trials / 15 | 7.40 ± 2.51 | 10.60 ± 0.89 | **0.008** |
| Post-hit consistency | 0.539 ± 0.224 | 0.876 ± 0.123 | **0.010** |
| Best latency (ms) | 431.11 ± 1.74 | 431.57 ± 1.90 | 0.841 (tied) |
| First fast-hit trial | 3.00 ± 1.22 | 3.80 ± 2.28 | — |
| Feasibility | 75 / 75 | 75 / 75 | — |

Under sequential dispatch both optimizers converged to essentially the
same single best-latency point (stds 1.74 ms and 1.90 ms), so the
best-latency variance gap — the headline concurrent finding — is
invisible here. The fast-cluster count difference (7.4 vs 10.6, p=0.008)
and post-hit consistency (0.54 vs 0.88, p=0.010) are as significant as
in the concurrent run, and in the same direction; TBA-TPE's cross-seed
standard deviation on fast-cluster count is **2.8× tighter** (0.89 vs
2.51).

## Single-seed pilot

The original single-seed run that motivated the multi-seed study.
The configuration space is bimodal: every feasible config lands in
either a "slow" cluster (~1950–2030 ms, ~50 tok/s goodput) or a "fast"
cluster (~430–450 ms, ~215–230 tok/s goodput), with no in-between.
Across both runs the split is explained entirely by one binary knob:
`enforce_eager`. Every slow trial had `enforce_eager=true`; every fast
trial had `enforce_eager=false`. TBA-TPE's explore phase (trials 1–6)
happened to sample `enforce_eager=true` six times in a row, so its
best-so-far latency stayed ~1940 ms until the phase handoff at trial 7
— at which point every remaining trial (7–15) landed in the fast
cluster. Random search found the fast cluster earlier (trial 4) but
kept gambling: trials 7, 8, 10, and 15 were back in the slow cluster,
and trial 13 crashed. Final best-goodput numbers are close (TBA-TPE
230 tok/s vs Random ~224 tok/s), so even at n=1 the real story is not
peak goodput but that TBA-TPE is consistently fast after trial 6 while
Random is still rolling dice at trial 15.

## Limitations

Results are from a single model (Qwen2-1.5B) on a single GPU (Colab
A100 40 GB) with 5 seeds of 15 trials each. 5-seed Mann-Whitney is
small but non-parametric, so the significant p-values on consistency
are robust to distribution shape — not to model or hardware shift. We
haven't run the comparison on other GPUs (T4, L4, H100) or other models
(Llama-3, larger Qwen, mixture-of-experts), so we cannot claim
generality. The search space is dominated by one binary knob
(`enforce_eager`), which collapses the optimization to "find the flip";
deployments with richer knob interactions between batching, KV-cache
sizing, and prefill strategy may tell a different story. We have no
wall-clock cost accounting — TBA-TPE spent 6 trials in the slow cluster
during its explore phase, so when trial cost is high this "better
consistency after first hit" benefit has to be weighed against that
upfront cost. Treat these plots as a grounded first data point, not a
claim of universal superiority.
