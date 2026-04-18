# Qwen2-1.5B on Colab A100 — benchmark findings

## Single-seed pilot

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

## Multi-seed results (5 seeds × 2 optimizers × 15 trials = 150 trials)

Aggregates across seeds 42, 142, 242, 342, 442 on Colab A100 40GB. All
150 trials were feasible; zero crashes either side. Numbers below come
from `scripts/compute_multiseed_stats.py` (raw floats, no rounding).

| Metric | Random (n=5) | TBA-TPE (n=5) | Mann-Whitney p |
|---|---|---|---|
| Fast-cluster trials / 15 | 7.40 ± 2.51 | 10.60 ± 0.89 | **0.008** |
| Post-hit consistency | 0.539 ± 0.224 | 0.876 ± 0.123 | **0.010** |
| Best latency (ms) | 431.11 ± 1.74 | 431.57 ± 1.90 | 0.841 (tied) |
| First fast-hit trial | 3.00 ± 1.22 | 3.80 ± 2.28 | — |
| Feasibility | 75 / 75 | 75 / 75 | — |

**TBA-TPE's advantage is budget consistency, not peak performance.** Best
latency is statistically indistinguishable between optimizers
(Mann-Whitney two-sided p=0.84) — both end up at the global optimum
around 431 ms on every seed. The significant difference is in *how many*
of the 15 trials land in the fast cluster (7.4 vs 10.6, p=0.008) and in
*how much this varies across seeds*: TBA-TPE's cross-seed standard
deviation is **2.8× tighter** (0.89 vs 2.51 trials). Put differently:
Random hit the fast cluster first slightly more often (mean first-fast
trial 3.0 vs 3.8) because TBA-TPE's explore phase deliberately samples
broadly before committing, but once either optimizer finds the fast
knob, TBA-TPE *stays* there (post-hit consistency 0.88 vs 0.54). The
peak-goodput race is a tie; the budget-waste race is not.

## Limitations

These results are from a single model (Qwen2-1.5B) on a single GPU
(Colab A100 40GB), with 15 trials per run. 5-seed Mann-Whitney is small
but non-parametric, so the significant p-values on consistency are
robust to the distribution shape — not to model/hardware shift. We
haven't run the comparison on other GPUs (T4, L4, H100) or other models
(Llama-3, larger Qwen, mixture-of-experts), so we cannot claim
generality. The search space is dominated by one binary knob
(`enforce_eager`), which collapses the optimization to "find the flip";
deployments with richer interactions between batching, KV-cache sizing,
and prefill strategy may tell a different story. We also have no
wall-clock cost accounting — TBA-TPE spent 6 trials in the slow cluster
during its explore phase, so when trial-cost is high this "better
consistency after first hit" benefit has to be weighed against that
upfront cost. Treat these plots as a grounded first data point, not a
claim of universal superiority.
