# Finding 01: the standard CKA recipe is unusable at the planned sample size

Measured in-container, 2026-09-20. No model weights needed - this is a property
of the estimator, not of any model.

## What was measured

Biased linear CKA (Kornblith et al. 2019) applied to two INDEPENDENT Gaussian
matrices, i.e. data whose true representational similarity is exactly zero, at
the hidden dimensions and sample sizes in the proposal.

| model        |    d | n=500 | n=1000 | n=2000 |
|--------------|-----:|------:|-------:|-------:|
| XLM-R base   |  768 | 0.606 |  0.435 |  0.277 |
| MuRIL        |  768 | 0.606 |  0.434 |  0.278 |
| XLM-R large  | 1024 | 0.672 |  0.506 |  0.338 |
| Qwen2.5-1.5B | 1536 | 0.754 |  0.605 |  0.434 |
| Qwen2.5-3B   | 2048 | 0.804 |  0.672 |  0.506 |
| Gemma-2-2B   | 2304 | 0.822 |  0.698 |  0.536 |
| Aya-8B       | 4096 | 0.891 |  0.804 |  0.672 |

The unbiased HSIC-based estimator returns |CKA| < 0.003 on the same data in
every cell.

## Why it matters for this project

1. At the proposal's planned n (1000-2000 sentences), a reported CKA of 0.8 in
   a large decoder is **indistinguishable from noise**. The headline depth
   curve would be measuring finite-sample bias.

2. The bias **scales with hidden dimension**. The proposal singles out
   "MuRIL against XLM-R" as worth its own line because one saw transliterated
   pretraining and the other did not. Those two share d=768, so that specific
   comparison survives - but any comparison across models of different width
   (XLM-R base vs large, Qwen 1.5B vs 3B, anything vs Aya) would be partly
   reading hidden width rather than representational similarity.

3. A layerwise curve is not immune. Layers differ in effective rank, so the
   bias varies along the curve and can manufacture or flatten a depth trend.

## Consequences adopted in the pipeline

- `unbiased_linear_cka` is the default estimator; the biased one is retained
  only for comparability with prior work that used it.
- Every reported CKA ships with a row-permutation baseline over the same
  matrices, and the headline quantity is `delta = aligned - baseline_mean`.
- Target n raised: use >= 2000 aligned pairs. The cost is forward passes only.
- Cross-model comparisons are reported on `delta`, never on raw CKA.

## Relevance to prior work

Moosa et al. (2023) measured cross-lingual representation similarity with CKA
on parallel FLORES sentences. Any replication or contrast against that line
should state which estimator and which n, since the two choices can move the
number by more than the effect being claimed.
