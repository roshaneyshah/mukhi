# Finding 02: a falling script-probe curve does not mean script was abstracted away

Measured in-container, 2026-09-20, on synthetic activations with a planted,
known structure. No model weights required.

## The problem

The proposal's central question is "at what depth does representation stop
tracking orthography and start tracking language?", answered by a layerwise
script-probe curve. That curve is not scale-invariant, and the confound runs
in exactly the direction that would produce a false positive.

Construction: script encoded as a fixed displacement along a fixed unit
direction, weight 6.0, **identical in every condition**. The only thing varied
is the weight on a shared semantic component, which grows with depth in real
transformers (activation norms grow with depth).

| content weight | mean act. norm | probe acc | selectivity | Cohen's d | paired consistency |
|---------------:|---------------:|----------:|------------:|----------:|-------------------:|
| 0.0            |           10.0 |     1.000 |       0.470 |    12.072 |              0.729 |
| 1.5            |           15.6 |     0.998 |       0.418 |     6.235 |              0.734 |
| 3.0            |           25.8 |     0.948 |       0.353 |     3.610 |              0.729 |
| 4.5            |           37.1 |     0.865 |       0.247 |     2.513 |              0.729 |
| 6.0            |           48.7 |     0.775 |       0.187 |     1.891 |              0.735 |

Probe accuracy falls 1.000 -> 0.775 and selectivity 0.470 -> 0.187 while the
script signal is **bit-for-bit unchanged**. Cohen's d along the class-mean
direction is no better: it falls by 10.2, because the pooled within-class
spread absorbs the growing content variance.

Read naively, that decaying curve says "orthography is abstracted away with
depth". Nothing was abstracted away.

## The fix, and why it is specific to this design

Because the corpus is a true minimal pair -- the same sentence in two scripts
-- the shared semantic component **cancels in the per-item difference**:

    delta_i = h_shahmukhi(i) - h_gurmukhi(i)

`paired_script_displacement` reports

    consistency = ||mean(delta)|| / sqrt(mean ||delta_i||^2)     in [0, 1]

which is flat to within 0.006 across the whole sweep above.

This is the argument for the Punjabi design that survives Karne (2026) and
Verma et al. (2026) both having published on the general question. It is not
"nobody used this pair". It is: **a minimal pair does not merely control the
confound, it licenses a measurement that unpaired and synthetically-romanized
designs cannot make.** Verma et al. compare neuron SETS across native and
romanized inputs via Jaccard overlap, which is unpaired at the item level and
therefore cannot form delta_i at all.

## Consequences adopted in the pipeline

- `paired_script_displacement` runs at every layer beside the probe.
- Layerwise mean activation norm is logged so any accuracy decay can be
  checked against norm growth.
- Probe accuracy and selectivity are still reported, for comparability with
  the probing literature, but the depth claim rests on the paired statistic.
- Per-item L2 normalisation is the default before computing the paired
  statistic, removing layerwise norm growth directly.

## Caveat on the statistic

`consistency` is bounded above by noise: two independent noise draws alone
give a nonzero value. Report it against a permutation baseline in which delta
is formed from MISMATCHED sentence pairs, the same way CKA is reported against
its row-permutation baseline.
