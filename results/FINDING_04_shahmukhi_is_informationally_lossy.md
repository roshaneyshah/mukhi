# Finding 04: the Punjabi pair is asymmetric in information, not only in resources

Measured 2026-09-21 on PuMVR (Singh et al., EMNLP 2026 Findings), 1,000
hand-verified parallel items, repository commit 2bc5464. Evaluation only; the
data is not redistributed (no licence file in the source repository).

## Transliteration quality

Rules were developed and the five ambiguous defaults grid-searched (162
configurations) on a DEV split of 545 unique word pairs. The held-out TEST
split of 233 unique word pairs was scored once, after tuning.

| system                     | test words CER | 95% CI       | exact match | sentences CER |
|----------------------------|---------------:|--------------|------------:|--------------:|
| naive character map        |          50.9% | [47.5, 54.1] |        9.0% |         38.4% |
| context-sensitive rules    |          16.0% | [14.1, 18.0] |       43.3% |         15.1% |

Dev CER 16.1% vs test 16.0%: no sign of overfitting the rules.

The grid search moved CER only between 16.1% and 19.5%, so the choice of
defaults for ambiguous letters matters little. What dominates is below.

## What the residual error is

Of the edits remaining on the 1,000 sentences, **38.7% are insertions of
sihari (ਿ), aunkar (ੁ) or addak (ੱ)**: short vowels and gemination that
running Shahmukhi does not write. No rule system can recover them because
they are not in the input. The rest is dominated by genuinely ambiguous
letters: ن as ਨ or ਣ (37 dev errors), و as ੋ/ੂ/ੌ (40), medial ی as ੀ/ੇ/ੈ (46).

## Why it matters for the paper

Serbian Cyrillic and Latin are information-equivalent: a deterministic,
lossless, bijective map. Karne (2026) could claim "meaning held exactly
constant" and it was true at the character level.

Punjabi is not. A Gurmukhi sentence carries vowel and gemination information
its Shahmukhi twin does not. So the pair differs along two axes Serbian holds
fixed:

1. **resource**: how much of each script the model saw in pretraining;
2. **information**: how much of the phonology each script encodes.

Consequences:

- The framing "unbalanced digraphia" should name both axes. It is a stronger
  contrast with Serbian than resource imbalance alone.
- Any residual script gap in the representations is an upper bound on the
  orthography effect proper; part of it can be the model genuinely having less
  to go on. State this rather than let a reviewer find it.
- The transliteration intervention (Shahmukhi -> Gurmukhi at inference) cannot
  restore information the input lacks. Its gain is bounded, and the NMT system
  (with a lexicon) should beat rules on exactly the unwritten-vowel errors.
  Reporting intervention gain for both systems separates "script as surface
  form" from "script as information carrier".
- The paired displacement statistic (FINDING_02) remains valid: content still
  cancels. But the displacement it measures bundles surface form with the
  missing vowels; say so.

## Uses in the pipeline

- `ctx_shah_to_gur` replaces the naive map as system B in the audit.
- The unwritten-character share is itself a descriptive statistic worth one
  sentence and one number in the paper.
