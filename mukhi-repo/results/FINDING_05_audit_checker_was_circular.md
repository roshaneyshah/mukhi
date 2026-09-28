# Finding 05: the original audit design used a circular checker

Found 2026-09-21 while wiring the improved transliterator into the audit.

## The flaw

The first audit design flagged a corpus pair when two transliteration systems,
the SLPG NMT model and a rule-based map, disagreed about the Shahmukhi side.
Two problems:

1. That measures disagreement between transliterators, not whether the corpus
   pair (Gurmukhi, Shahmukhi) is itself correct.
2. The SLPG NMT model was trained on the SLPG corpus. Using it to validate
   that corpus is circular: it will tend to agree with the corpus's own
   errors.

## The fix

Flag a pair when an INDEPENDENT rule-based transliteration of its Shahmukhi
side disagrees with its own Gurmukhi side, after removing the differences no
rule could resolve (unwritten short vowels and gemination; the ambiguous
vowel and nasal letters). The NMT output is kept as a display column for the
annotator only.

## Calibration on held-out PuMVR sentences

Threshold set on 500 sentences as the 95th percentile of the suspicion score
on known-good pairs (tau = 0.139), then measured on the other 500:

| pair type                            | flagged | AUROC vs correct |
|--------------------------------------|--------:|-----------------:|
| correct                              |    3.0% |                  |
| misaligned (wrong sentence)          |  100.0% |            1.000 |
| 30% of Gurmukhi truncated            |  100.0% |            0.999 |
| two words replaced (~9-word sentence)|   90.0% |            0.989 |
| one word replaced (~9-word sentence) |   53.4% |            0.934 |

Gross alignment errors are always caught. One-word errors are caught about
half the time, which is why the design keeps a uniformly sampled unflagged
stratum: it catches what slips under the threshold, and the stratified
estimator stays unbiased.
