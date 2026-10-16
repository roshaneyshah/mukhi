# Finding 09: a Unicode normalisation difference between conditions

## What the UNK rate turned out to be

XLM-R maps eight Gurmukhi codepoints to UNK. Five occur in FLORES:

| codepoint | letter | occurrences | share of Gurmukhi characters |
|---|---|---:|---:|
| U+0A36 | SHA | 1,444 | 0.556% |
| U+0A5B | ZA | 535 | 0.206% |
| U+0A5E | FA | 232 | 0.089% |
| U+0A59 | KHHA | 58 | 0.022% |
| U+0A5A | GHHA | 4 | 0.002% |

Total 2,273 characters, 0.875% of the Gurmukhi text, touching 1,153 of 2,009
sentences (57.4%). The other three, LLA, YAKASH and the abbreviation sign, do
not appear.

These are not arbitrary. All five are nukta letters, the characters Punjabi
uses to write Perso-Arabic phonemes in loanwords. The vocabulary gap is
concentrated on exactly the lexical stratum Punjabi shares with Shahmukhi and
Urdu.

## Why they are UNK, which is not a vocabulary gap at all

Gurmukhi's nukta letters are on the Unicode composition exclusion list.
NFC therefore *decomposes* U+0A36 into U+0A38 + U+0A3C rather than leaving
it precomposed, and XLM-R's vocabulary covers the decomposed form. The
precomposed codepoints are UNK. The same letters written as base + nukta are
not. FLORES ships the precomposed form.

So the UNK rate is a normalisation artifact, and one line fixes it.

## The defect that matters more

`devowel_gurmukhi` called `unicodedata.normalize("NFC", ...)` on its input.
Nothing else did. The conditions therefore split into two groups:

| condition | NFC | precomposed nukta letters |
|---|---|---:|
| pan_Guru | no | 2,273 |
| hin_Deva, urd_Arab | no | n/a |
| pan_Guru_devowel | yes | 0 |
| pan_Shah, srp_Latn, every romanised condition | yes | 0 |

The de-voweled control was meant to differ from Gurmukhi in exactly three
marks. It also differed in Unicode form, and that difference removed 2,273
UNK tokens from one side of the contrast. Any contrast pairing pan_Guru with
pan_Guru_devowel, and the script-form ratio built from it, is confounded.

## What is and is not affected

Not affected: Serbian Cyrillic against Latin (both NFC, no nukta), Hindi
against Urdu (both raw), and the romanised contrasts. The headline Punjabi
against Serbian comparison does not involve the de-voweled condition.

Affected: Gurmukhi against de-voweled Gurmukhi, de-voweled Gurmukhi against
Shahmukhi, the script-form ratio, and the de-voweled ceiling in the SIB
intervention.

Separately, every Punjabi contrast carried a one-sided UNK burden on the
Gurmukhi side. The draft described this as biasing against our own effect.
That was a guess and it was probably backwards: UNK collapses five distinct
letters onto one token in 57% of sentences, which plausibly makes the
Gurmukhi representation more uniform and the Gurmukhi-Shahmukhi difference
more consistent, inflating the statistic rather than suppressing it.

## Fix, and the part of it I got wrong first

`build_conditions` now NFC-normalises every FLORES condition at the source,
before any derived condition is built, so no two conditions can differ by
normalisation alone. This also removes the UNK problem: after normalisation
no condition contains a precomposed nukta letter.

That fix was incomplete. FLORES is one of three entry points that build
conditions, and the other two were missed:

- the SIB-200 path in `run_all.py`, which builds the intervention arm. Its
  `pan_Guru` split still carried 1,194 precomposed nukta characters while its
  de-voweled control carried none, so the intervention's de-voweled ceiling
  had the same confound the FLORES contrasts did.
- the PuMVR path, the human-text robustness set.

Both now normalise at the source. `tests/test_normalisation.py` checks the
decomposition behaviour, that de-voweling changes only the three marks, and
that each of the three entry points normalises before deriving anything, so a
fourth entry point cannot repeat this quietly.

Recomputed: the two FLORES contrasts that involve `pan_Guru`. Still to
recompute when the SIB and PuMVR conditions are rebuilt: the intervention
arm (Section 5.5) and the tokeniser premium, whose current numbers come from
a pre-normalisation run. The Serbian comparison and the scale result are
unaffected throughout.
