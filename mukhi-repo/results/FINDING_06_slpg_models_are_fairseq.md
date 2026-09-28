# Finding 06: the SLPG transliteration models are fairseq checkpoints

Found 2026-09-21 while choosing how to generate the Shahmukhi condition.

## The problem

Both SLPG transliteration models (`SLPG/Punjabi_Gurmukhi_to_Shahmukhi_Transliteration`,
`SLPG/Punjabi_Shahmukhi_to_Gurmukhi_Transliteration`) ship as a fairseq
`checkpoint_13_129000.pt` plus `dict.pa.txt` / `dict.pk.txt`. The first
version of this repo loaded them with `AutoModelForSeq2SeqLM`, which would
have failed on Kaggle. The model card documents no preprocessing, so the
input segmentation has to be inferred from the dictionary, and fairseq itself
often fails to install on Python 3.11.

A core experimental condition should not rest on that.

## The fix

Gurmukhi -> Shahmukhi is the easy direction: Gurmukhi writes every vowel
Shahmukhi needs. A rule system was built, its three ambiguous defaults
grid-searched on the PuMVR dev split, and the held-out split scored once:

| split                 | CER   | 95% CI       | exact match |
|-----------------------|------:|--------------|------------:|
| dev words (545)       | 5.2%  |              |             |
| test words (233)      | 5.6%  | [3.8, 7.6]   |       83.3% |
| 1,000 sentences       | 6.4%  | [6.1, 6.6]   |       16.9% |

Residual errors are almost all lexical Arabic-origin spellings (ਕ as ق in
قالین, final ਾ as ہ in چشمہ, ਜ਼ as ض in قمیض), which no general rule should
chase.

## How the pipeline uses it

- `translit_g2s.gur_to_shah` generates FLORES Shahmukhi by default.
- The fairseq model is supported (`load_slpg_fairseq`) but GATED: the
  `g2s_compare` stage scores it on PuMVR in-session and it replaces the rules
  only if its CER is lower. SLPG report 99.5% word accuracy on FLORES-101, so
  it may well win; the point is to measure rather than assume.
- The paper must say the FLORES Shahmukhi is machine-generated. Generated
  Shahmukhi is more regular than natural writing (no idiosyncratic
  Arabic-origin spellings), which if anything makes the script contrast
  cleaner. The human-written PuMVR sentences are the robustness check.
