# MUKHI

Script or language? A controlled probe of multilingual representation using
Punjabi's two writing systems.

Punjabi is written in Gurmukhi (India) and Shahmukhi, a Perso-Arabic script
(Pakistan). Same language, two scripts: a minimal pair that holds language
constant while orthography varies.

## Status

The pipeline is complete and tested: 613 assertions across 20 test suites,
including planted-structure tests the analysis must recover, matched nulls it
must not report, and a full run of every stage on real Punjabi text through
the real transformers code path.

Seven models have trained runs and matched random-weight baselines, each
averaged over three random-init seeds. Four encoders (XLM-R base, large and
XL, and MuRIL) carry the results. Three decoders (Qwen2.5-1.5B, Qwen2.5-3B,
Gemma-2-2B) were first pooled at the last token, which made the paired
statistic measure two punctuation marks (Finding 8). Rerun with mean pooling
(paper Section 5.8), none shows the encoder pattern: both Qwen models
separate Serbian more than Punjabi at every layer, and Gemma is mixed.

The 65 to 90% script-form range rests on the share falling as information is
removed. Two intermediate controls test this (`tools_bracket_check.py`). The
share falls at every step in all four encoders, and the two ends keep their
order for every random-init seed.

Script resource is measured for XLM-R, whose training corpus has a public recreation (CC-100) (`tools_resource_count.py`, `tools_resource_summary.py`, paper
Section 3.6). Shahmukhi Punjabi is 0.8 to 1.3% of Gurmukhi Punjabi. Serbian
Latin is about 5.5% of Serbian Cyrillic, but the same Latin orthography is
four times as frequent as Serbian Cyrillic in the Croatian and Bosnian files.

Both transliteration systems are scored against 1,000 human-written parallel
items, and 300 generated FLORES sentences have been judged by hand: 7.4%
carry at least one error, 95% CI [4.9%, 10.9%].

Headline numbers are in `results/REPORT.md`, the figure in
`results/figures/headline_depth.png`, and the write-up in `paper/draft.md`.

## What it finds so far

Paired script displacement, trained model minus the same architecture with
random weights (FLORES+, n = 2,009), each averaged over three random-init
seeds:

| model | contrast | embedding | mid-stack minimum | last layer |
|---|---|---:|---:|---:|
| XLM-R base | Punjabi Gurmukhi vs Shahmukhi (unbalanced) | 0.537 | 0.323 (L4) | 0.443 |
| XLM-R base | Serbian Cyrillic vs Latin (balanced) | 0.612 | 0.105 (L5) | 0.539 |
| XLM-R base | romanised Shahmukhi vs romanised Urdu (no script difference) | 0.010 | -0.009 (L2) | 0.106 |
| XLM-R large | Punjabi Gurmukhi vs Shahmukhi | 0.543 | 0.341 (L15) | 0.477 |
| XLM-R large | Serbian Cyrillic vs Latin | 0.651 | 0.032 (L11) | 0.578 |
| XLM-R XL | Punjabi Gurmukhi vs Shahmukhi | 0.527 | 0.346 (L20) | 0.508 |
| XLM-R XL | Serbian Cyrillic vs Latin | 0.602 | -0.035 (L18) | 0.656 |

Across the three XLM-R sizes the Punjabi trough rises slightly (0.323, 0.341,
0.346) and the Serbian trough falls below its random baseline (0.105, 0.032,
-0.035). Each step is tested on paired bootstrap draws and every interval
excludes zero (paper Section 5.7).

Balanced digraphia is largely abstracted away in the middle of the stack.
Unbalanced digraphia is not. Two different languages in one script are
separated far less than one language in two scripts.

Information matching says most of the gap is script form. Four Gurmukhi
controls remove information in steps, and the share of the gap left falls at
every step. Between the two ends it is 65 to 90% across the four encoders,
measured on the same trained-minus-random basis as everything else.

Transliterating Shahmukhi back to Gurmukhi closes 45% of the topic-transfer
gap on SIB-200 in XLM-R (0.275 to 0.485 against 0.745) and 43% in MuRIL
(0.475 to 0.603 against 0.775). In XLM-R the remainder stays significantly
below the matched ceiling of 0.608, which points to transliteration quality
ahead of missing information without isolating it. In MuRIL the remainder is
not significant.

Caveats the paper states and this summary should too: the balanced-versus-
unbalanced comparison rests on XLM-R base, since MuRIL sends 13.4% of Serbian
Cyrillic to UNK. Probes are at ceiling at every layer and carry no depth
claim. And the trained-minus-random difference is reported as a point
estimate beside the seed range, with no significance test of its own.

## The contribution, after the literature check

The Gurmukhi/Shahmukhi pair has not been used for representation analysis.
The *design* has, on other pairs, in 2026:

- Karne, *One Language, Two Scripts* (arXiv 2603.08869): Serbian Cyrillic and
  Latin, Gemma SAE features. Finds script invariance that grows with scale.
- Verma, Chatterjee, Gupta and Chakraborty, *Multilingual Language Models
  Encode Script Over Linguistic Structure* (ACL 2026): synthetic romanisation.
  finds near-disjoint script-bound units. The opposite conclusion.
- Singh et al., PuMVR (EMNLP 2026 Findings): a three-script parallel Punjabi
  benchmark for vision-language models. Its evidence is behavioural.

So the paper cannot claim the idea of a minimal pair. It claims three things
the others do not have:

1. **Unbalanced digraphia.** Serbian's two scripts are balanced: both in
   everyday use and informationally equivalent (a lossless bijection).
   Punjabi's differ in information, and in a public recreation of XLM-R's training data (CC-100) Shahmukhi
   Punjabi is about 1% as frequent as Gurmukhi (paper Section 3.6). About 39% of the residual error in
   transliterating Shahmukhi to Gurmukhi is short vowels and gemination that
   Shahmukhi simply does not write (FINDING_04). Running the same pipeline on
   Serbian and Punjabi tests whether script invariance survives asymmetry, and
   speaks to why the two 2026 papers disagree.
2. **A paired measurement.** In a minimal pair the shared content cancels in
   the per-item difference `h_shah(i) - h_guru(i)`. That supports a statistic
   immune to the activation-norm growth that makes probe-accuracy depth curves
   misleading (FINDING_02). Unpaired and neuron-set designs cannot form it.
3. **An information / script-shape decomposition.** Removing from Gurmukhi the
   characters Shahmukhi leaves unwritten gives an information-matched
   Gurmukhi. The Gurmukhi/Shahmukhi gap then splits into an information part
   and a script-shape part, each measured directly.

## Nine findings from building it

Full write-ups in `results/FINDING_*.md`.

| # | finding | consequence |
|---|---|---|
| 01 | Biased linear CKA reads 0.44 (XLM-R base) to 0.80 (Aya-8B) on pure noise at n=1000, and the bias grows with hidden width | unbiased estimator, permutation baselines, n about 2000, compare models on deltas |
| 02 | With the script signal held bit-for-bit constant, growing content variance drops probe accuracy 1.00 to 0.78 | the depth claim rests on the paired statistic alone |
| 03 | The row-permutation null used for CKA is invalid for the paired statistic (a constant offset survives it) | sign-flip null, checked against the analytic 1/sqrt(n) floor |
| 04 | Shahmukhi omits information: 39% of the residual S->G error is unwritten vowels and gemination | information-matched control and unbalanced-digraphia framing |
| 05 | The first audit design validated the SLPG corpus with a model trained on it | independent rule-based checker, calibrated on held-out PuMVR |
| 06 | SLPG transliteration models are fairseq checkpoints with undocumented preprocessing | measured rule-based G->S generator and fairseq used only if it wins in-session |
| 07 | A randomly initialised model already shows a large, significant script displacement | every model gets a random-init baseline and report trained minus random |
| 08 | Under last-token pooling the paired statistic reduces to one punctuation embedding and trained and random agree to three decimals | last-token decoders excluded, Qwen2.5-1.5B rerun with mean pooling (paper 5.8) |
| 09 | Only generated conditions were NFC-normalised, so the de-voweled control differed from Gurmukhi in Unicode form and XLM-R's Gurmukhi UNK rate was precomposed nukta letters. XLM-R covers the decomposed form. The first fix reached FLORES only, leaving the SIB and PuMVR paths carrying it | normalise at the source in all three entry points and affected contrasts, tokeniser premium and intervention recomputed |

## Transliteration, measured on human text

Rules tuned on a dev split of PuMVR word pairs. The held-out split scored once.

| system | held-out word CER | 95% CI | sentence CER |
|---|---:|---|---:|
| Shahmukhi to Gurmukhi, naive character map | 50.9% | [47.5, 54.1] | 38.4% |
| Shahmukhi to Gurmukhi, context rules | 16.0% | [14.1, 18.0] | 15.1% |
| Gurmukhi to Shahmukhi, rules | 5.6% | [3.8, 7.6] | 6.4% |

The audit flag (independent round-trip check) was calibrated on half of PuMVR
and tested on the other half: it flags 3% of correct pairs, 100% of misaligned
pairs, 90% of pairs with two words replaced and 53% with one word replaced.

## Design

Every condition is aligned item by item to the same FLORES+ sentences
(n = 2,009), so any two form a contrast.

| contrast | what varies | held constant |
|---|---|---|
| Gurmukhi vs Shahmukhi | script and information | language, content |
| Gurmukhi vs de-voweled Gurmukhi | information only | language, content, script |
| de-voweled Gurmukhi vs Shahmukhi | script shape (information about matched) | language, content |
| Shahmukhi vs Urdu | language | script, content |
| Hindi vs Urdu | script, with a register confound | content |
| romanised Gurmukhi vs romanised Shahmukhi | information only, shared Latin script | language, content |
| romanised Shahmukhi vs romanised Urdu | language, script removed | content |
| Serbian Cyrillic vs Serbian Latin | script, balanced and lossless | language, content |

FLORES+ has no Shahmukhi and no Serbian Latin. Serbian Latin is derived by the
standard lossless mapping. Shahmukhi is generated by the measured G->S rules
(or the SLPG model if it wins in-session), which the paper must state. The
1,000 human-written PuMVR sentences are the robustness set.

Per contrast and layer: unbiased CKA with a 200-permutation baseline, a
linear probe with a Hewitt and Liang control task keyed to sentence id, and
paired displacement with a sign-flip null. Benjamini-Hochberg is applied over
every layer, contrast and statistic of a model at once.

The intervention (proposal step 5) has two arms. Controlled: SIB-200 topic
classification on frozen features, trained on Gurmukhi and tested on
Gurmukhi, Shahmukhi, Shahmukhi transliterated back, and de-voweled Gurmukhi.
Natural: PuMVR option scoring by decoder likelihood on human-written text.

## Running it

`notebooks/mukhi_selfcontained.ipynb` is a single Kaggle notebook with the
code embedded, so it needs no dataset upload. Set the accelerator to GPU and
Internet on, then Save Version and Run All. It skips models whose results are
already present.

```bash
python run_all.py --stage smoke          # planted-structure check, no network
python run_all.py --stage translit_eval  # PuMVR via GitHub
python run_all.py --stage conditions     # FLORES+ and SIB-200 (HF token)
python run_all.py --stage fertility      # tokenisers only
python run_all.py --stage audit          # then fill results/audit_sheet.csv
python run_all.py --stage score_audit
python run_all.py --stage main --models xlm-roberta-base          # GPU
python run_all.py --stage main --models xlm-roberta-base --random-init --seed 0
python run_all.py --stage intervention
python run_all.py --stage intervention_mcq
python run_all.py --stage report         # results/REPORT.md
```

Useful flags: `--data pumvr` (human-text robustness set), `--contrasts` (a
subset, as `a|b,c|d`), `--n-perm`, `--purge-cache`, `--cache-dir`,
`--n-sentences` for quick checks.

Cost, measured on a Kaggle T4: about 12 hours for four models at 2,009
sentences, trained and random-weight, over five contrasts. Extraction is
minutes per condition. The permutation analysis on CPU dominates.

## Tests

```bash
for t in tests/test_*.py; do python3 $t | tail -1; done
```

`tests/test_style.py` enforces the no-em-dash rule across the repository.
`tests/test_notebook.py` parses every notebook command with the real argument
parser.

## Layout

```
src/
  cka.py            biased and unbiased CKA, O(n^2) permutation baselines
  probes.py         probes, control tasks, selectivity, paired displacement
  analysis.py       layerwise orchestration, contrasts, BH, decomposition
  extract.py        extraction with masked pooling, random-init baseline
  conditions.py     FLORES+ conditions, Serbian mapping, de-voweling, uroman
  translit_ctx.py   Shahmukhi -> Gurmukhi context rules
  translit_g2s.py   Gurmukhi -> Shahmukhi rules
  translit_eval.py  CER, edit ops, audit suspicion score, AUROC
  translit.py       gated SLPG fairseq loader, naive baseline map
  intervention.py   both intervention arms, McNemar, paired bootstrap
  fertility.py      tokeniser premium, byte fallback, paired Wilcoxon
  data.py           audit sampling and stratified error estimate
  pumvr.py          pinned-commit PuMVR fetch
  plots.py, synthetic.py
run_all.py          every stage
notebooks/          kaggle_run.ipynb
paper/              draft.md
results/            findings, measured JSON, REPORT.md
tests/              20 suites
```

## What only you can do

1. **Judge the audit sheet.** 300 rows of generated Shahmukhi. The estimate
   reweights the flagged and random strata into an honest corpus error rate.
2. **Spot-check the rules on words you know.** They reach 16% and 6% CER on
   PuMVR, but PuMVR is everyday vocabulary. FLORES is news and Wikipedia, with
   more names and Arabic-origin words, where the G->S rules are weakest.
3. **Accept three Hugging Face licences** before the Kaggle run: FLORES+,
   Gemma-2 and Aya Expanse.

## Data and licences

- FLORES+ (`openlanguagedata/flores_plus`): CC BY-SA 4.0, gated.
- SIB-200 (`Davlan/sib200`): CC BY-SA 4.0.
- PuMVR (github.com/prabhjotschugh/Not-Truly-Multilingual-PuMVR): no licence
  file. Fetched at commit 2bc5464 for evaluation, never redistributed, and
  excluded by `.gitignore`. Ask the authors before publishing anything derived
  from it beyond aggregate scores.
- SLPG transliteration models: fairseq checkpoints. See FINDING_06.

## Outstanding

1. Figures for the paper beyond the depth curve in Figure 1.
2. The ACL template version and an anonymised repository link.

Done, previously listed here: the paired sentence-level bootstrap with
magnitude statistics, a round-trip information control and three random-init
seeds for every encoder (Section 4.6, run with `--stage bootstrap` and then
`tools_bootstrap_crossmodel.py`). Also done: the bracket test with two
intermediate controls and a mean-pooled decoder (`--boot-only`, `--pooling
mean`, then `tools_bracket_check.py`). Their draws are in `results/bracket/`
and `results/decoder/`. Round 3 (`results/round3/`) added every remaining
contrast, per-seed draws, PuMVR with random baselines, generated against human
Shahmukhi, two more mean-pooled decoders, the 7B behavioural arm and the
resource counts. Round 4 (`results/round4/`) added XLM-R XL and the Croatian
and Bosnian counts. `tools_plot_scale.py` draws Figure 1.
