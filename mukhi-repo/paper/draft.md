# One Language, Two Unequal Scripts: Separating Orthography, Information and Resource in Multilingual Representations with Punjabi

Roshane Shahbaz

*Draft, September 2026. Results are from two encoders, XLM-R base and MuRIL.
The scale section marked [SCALE] awaits four larger models. Items marked
[PENDING] await a human judgement.*

---

## Abstract

Multilingual language models are usually said to abstract away surface form
in their middle layers. Tests of that claim compare languages that differ in
script and language at once. Digraphic languages separate the two, and
recent work on Serbian finds script-invariant representations while work on
romanised text finds script-bound ones. We argue the disagreement is partly
a property of the test pair. Serbian Cyrillic and Latin are balanced: used
with near-equal frequency (Karne, 2026) and related by a lossless mapping.
Punjabi Gurmukhi and Shahmukhi are not. Shahmukhi is less represented in
pretraining and also omits information: on 1,000 human-written parallel
items, 39% of the residual error in transliterating Shahmukhi to Gurmukhi
consists of short vowels and gemination absent from the Shahmukhi input. We
exploit the minimal pair with a paired displacement statistic in which
shared content cancels, which we show is immune to the activation-norm
growth that makes probe-accuracy depth curves misleading, and with an
information-matched Gurmukhi control that splits the script gap into an
information part and a script-shape part. Across two multilingual encoders,
the Punjabi script gap stands above matched random-init baselines at every
depth, narrowing to a mid-stack minimum of 0.307 above random in XLM-R base
before rising again, and 85 to 93% of it survives information matching,
making it script form rather than the vowels Shahmukhi omits. In XLM-R base
the same pipeline puts Serbian's balanced pair at a minimum of 0.105, a
third of Punjabi's, which is what would follow if the asymmetry of the pair,
and not digraphia alone, governed whether script is abstracted away; the
second encoder cannot test this, since it sends 13.4% of Serbian Cyrillic to
UNK. A transliteration intervention on topic classification closes 42% of
the Gurmukhi-Shahmukhi transfer gap, and the remainder stays significantly
below the information-matched ceiling, so what limits it is transliteration
quality rather than information the Shahmukhi input never had.

---

## 1 Introduction

A multilingual model that shares meaning across languages should not care
which script a sentence arrived in, at least past the first few layers.
Whether it does is usually tested on language pairs that also differ in
script, so the two effects cannot be told apart. Digraphic languages, written
in two scripts, hold language constant.

Two 2026 papers take this route and disagree. Karne (2026) finds that Gemma
sparse-autoencoder features for identical Serbian sentences in Cyrillic and
Latin overlap far above chance, increasingly so with scale. Verma et al. (2026)
find that romanising Hindi and other languages yields language-associated
units nearly disjoint from the native-script ones.

We argue part of the answer lies in the pair. Serbian Cyrillic and Latin are
*balanced*: both are high-resource and related by a lossless one-to-one
mapping, so a model sees both often and loses nothing converting between
them. Synthetic romanisation, by contrast, produces text a model rarely sees.
Most digraphic languages sit in between, and Punjabi is a clear case of
*unbalanced digraphia*. Gurmukhi, used in India, is far better represented
online than Shahmukhi, used in Pakistan by more speakers; and Shahmukhi, a
Perso-Arabic script, usually leaves short vowels and gemination unwritten.
The two scripts differ in resource and in information.

This paper makes three contributions.

1. **A measurement that uses the pairing.** In a minimal pair, the content
   shared by a sentence's two renderings cancels in their difference. We
   define a paired displacement statistic on that difference and show on
   planted data that it is unaffected by content variance, whereas linear
   probe accuracy, the usual instrument, falls from 1.00 to 0.78 while the
   script signal is held exactly constant (Section 4.2).
2. **An information / script-shape decomposition.** Deleting from Gurmukhi
   the characters Shahmukhi omits yields an information-matched Gurmukhi. The
   Gurmukhi-Shahmukhi gap then decomposes into an information part (Gurmukhi
   vs matched Gurmukhi) and a script-shape part (matched Gurmukhi vs
   Shahmukhi) (Section 3.3).
3. **Evidence across models,** with every effect reported against the same
   architecture with random weights, which we find already separates the
   scripts strongly (Section 4.4), and against a balanced Serbian contrast run
   through the identical pipeline.

---

## 2 Related work

**Language neutrality of multilingual representations.** Pires et al. (2019)
and Wu and Dredze (2019) documented cross-lingual transfer in multilingual
BERT without shared vocabulary; Libovický et al. (2020) measured how
language-neutral its representations are; Conneau et al. (2020) scaled the
recipe in XLM-R. Kudugunta et al. (2019) observed that later encoder layers
of multilingual translation models reflect script less, with Serbian and
Croatian becoming near neighbours despite disjoint subwords. Neuron-level work
identifies language-associated units (Tang et al., 2024).

**Script as a variable.** Karne (2026) uses Serbian digraphia with SAE
features and finds script invariance. Verma et al. (2026) find
language-associated units conditioned on orthography, using romanisation
produced by a transliterator. Singh et al. (2026) introduce PuMVR, 1,000
parallel Punjabi items in Gurmukhi, Shahmukhi and Roman script, and find
vision-language models answer identical items differently by script; they
note the behavioural gap is consistent with data imbalance and tokeniser
confounds they cannot separate. We supply a representational analysis of the
same language with those confounds measured.

**Transliteration for transfer.** Transliterating related languages to a
common script helps low-resource transfer (Moosa et al., 2023); Jayakumar et
al. (2026) survey the area. Our intervention asks a narrower question: for one
language in two scripts, how much of a script gap transliteration can close,
and whether what remains is information the source script lacks.

**Method.** We use centred kernel alignment (Kornblith et al., 2019) with the
unbiased HSIC estimator (Song et al., 2012), probes with control tasks
(Hewitt and Liang, 2019), and Benjamini-Hochberg control (Benjamini and
Hochberg, 1995). Tokeniser quality follows Rust et al. (2021).

---

## 3 Data

### 3.1 Parallel text

The main set is FLORES+ dev and devtest (2,009 sentences), aligned by sentence
id across `pan_Guru`, `urd_Arab`, `hin_Deva` and `srp_Cyrl`. FLORES+ contains
no Shahmukhi and no Serbian Latin.

**Serbian Latin** is produced by the standard lossless Cyrillic-Latin mapping.

**Shahmukhi** is generated from the Gurmukhi. Gurmukhi-to-Shahmukhi is the
easy direction, since Gurmukhi writes every vowel Shahmukhi needs. We built a
rule system, selected its three ambiguous defaults on a development split of
PuMVR word pairs, and scored a held-out split once (Table 1). The pretrained
SLPG model (Shehzadi et al., 2024) is used instead only if it scores lower on
PuMVR in the same run ; on this run the rules won and were used throughout, at 5.6% held-out word CER, since the SLPG checkpoint's preprocessing could not be verified (Finding 6). The Shahmukhi condition
is therefore machine-generated, which makes it more regular than natural
writing; the human-written PuMVR set (Section 3.4) is our robustness check.

**Table 1.** Transliteration quality against human PuMVR text. Word figures
are on 233 held-out unique word pairs; sentence figures on all 1,000 items.

| system | word CER | 95% CI | word exact | sentence CER |
|---|---:|---|---:|---:|
| Shahmukhi to Gurmukhi, naive map | 50.9% | [47.5, 54.1] | 9.0% | 38.4% |
| Shahmukhi to Gurmukhi, context rules | 16.0% | [14.1, 18.0] | 43.3% | 15.1% |
| Gurmukhi to Shahmukhi, rules | 5.6% | [3.8, 7.6] | 83.3% | 6.4% |

Development and held-out error agree to within half a point in both directions (16.1% vs 16.0%; 5.2% vs 5.6%).

### 3.2 Shahmukhi omits information

Of the edits remaining after the best Shahmukhi-to-Gurmukhi rules, 38.7% are
insertions of sihari (ਿ), aunkar (ੁ) or addak (ੱ): short vowels and gemination
the Shahmukhi input does not contain. The rest is dominated by letters whose
Gurmukhi value is genuinely ambiguous (ن as ਨ or ਣ; و as ੋ, ੂ or ੌ; medial ی as
ੀ, ੇ or ੈ). A 162-configuration search over those choices moved error only
between 16.1% and 19.5%. The Punjabi pair therefore differs in information as
well as resource, a property Serbian lacks.

### 3.3 Conditions and contrasts

Every condition is aligned to the same sentences. Removing sihari, aunkar and
addak from Gurmukhi gives **de-voweled Gurmukhi**, an information-matched
control: 7.6% of Gurmukhi characters are removed, and the error of
rule-transliterated Shahmukhi against it falls from 15.1% to 10.4%, the
remainder being ambiguous letter choice rather than missing information.
Romanised versions of Gurmukhi, Shahmukhi and Urdu are produced with uroman
(Hermjakob et al., 2018). Romanising does not remove the information
difference: Gurmukhi ਪੰਜਾਬੀ romanises to *pamjaabii*, Shahmukhi پنجابی to
*pnjabi*.

**Table 2.** Contrasts.

| contrast | varies | held constant |
|---|---|---|
| Gurmukhi vs Shahmukhi | script and information | language, content |
| Gurmukhi vs de-voweled Gurmukhi | information | language, content, script |
| de-voweled Gurmukhi vs Shahmukhi | script shape | language, content |
| Shahmukhi vs Urdu | language | script, content |
| Hindi vs Urdu | script (register confound) | content |
| romanised Gurmukhi vs romanised Shahmukhi | information, in Latin | language, content |
| romanised Shahmukhi vs romanised Urdu | language, no script signal | content |
| Serbian Cyrillic vs Serbian Latin | script, balanced and lossless | language, content |

Hindi and Urdu share colloquial grammar but diverge in formal vocabulary; we
report that contrast as the weaker one.

### 3.4 Human-written robustness set

PuMVR (Singh et al., 2026) provides 1,000 questions written by native speakers
in Gurmukhi, Shahmukhi and Roman script. We run the Punjabi contrasts on these
questions as well, including a contrast against human romanisation.

### 3.5 Quality control of the generated Shahmukhi

A sample of 300 generated sentences is judged by a reader of both scripts
[PENDING: corpus error estimate and CI, awaiting the human judgement]. Sampling is stratified: 60% from pairs
flagged by an independent check (transliterating the Shahmukhi back with the
context rules and comparing to the source after removing differences no rule
could resolve), 40% at random, and the two strata are reweighted into a
corpus estimate. On held-out PuMVR the flag marks 3% of correct pairs, all
misaligned pairs, and 53% of pairs with one word replaced.

---

## 4 Method

### 4.1 Models and extraction

XLM-R base and large (Conneau et al., 2020), MuRIL, Qwen2.5 1.5B and 3B,
Gemma-2 2B and Aya Expanse 8B. Encoders are mean-pooled over real tokens,
decoders take the last real token, with padding handled explicitly on either
side. All hidden layers including the embedding layer are extracted. Inputs
are truncated at 256 tokens and the truncated fraction is reported per
condition, since high-fertility scripts truncate first. MuRIL and XLM-R base
share hidden width 768, which matters for the CKA comparison below.

### 4.2 Paired displacement

For aligned activations $A, B \in \mathbb{R}^{n \times d}$ at one layer, each
row L2-normalised, let $\delta_i = b_i - a_i$. We define

$$\text{consistency} = \frac{\lVert \bar{\delta} \rVert}{\sqrt{\frac{1}{n}\sum_i \lVert \delta_i \rVert^2}} \in [0, 1].$$

It is 1 when every sentence is displaced in the same direction and near
$1/\sqrt{n}$ for isotropic noise. Content common to both renderings of a
sentence cancels in $\delta_i$.

*Why not probe accuracy.* We planted a script displacement of fixed size and
direction and varied only the weight of a content component shared by both
renderings, mimicking activation-norm growth with depth. Probe accuracy fell
from 1.000 to 0.775 and selectivity from 0.470 to 0.187; Cohen's d along the
class-mean direction fell by 10.2. Consistency varied by less than 0.01. A falling
script-probe curve can therefore arise with no abstraction at all.

*Null.* The row-permutation null used for CKA is invalid here: a constant
displacement survives row permutation, so the baseline absorbs the signal. We
flip the sign of each $\delta_i$ independently (200 draws), which destroys any
shared direction while preserving magnitudes; the null mean agrees with the
analytic $1/\sqrt{n}$ floor (0.052 measured vs 0.058 analytic at $n = 300$).
We report consistency minus the null mean.

### 4.3 CKA and probes

Linear CKA uses the unbiased HSIC estimator. The standard biased estimator
reads 0.435 (d = 768) to 0.804 (d = 4096) on independent Gaussian data at
$n = 1000$, and the inflation grows with hidden width, so raw CKA is not
comparable across models; the unbiased estimator reads under 0.003 on the
same data. Each value is reported against 200 row permutations.

Probes are cross-validated logistic regressions discriminating the two
conditions of a contrast, with a Hewitt-Liang control task in which each
sentence id receives a random label reused across layers and conditions.
Keying control labels to sentence id rather than row keeps the control from
encoding script. We report selectivity and, optionally, its dependence on
regularisation strength.

### 4.4 Random-init baseline

Running the full pipeline on randomly initialised models produced a large
Gurmukhi-Shahmukhi displacement, significant at every layer. This is expected:
disjoint token inventories start at different embeddings and a random network
propagates the difference. Every contrast is therefore reported for the
trained model and for the same architecture with random weights and the same
tokeniser (three seeds), and our claims concern the difference.

### 4.5 Multiple comparisons

Permutation p-values for CKA and paired displacement are corrected with
Benjamini-Hochberg over all layers, contrasts and statistics of a model
jointly.

### 4.6 Intervention

*Controlled.* SIB-200 topic classification (Adelani et al., 2024) on the
`pan_Guru` sentences, with Shahmukhi generated as in 3.1. A logistic
classifier on frozen features is trained on Gurmukhi at the layer with the
best Gurmukhi validation accuracy, then tested on Gurmukhi, Shahmukhi,
Shahmukhi transliterated back to Gurmukhi, and de-voweled Gurmukhi. The gain
is restored minus Shahmukhi; if the restored condition matches de-voweled
Gurmukhi, the remaining gap is information absent from the input. Paired
differences use McNemar's exact test and a paired bootstrap. We also train
and test within Shahmukhi, which asks whether its representations carry the
label at all.

*Natural.* PuMVR questions and options, scored by length-normalised decoder
log-likelihood, in human Gurmukhi, human Shahmukhi, human Shahmukhi
transliterated to Gurmukhi, and human Roman. The task is text-only, so
absolute accuracy is not comparable to PuMVR's image-conditioned results; the
comparison across scripts is paired on identical items.

### 4.7 Validation of the pipeline

Every statistic was tested on activations with a planted depth structure,
where the pipeline must recover the planted crossover layer, and on a matched
null, where it must report none. The decomposition was tested on planted
worlds where the gap is all information (recovered script-shape share −0.01)
or all shape (0.99). The full pipeline was run end to end on PuMVR text with
small local models. Code and tests are released [URL].

---

## 5 Results

### 5.1 Tokenisation

Shahmukhi costs XLM-R 10.3% more tokens per character than Gurmukhi
(0.529 against 0.480, Wilcoxon p = 2e-192, n = 2,009 paired sentences) and
MuRIL 35.3% more (0.497 against 0.367). The direction reverses for the
decoders, whose byte-level BPE is far more economical on Perso-Arabic than on
Gurmukhi: Qwen2.5 spends 1.850 tokens per Gurmukhi character against 0.878
for Shahmukhi, a ratio of 0.47, and Gemma-2 0.819 against 0.622. The
tokenisation premium is therefore not a fixed property of the script pair but
of the tokeniser: neither script is uniformly the expensive one across
models.
Byte fallback is zero everywhere, but the fragment rate, which also catches
byte-level BPE, is not.

Two coverage figures matter for what follows. XLM-R maps 2.3% of Gurmukhi
characters to UNK against 0.006% of Shahmukhi, so its Punjabi script gap is
measured against a slightly degraded Gurmukhi condition, which if anything
works against the effect we report. MuRIL maps 13.4% of Serbian Cyrillic and
4.1% of Serbian Latin to UNK, which is too much to interpret: we report
MuRIL's Serbian contrast but do not rely on it.

### 5.2 Does representation track script, and at what depth?

Paired displacement is large and BH-significant at every layer of both
models, but so is the random-init baseline (Finding 7), so only the
difference is evidence. Trained minus random, averaged over three random-init
seeds, XLM-R base places Gurmukhi and Shahmukhi 0.533 apart at the embedding
layer, falls to a minimum of 0.307 at layer 4, and rises again to 0.433 at
the final layer. MuRIL follows the same shape with a shallower trough: 0.356,
a minimum of 0.270 at layer 1, and 0.461 at the end. The seed spread at the
minimum is 0.011 for XLM-R and 0.029 for MuRIL, an order of magnitude below
the effect.

The script signal is therefore attenuated in the lower middle of the stack
and then recovers, and at no depth in either model does it approach the
random-init baseline. Nothing here is consistent with script being discarded.

The strongest constraint comes from the control that removes script. Two
different languages in one script, romanised Shahmukhi against romanised
Urdu, sit at 0.010 and 0.014 above random at the embedding layer, reach a
minimum of
-0.009 in XLM-R and -0.048 in MuRIL, and end at 0.107 and 0.049. One language
in two scripts is displaced several times further than two languages in one
script. Whatever these models separate, orthography dominates language in
it.

There is no crossover, and the reason is instructive. A linear probe
separates the two scripts at 1.00 accuracy at every layer of both models,
with selectivity against a Hewitt and Liang control task flat at 0.45 to 0.50
throughout. A statistic at ceiling everywhere cannot locate a depth, which is
Finding 2 appearing in measured data rather than in simulation: the crossover
layer the pipeline reports (layer 2 for both models) is an artefact of a
saturated probe and we do not interpret it. The depth claim rests on paired
displacement alone.

### 5.3 Information versus script shape

Almost none of the Gurmukhi-Shahmukhi displacement is the information
Shahmukhi omits. The information-matched control, Gurmukhi with the short
vowels and gemination marks Shahmukhi does not write removed, is displaced
from Gurmukhi by 0.171 at the embedding layer and 0.284 at the last in XLM-R,
well below the 0.533 and 0.433 of the full script contrast.

The shape share, the part of the script gap surviving information matching,
must be formed from the same trained-minus-random quantities as everything
else, because the random-init runs themselves have shape shares of 1.00 to
1.05: on raw displacements the decomposition would be reporting a property
of the architecture. Corrected, it averages 0.930 in XLM-R (range 0.870 to
1.003) and 0.855 in MuRIL (0.748 to 1.034). On raw displacements the same
quantities read 0.978 and 0.941, which is the number a pipeline without
baselines would report.

De-voweled Gurmukhi against Shahmukhi, the contrast that varies script form
with information held approximately equal, tracks the full script contrast
closely: minimum 0.285 against 0.307 in XLM-R, 0.237 against 0.270 in MuRIL.
What separates the two scripts in representation is the form of the writing,
not what it fails to record.

### 5.4 Balanced versus unbalanced digraphia

Run through the identical pipeline, Serbian behaves differently. In
XLM-R base the Cyrillic-Latin displacement starts higher than Punjabi's
(0.614 against 0.533 above random) and falls to 0.105 at layer 5, a third of
Punjabi's 0.307 trough, before recovering to 0.541. Balanced digraphia is
substantially abstracted away in the middle of the stack; unbalanced
digraphia is not. Both recover in the upper layers, which is consistent with
script being re-encoded near the output rather than never encoded at all.

This is the paper's central comparison and it currently rests on one model.
MuRIL's Serbian trough is 0.184 against Punjabi's 0.270, a much smaller
separation, but MuRIL sends 13.4% of Serbian Cyrillic to UNK and its Serbian
numbers are not interpretable. We state the finding as XLM-R base shows it
and treat replication across the larger models as the test it needs.

### 5.5 Intervention

Topic classification on frozen features, trained on Gurmukhi at the
layer chosen on validation (layer 11 for both models), transfers poorly to
Shahmukhi. XLM-R scores 0.721 on Gurmukhi and 0.270 on Shahmukhi, a gap of
0.451 (McNemar p = 3e-18) against a 0.25 majority baseline. Transliterating
the Shahmukhi test set back to Gurmukhi recovers 0.461, a gain of 0.191
(95% CI [0.098, 0.284], McNemar p = 9e-05), which closes 42% of the gap.

The remainder is not simply the information Shahmukhi omits. The
information-matched ceiling, de-voweled Gurmukhi, scores 0.588, and restored
text stays significantly below it (McNemar p = 4e-04). Since our
transliteration carries 16.0% word CER, the binding constraint on the
intervention is transliteration quality, not missing vowels.

MuRIL starts from a smaller gap, 0.794 against 0.529 (p = 4e-11), and its
restoration gain of 0.069 is not significant (95% CI [-0.005, 0.147],
McNemar p = 0.11). We therefore report no reliable intervention effect for
MuRIL rather than the 26% of the gap its point estimate suggests.

### 5.6 Human-written text

The 1,000 human-written PuMVR items reproduce the pattern on text no
transliterator produced. Gurmukhi against Shahmukhi is displaced 0.844 at the
embedding layer of XLM-R, with a minimum of 0.491 and 0.660 at the last
layer; MuRIL gives 0.593, 0.498 and 0.735. The information-matched contrast
is again far smaller (0.305 to 0.406 in XLM-R). Random-init baselines were
not run on PuMVR, so these figures are raw rather than differences and are
not comparable in level with Section 5.2; the ordering, which is what the
robustness check is for, is the same.

The behavioural arm on the same human text does not support the picture the
representational results give, and we report it as a null. Scoring PuMVR
answer options by Qwen2.5-1.5B likelihood gives 0.281 on Gurmukhi, 0.325 on
Shahmukhi, 0.288 on transliterated-back Shahmukhi and 0.336 on Roman, against
a four-option chance rate of 0.25. Every condition is within 0.09 of chance,
the restoration gain is negative (-0.037, McNemar p = 0.04) and Shahmukhi
exceeds Gurmukhi (p = 0.02). At this model size the task is too hard for the
comparison to mean anything, and we draw no conclusion from it in either
direction; a decoder that actually performs the task is what the arm needs.


### 5.7 Does the gap close with scale?

[SCALE: XLM-R large, Qwen2.5-1.5B, Qwen2.5-3B and Gemma-2-2B through the same
five contrasts, each with a matched random-init baseline. Two within-family
ladders (XLM-R base to large, Qwen 1.5B to 3B) test whether the Punjabi
trough deepens toward Serbian's with capacity, which is what Karne (2026)
reports for a balanced pair.]

---

## 6 Limitations

The FLORES Shahmukhi is machine-generated (5.6% word CER on human text), and
generated text lacks the lexical irregularity of natural writing, especially
in Arabic-origin words. The human-written set is smaller (1,000 questions)
and question-style. De-voweling matches information only approximately: it
removes what Shahmukhi omits but cannot reproduce Shahmukhi's ambiguous
letters. Resource imbalance is measured through tokenisation and bounded by
the balanced Serbian contrast, but not manipulated. The Hindi-Urdu contrast
carries a register confound.

Three limits are specific to what we can currently claim. The results are two
base-size encoders, so the balanced-versus-unbalanced comparison that carries
the argument rests on XLM-R base alone; Section 5.7 is the replication it
needs. XLM-R sends 2.3% of Gurmukhi characters to UNK, an asymmetry we have
not diagnosed, though it biases against our own effect. MuRIL sends 13.4% of
Serbian Cyrillic to UNK, so its Serbian contrast is reported and not used.
Probes are at ceiling at every layer, so no depth claim rests on them, and
random-init baselines were not run on the human-written PuMVR set, whose
figures are therefore raw rather than differences.

Most importantly, the trained-minus-random difference on which every claim
rests carries no significance test of its own. The BH-corrected p values in
our tables attach to the sign-flip null for a single run, not to the
difference between two runs, and with three random-init seeds we report the
difference as a point estimate beside the seed range rather than as a tested
effect. The Serbian trough of 0.105 sits about seven seed ranges above zero
but is also the size of the upper-layer values of our no-script control, so
the balanced-digraphia claim should be read as an ordering between two
contrasts measured identically, not as a claim that Serbian's script signal
reaches zero.

---

## References

Adelani, D. I., et al. 2024. SIB-200: A simple, inclusive, and big evaluation
dataset for topic classification in 200+ languages and dialects. EACL.

Benjamini, Y., and Hochberg, Y. 1995. Controlling the false discovery rate.
Journal of the Royal Statistical Society B 57(1).

Conneau, A., et al. 2020. Unsupervised cross-lingual representation learning
at scale. ACL.

Hermjakob, U., May, J., and Knight, K. 2018. Out-of-the-box universal
romanization tool uroman. ACL System Demonstrations.

Hewitt, J., and Liang, P. 2019. Designing and interpreting probes with
control tasks. EMNLP-IJCNLP.

Jayakumar, T., Halder, D., and Dabre, R. 2026. Scripts through time: A survey
of the evolving role of transliteration in NLP. Findings of ACL.

Karne, S. 2026. One language, two scripts: Probing script-invariance in LLM
concept representations. arXiv:2603.08869.

Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. 2019. Similarity of
neural network representations revisited. ICML.

Kudugunta, S., Bapna, A., Caswell, I., and Firat, O. 2019. Investigating
multilingual NMT representations at scale. EMNLP-IJCNLP.

Libovický, J., Rosa, R., and Fraser, A. 2020. On the language neutrality of
pre-trained multilingual representations. Findings of EMNLP.

Moosa, I. M., Akhter, M. E., and Habib, A. B. 2023. Does transliteration help
multilingual language modeling? Findings of EACL.

Pires, T., Schlinger, E., and Garrette, D. 2019. How multilingual is
multilingual BERT? ACL.

Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., and Gurevych, I. 2021. How good
is your tokenizer? On the monolingual performance of multilingual language
models. ACL.

Shehzadi, A., Rauf, S. A., Malik, M. G. A., and Imran, M. 2024. Unsupervised
Punjabi corpus and neural machine transliteration system. Heliyon (under
review; cited from the SLPG model card).

Singh, P., Pawar, B., Reddiboina, M., and Sheth, R. 2026. Not truly
multilingual: Script consistency as a missing dimension in VLM evaluation.
Findings of EMNLP. arXiv:2606.17188.

Song, L., Smola, A., Gretton, A., Bedo, J., and Borgwardt, K. 2012. Feature
selection via dependence maximization. JMLR 13.

Tang, T., et al. 2024. Language-specific neurons: The key to multilingual
capabilities in large language models. ACL.

Verma, A. A. K., Chatterjee, A., Gupta, M., and Chakraborty, T. 2026.
Multilingual language models encode script over linguistic structure. ACL.

Wu, S., and Dredze, M. 2019. Beto, bentz, becas: The surprising cross-lingual
effectiveness of BERT. EMNLP-IJCNLP.
