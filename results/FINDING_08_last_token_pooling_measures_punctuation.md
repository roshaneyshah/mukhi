# Finding 08: last-token pooling turns the paired statistic into a punctuation probe

## What happened

The six-model run put three decoders (Qwen2.5-1.5B, Qwen2.5-3B, Gemma-2-2B)
beside three encoders. The decoders behaved unlike the encoders from the
first layer: trained minus random-init was 0.000 at the embedding layer for
every Punjabi contrast, against 0.53 to 0.54 for the encoders.

Raw consistency at layer 0, Gurmukhi against Shahmukhi. Random-init is the
mean over the available seeds (three for XLM-R base, one for the rest):

| model | pooling | trained | random-init |
|---|---|---:|---:|
| xlm-roberta-base | mean | 0.952 | 0.415 |
| xlm-roberta-large | mean | 0.960 | 0.417 |
| Qwen2.5-1.5B | last | 0.979 | 0.980 |
| Qwen2.5-3B | last | 0.980 | 0.980 |
| gemma-2-2b | last | 0.970 | 0.974 |

A trained model and a randomly initialised one agreeing to two decimals is
not a result. It is the statistic measuring something that does not depend on
the weights.

## Why

Decoders are pooled at the last token. At layer 0 that representation is the
embedding vector of the final token and nothing else. In FLORES, 1,941 of
2,009 Punjabi sentences end in a full stop, and the two scripts use different
ones: Gurmukhi `।` (U+0964, the Devanagari danda, which is what FLORES+
uses in Gurmukhi text), Shahmukhi `۔` (U+06D4). So the per-item
difference is, for 97% of items, the same fixed vector: embedding(`۔`) minus
embedding(`।`). A constant offset has consistency 1 by construction, whatever
the embedding matrix contains, which is why trained and random agree.

The paired statistic is designed so that shared content cancels and script
survives. Under last-token pooling the sentence never enters it, so what
survives is the identity of one punctuation mark.

## Why it also breaks the central comparison

Serbian is worse, in the opposite direction. Both Serbian conditions end in
`.`, the same token, so the final-token difference is the zero vector and the
Serbian contrast is measuring residual noise. Punjabi gets a constant offset,
Serbian gets nothing. The balanced-versus-unbalanced comparison on decoders
is therefore not a comparison between two script pairs but between a pair
whose terminal punctuation differs and a pair whose terminal punctuation does
not.

## Consequence

1. No decoder result from this run enters the paper. The three decoder models
   are reported as a methodological finding, not as evidence about scale.
2. The scale claim rests on the encoder ladder, XLM-R base to large, which is
   mean-pooled and where the effect replicates and sharpens.
3. Any future decoder run must either pool over the sequence, matching the
   encoders, or strip terminal punctuation from every condition first. Mean
   pooling is the better fix, since it makes the two model families
   comparable. Stripping punctuation alone leaves last-token pooling
   measuring one arbitrary token.
4. This is a specific instance of a general hazard: a pooling choice can make
   a statistic depend on a single token that correlates with the condition
   label. The random-init baseline is what exposed it. Without Finding 07 the
   decoder numbers would have looked like large, highly significant script
   effects, and the last-layer values (Punjabi 0.449 to 0.608, Serbian 0.837
   to 0.869) would have been read as decoders separating Serbian more than
   Punjabi, reversing the paper's claim on the strength of punctuation.
