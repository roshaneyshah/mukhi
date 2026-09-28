# Finding 07: any network separates the scripts; only trained minus random-init counts

Found 2026-09-21 in a dry run of the full pipeline on PuMVR human text, using
tiny RANDOMLY INITIALISED encoder and decoder models.

## What happened

The random-weight decoder produced a paired Gurmukhi/Shahmukhi displacement of
0.58 at the last layer, significant after Benjamini-Hochberg at every layer.
Nothing had been learned. Gurmukhi and Shahmukhi use disjoint token
inventories, so their inputs start at different embeddings, and a random
network carries that difference through every layer.

So a significant script displacement in a trained model is not, by itself,
evidence that the model encodes script. It is what any network does with
disjoint tokens.

## The fix

`--random-init` builds the same architecture from its config with random
weights and the same tokeniser; `--seed` repeats it. The report places, for
every contrast, the trained model's displacement beside the random-init mean
and seed range, and reports trained minus random-init. The claim the paper can
make is about that difference.

The dry run shows the baseline doing its job: the tiny "trained" encoder was
itself a seed-0 random network, and trained minus random-init came out at
exactly 0.000 on all five contrasts, while every raw displacement was large
and significant. Across two random seeds the displacement varied by about
0.04, so a trained-minus-random difference below the seed range should be read
as noise.

## Recommended runs on Kaggle

Random-init baselines, 3 seeds each, for XLM-R base, MuRIL, XLM-R large and
Qwen2.5-1.5B. Skip Aya-8B: initialising it in fp32 needs about 32 GB of host
memory, above Kaggle's limit; the smaller models establish the baseline.
