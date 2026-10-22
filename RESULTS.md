# Results

All runs: 2-layer transformer, width 64 unless stated, 2 heads, RoPE, QK-norm, SwiGLU, RMSNorm, gated attention, a word table added to the values (AdamW at 32× the base lr), Muon on the matrices, AdamW elsewhere. CPU, about 5 s per run.

**Loss used:** the weighted test loss of the 7 base skills (lower is better). Two runs differing only in seed differ by about 0.02–0.03.

**Knobs:**
- `VE_GATE=1`: the word table is scaled per head by 2·sigmoid(W h), with W starting at zero.
- `VE_SLOTS=k`: k drawers per word, mixed by softmax(W h) of the layer input.
- `VS_FIX=1`: the drawers are mixed evenly, as a control.
- `VS_L=1`: drawers only in the second layer.
- `FREEZE=v V_ZERO=1`: the value matrix is set to zero and kept there, so no value matrix.

## Main result: task suite, 400 steps, no mixture of experts

`cd toy && env [knobs] ./suite_run.sh SEED`, then `python3 summ.py`

| setup | s0 | s1 | s2 | s3 | mean |
|---|---|---|---|---|---|
| normal attention | 1.073 | 1.100 | 1.073 | 1.089 | 1.084 |
| gate | 1.039 | 1.039 | 1.029 | 1.069 | 1.044 |
| 4 drawers | 1.091 | 1.067 | 1.067 | 1.030 | 1.064 |
| gate, no value matrix | 1.069 | 1.047 | 1.044 | 1.059 | 1.055 |
| 4 drawers, no value matrix | 1.053 | 1.030 | 1.047 | 1.020 | 1.038 |
| gate + 4 drawers, value matrix kept | 1.005 | 1.016 | 0.992 | 0.998 | 1.003 |
| **gate + 4 drawers, no value matrix** | 1.019 | 0.994 | 0.978 | 0.992 | **0.996** |
| control: gate + 4 drawers mixed evenly, no value matrix | 1.055 | 1.059 | 1.038 | 1.027 | 1.045 |
| gate + 2 drawers, no value matrix | 1.083 | 1.012 | 1.023 | 1.029 | 1.037 |
| gate + 8 drawers, no value matrix | 1.001 | 0.983 | 1.016 | 0.991 | 0.998 |
| gate + 4 drawers in layer 2 only, value matrix kept | 1.023 | 1.012 | 0.991 | 1.025 | 1.013 |
| gate + 4 drawers in layer 2 only, no value matrix | 1.003 | 1.020 | 0.993 | 1.017 | 1.008 |
| gate + 8 drawers in layer 2 only, value matrix kept | 1.050 | 1.017 | 0.988 | 1.013 | 1.017 |

- Most of the gain is on the rare-fact skill (loss about 1.1 → 0.3–0.5) and on sorting.
- Once drawers are picked by context, the value matrix adds nothing (1.003 kept vs 0.996 removed, inside noise).
- More than 4 drawers gives nothing more.

## Same test on the toy with a mixture of experts (8 experts, 260 steps)

`cd toy && env [knobs] ./toy.sh SEED`

| setup | s0 | s1 | s2 | s3 | mean |
|---|---|---|---|---|---|
| normal attention | 1.360 | 1.457 | 1.367 | 1.370 | 1.389 |
| gate + 4 drawers, no value matrix | 1.297 | 1.337 | 1.320 | 1.307 | **1.315** |
| control: drawers mixed evenly | 1.326 | 1.380 | 1.332 | 1.322 | 1.340 |
| same extra weights as 3 more experts (`MOE=11`) | 1.380 | 1.442 | 1.375 | 1.404 | 1.400 |

The 3 extra drawers per word are about 49k weights per layer, the size of about 3 experts.

## What the drawers hold

`cd toy && env [suite knobs] VE_GATE=1 VE_SLOTS=4 python3 sprobe.py 0 400` (suite knobs: as set in `suite_run.sh` and `toy.sh` with `MOE=0`)

| layer | how sharp the pick is (top drawer weight) | same drawer every time for a given word | what decides the drawer |
|---|---|---|---|
| 1 | 0.44 (soft) | 100% | only the word: the first layer has no context yet |
| 2 | 0.83 (hard) | 68%, and 97% for a given word + task | the task |

Second layer, share of positions whose top drawer is each drawer:

| task | d1 | d2 | d3 | d4 |
|---|---|---|---|---|
| copy a repeated part | 0 | 0 | 1.00 | 0 |
| repeat the word 2 back | 0 | 0 | 0.99 | 0 |
| key → value lookup | 0 | 0.01 | 0.98 | 0.02 |
| reverse lookup | 0 | 0 | 1.00 | 0 |
| attribute binding | 0 | 0 | 1.00 | 0 |
| sort | 0 | 0.12 | 0.88 | 0 |
| common fact | 0 | 0.99 | 0.01 | 0 |
| look up, then common fact | 0 | 1.00 | 0 | 0 |
| rare fact | 0 | 0 | 0 | 1.00 |
| next word of text | 0.45 | 0.22 | 0.04 | 0.29 |
| count | 0.88 | 0.11 | 0 | 0 |
| agreement | 0 | 0.11 | 0.20 | 0.68 |
| sum | 0 | 0 | 0.31 | 0.69 |

## Words with many meanings

The `frel` task (`EXTRA=frel`) is [relation, subject, SEP] → the answer for that pair. `FREL` sets the number of relations per subject and `NF` the number of subjects. Every row below has 256 facts. Setup: gate, value matrix kept, 3 seeds. Cells show frel loss (accuracy):

| layout | 1 drawer | 4 drawers | 16 drawers | 16 drawers mixed evenly |
|---|---|---|---|---|
| 64 words × 4 meanings | 1.14 (0.74) | 0.86 (0.83) | 0.96 (0.78) | |
| 16 words × 16 meanings | 1.70 (0.63) | 1.48 (0.67) | 1.83 (0.56) | |
| 256 words × 1 meaning | 0.38 (0.92) | 0.11 (0.97) | 0.05 (0.99) | 0.40 (0.93) |

- The guess "drawers = meanings per word" is wrong. Words with 16 meanings gain little, and words with one meaning gain the most.
- With 256 subjects, every word is both something to copy and a fact subject, which is the case the drawers serve.

## Width

lr is 0.028 at every width and was not retuned.

| width | normal attention | gate + 4 drawers, no value matrix | change |
|---|---|---|---|
| 64 (4 seeds) | 1.084 | 0.996 | −0.088 |
| 128 (4 seeds) | 1.027 / 1.068 / 1.045 / 1.060 = 1.050 | 1.047 / 1.010 / 1.043 / 1.017 = 1.029 | −0.021 |
| 256 (2 seeds) | 1.083 / 1.052 | 1.070 / 1.137 | +0.036 |
| 256, value matrix kept | | 1.061 / 1.069 | none |
| 256, 256 facts (`NF=256`) | 1.371 / 1.386 | 1.391 / 1.397 | loss on all tasks 1.40 → 1.28 |

- At width 256 the normal model already holds the 64 facts, so there is nothing for the drawers to add.
- With 256 facts, the drawers help facts again: look-up-then-fact accuracy goes 0.06 → 0.65 and 0.34 → 0.62.
- With no value matrix at width 256, key → value lookup got worse in 2 of 4 runs (loss 1.61 and 1.51, against at most 0.69 with the value matrix kept).

Width 256 with 256 facts, gate, value matrix kept, loss on all tasks (2 seeds):

| drawers | seed 0 | seed 1 |
|---|---|---|
| none (normal attention) | 1.449 | 1.352 |
| 1 | 1.273 | 1.287 |
| 2 | 1.166 | 1.118 |
| 8 | 1.514 | 1.214 |
| 16 | 1.161 | 1.011 |

Too noisy to choose a drawer count at this width.

## Literature

- MoME, arXiv 2609.15126: gate + top-2 of M drawers per word, added on top of the value matrix; about 0.002 bpb better than a plain word table at equal size on nanochat. It never removes the value matrix.
- Bank of Values, arXiv 2606.02780: word tables replace the values in the last third of layers, 135M and 780M models.
- Memory Attention, arXiv 2609.28399: value = key + word table, with no value matrix and one row per word. It estimates a 50% cache saving but does not measure it.
- nanochat: a word table with a sigmoid gate from the layer input.
