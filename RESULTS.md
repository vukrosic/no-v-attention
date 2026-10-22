# Results

All runs: 2-layer transformer, width 64 unless stated, 2 heads, RoPE, QK-norm, SwiGLU, RMSNorm, gated attention, a value embedding added to the values (AdamW at 32× the base lr), Muon on the matrices, AdamW elsewhere. CPU, about 5 s per run.

**Loss used:** the weighted test loss of the 7 base skills (lower is better). Two runs differing only in seed differ by about 0.02–0.03.

**Knobs:**
- `VE_GATE=1`: the value embedding is scaled per head by 2·sigmoid(W h), with W starting at zero.
- `VE_SLOTS=k`: k slots per word, mixed by softmax(W h) of the layer input.
- `VS_FIX=1`: the slots are mixed evenly, as a control.
- `VS_L=1`: slots only in the second layer.
- `FREEZE=v V_ZERO=1`: the value matrix is set to zero and kept there, so no value matrix.

## Main result: task suite, 400 steps, no mixture of experts

`cd toy && env [knobs] ./suite_run.sh SEED`, then `python3 summ.py`

| setup | s0 | s1 | s2 | s3 | mean |
|---|---|---|---|---|---|
| normal attention | 1.073 | 1.100 | 1.073 | 1.089 | 1.084 |
| gate | 1.039 | 1.039 | 1.029 | 1.069 | 1.044 |
| 4 slots | 1.091 | 1.067 | 1.067 | 1.030 | 1.064 |
| gate, no value matrix | 1.069 | 1.047 | 1.044 | 1.059 | 1.055 |
| 4 slots, no value matrix | 1.053 | 1.030 | 1.047 | 1.020 | 1.038 |
| gate + 4 slots, value matrix kept | 1.005 | 1.016 | 0.992 | 0.998 | 1.003 |
| **gate + 4 slots, no value matrix** | 1.019 | 0.994 | 0.978 | 0.992 | **0.996** |
| control: gate + 4 slots mixed evenly, no value matrix | 1.055 | 1.059 | 1.038 | 1.027 | 1.045 |
| gate + 2 slots, no value matrix | 1.083 | 1.012 | 1.023 | 1.029 | 1.037 |
| gate + 8 slots, no value matrix | 1.001 | 0.983 | 1.016 | 0.991 | 0.998 |
| gate + 4 slots in layer 2 only, value matrix kept | 1.023 | 1.012 | 0.991 | 1.025 | 1.013 |
| gate + 4 slots in layer 2 only, no value matrix | 1.003 | 1.020 | 0.993 | 1.017 | 1.008 |
| gate + 8 slots in layer 2 only, value matrix kept | 1.050 | 1.017 | 0.988 | 1.013 | 1.017 |

- Most of the gain is on the rare-fact skill (loss about 1.1 → 0.3–0.5) and on sorting.
- Once slots are picked by context, the value matrix adds nothing (1.003 kept vs 0.996 removed, inside noise).
- More than 4 slots gives nothing more.

## Same test on the toy with a mixture of experts (8 experts, 260 steps)

`cd toy && env [knobs] ./toy.sh SEED`

| setup | s0 | s1 | s2 | s3 | mean |
|---|---|---|---|---|---|
| normal attention | 1.360 | 1.457 | 1.367 | 1.370 | 1.389 |
| gate + 4 slots, no value matrix | 1.297 | 1.337 | 1.320 | 1.307 | **1.315** |
| control: slots mixed evenly | 1.326 | 1.380 | 1.332 | 1.322 | 1.340 |
| same extra weights as 3 more experts (`MOE=11`) | 1.380 | 1.442 | 1.375 | 1.404 | 1.400 |

The 3 extra slots per word are about 49k weights per layer, the size of about 3 experts.

## What the slots hold

`cd toy && env [suite knobs] VE_GATE=1 VE_SLOTS=4 python3 sprobe.py 0 400` (suite knobs: as set in `suite_run.sh` and `toy.sh` with `MOE=0`)

| layer | how sharp the pick is (top slot weight) | same slot every time for a given word | what decides the slot |
|---|---|---|---|
| 1 | 0.44 (soft) | 100% | only the word: the first layer has no context yet |
| 2 | 0.83 (hard) | 68%, and 97% for a given word + task | the task |

Second layer, share of positions whose top slot is each slot:

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

| layout | 1 slot | 4 slots | 16 slots | 16 slots mixed evenly |
|---|---|---|---|---|
| 64 words × 4 meanings | 1.14 (0.74) | 0.86 (0.83) | 0.96 (0.78) | |
| 16 words × 16 meanings | 1.70 (0.63) | 1.48 (0.67) | 1.83 (0.56) | |
| 256 words × 1 meaning | 0.38 (0.92) | 0.11 (0.97) | 0.05 (0.99) | 0.40 (0.93) |

- The guess "slots = meanings per word" is wrong. Words with 16 meanings gain little, and words with one meaning gain the most.
- With 256 subjects, every word is both something to copy and a fact subject, which is the case the slots serve.

## Width

lr is 0.028 at every width and was not retuned.

| width | normal attention | gate + 4 slots, no value matrix | change |
|---|---|---|---|
| 64 (4 seeds) | 1.084 | 0.996 | −0.088 |
| 128 (4 seeds) | 1.027 / 1.068 / 1.045 / 1.060 = 1.050 | 1.047 / 1.010 / 1.043 / 1.017 = 1.029 | −0.021 |
| 256 (2 seeds) | 1.083 / 1.052 | 1.070 / 1.137 | +0.036 |
| 256, value matrix kept | | 1.061 / 1.069 | none |
| 256, 256 facts (`NF=256`) | 1.371 / 1.386 | 1.391 / 1.397 | loss on all tasks 1.40 → 1.28 |

- At width 256 the normal model already holds the 64 facts, so there is nothing for the slots to add.
- With 256 facts, the slots help facts again: look-up-then-fact accuracy goes 0.06 → 0.65 and 0.34 → 0.62.
- With no value matrix at width 256, key → value lookup got worse in 2 of 4 runs (loss 1.61 and 1.51, against at most 0.69 with the value matrix kept).

Width 256 with 256 facts, gate, value matrix kept, loss on all tasks (2 seeds):

| slots | seed 0 | seed 1 |
|---|---|---|
| none (normal attention) | 1.449 | 1.352 |
| 1 | 1.273 | 1.287 |
| 2 | 1.166 | 1.118 |
| 8 | 1.514 | 1.214 |
| 16 | 1.161 | 1.011 |

Too noisy to choose a slot count at this width.

## Number of facts to memorize

Is the gain from extra room for facts? Test: change how many facts the model must memorize (`NF`) at width 32 and 64, on the task suite above, 2 seeds each. "Fact loss" is the mean loss of the three fact tasks (common fact, rare fact, look up then common fact). "Other" is the mean of copy, repeat 2 back, key → value lookup, sort, count and next word of text.

| width | facts | fact loss: normal → slots | other tasks: normal → slots | all 13 tasks: normal → slots |
|---|---|---|---|---|
| 32 | 16 | 0.10 → 0.04 | 1.07 → 1.08 | 0.846 → 0.786 |
| 32 | 32 | 0.27 → 0.24 | 1.17 → 1.02 | 0.943 → 0.838 |
| 32 | 64 | 1.30 → 1.04 | 1.15 → 1.11 | 1.252 → 1.075 |
| 32 | 128 | 2.19 → 1.53 | 1.16 → 1.01 | 1.466 → 1.179 |
| 32 | 256 | 3.22 → 2.12 | 1.10 → 1.09 | 1.675 → 1.351 |
| 64 | 16 | 0.06 → 0.07 | 0.96 → 0.88 | 0.749 → 0.675 |
| 64 | 32 | 0.08 → 0.03 | 0.97 → 0.96 | 0.776 → 0.708 |
| 64 | 64 | 0.45 → 0.22 | 0.91 → 0.87 | 0.830 → 0.728 |
| 64 | 128 | 1.33 → 0.53 | 0.99 → 0.85 | 1.100 → 0.797 |
| 64 | 256 | 2.50 → 1.61 | 0.96 → 0.82 | 1.377 → 1.059 |

- The fact gain grows with the number of facts: none at 16 facts (width 64), −0.22 at 64, −0.80 to −0.89 at 128–256. This supports the "extra room for facts" explanation.
- A wider model needing more facts before the gain starts was not visible between width 32 and 64: the two curves are about the same. The width table above shows the shift only at 128 and 256.
- A second gain does not depend on facts: with only 16 facts at width 64, sorting improves by 0.41 and "look up a key, then recall" by 0.48. Both need a clean copy of the token, which is what the second-layer "token itself" slot holds.
- Counting got worse with slots at 16–32 facts (+0.09 to +0.55).

Command: `cd toy && VE_GATE=1 VE_SLOTS=4 FREEZE=v V_ZERO=1 WIDTH=64 NF=128 ./suite_run.sh 0` (drop the first four knobs for normal attention).

## Literature

- MoME, arXiv 2609.15126: gate + top-2 of M slots per word, added on top of the value matrix; about 0.002 bpb better than a plain value embedding at equal size on nanochat. It never removes the value matrix.
- Bank of Values, arXiv 2606.02780: value embeddings replace the values in the last third of layers, 135M and 780M models.
- Memory Attention, arXiv 2609.28399: value = key + value embedding, with no value matrix and one row per word. It estimates a 50% cache saving but does not measure it.
- nanochat: a value embedding with a sigmoid gate from the layer input.

## Training situations

Each situation changes one thing from the default (2 layers, width 64, 256 words, 64 facts, 400 steps). Loss is the mean over all 13 tasks. "Slots, no value matrix" = gate + 4 context-chosen slots in every layer, value matrix removed. "Value matrix kept" = gate + 4 slots in layers after the first, value matrix kept.

| situation | knobs | seeds | normal attention | slots, no value matrix | value matrix kept |
|---|---|---|---|---|---|
| default | | 3 | 0.823 | 0.718 (−0.105) | 0.720 (−0.103) |
| 4 layers | `LAYERS=4` | 3 | 0.807 | 0.712 (−0.096) | 0.695 (−0.113) |
| 1024 words | `NV=1024` | 3 | 1.306 | 1.104 (−0.202) | 1.150 (−0.157) |
| 256 facts, uneven frequency | `NF=256 FACTZIPF=1` | 3 | 1.219 | 0.885 (−0.334) | 0.923 (−0.296) |
| 3x longer training | `STEPS=1200` | 3 | 0.439 | 0.408 (−0.031) | 0.401 (−0.038) |
| all three | `NV=1024 NF=256 FACTZIPF=1 STEPS=1200` | 2 | 0.748 | 0.673 (−0.075) | 0.658 (−0.090) |
| width 256 | `WIDTH=256` | 2 | 0.781 | 0.820 (+0.039) | 0.775 (−0.006) |

Per-task loss change against normal attention (negative = better). chain = next word of text, factc / factr = common / rare fact, rfact = look up then common fact, rrecall = look up a key then recall, agree = agreement over a distance, bind = bind two items, sum3 = sum of three.

| situation | version | chain | copy | factc | back2 | recall | sort | factr | count | rfact | agree | rrecall | bind | sum3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | no value matrix | −0.05 | −0.03 | −0.04 | −0.03 | −0.09 | −0.40 | −0.39 | +0.41 | −0.25 | −0.00 | −0.35 | +0.01 | −0.16 |
| default | value matrix kept | −0.04 | −0.02 | −0.04 | −0.04 | −0.09 | −0.23 | −0.44 | −0.04 | −0.22 | +0.20 | −0.25 | +0.00 | −0.15 |
| 4 layers | no value matrix | −0.05 | −0.04 | −0.03 | −0.01 | −0.07 | −0.12 | −0.29 | +0.36 | −0.17 | −0.01 | −0.44 | +0.00 | −0.37 |
| 4 layers | value matrix kept | −0.04 | −0.05 | −0.03 | −0.03 | −0.08 | −0.24 | −0.20 | +0.17 | −0.08 | −0.01 | −0.42 | −0.02 | −0.41 |
| 1024 words | no value matrix | −0.05 | −0.11 | −0.03 | −0.14 | −0.25 | −0.43 | −0.41 | −0.34 | −0.19 | −0.00 | −0.50 | −0.05 | −0.12 |
| 1024 words | value matrix kept | −0.04 | −0.11 | −0.03 | −0.11 | −0.24 | −0.34 | −0.28 | −0.27 | −0.12 | +0.17 | −0.43 | −0.06 | −0.16 |
| 256 uneven facts | no value matrix | −0.03 | −0.03 | −0.60 | −0.05 | −0.12 | −0.35 | −0.02 | −0.35 | −1.72 | −0.43 | −0.45 | +0.00 | −0.20 |
| 256 uneven facts | value matrix kept | −0.01 | −0.03 | −0.62 | −0.03 | −0.12 | −0.30 | −0.25 | −0.12 | −1.75 | −0.02 | −0.46 | −0.04 | −0.09 |
| 3x training | no value matrix | −0.01 | +0.01 | −0.00 | −0.00 | −0.07 | −0.06 | −0.02 | +0.08 | −0.01 | +0.00 | −0.23 | −0.00 | −0.09 |
| 3x training | value matrix kept | −0.01 | −0.00 | −0.00 | +0.00 | −0.06 | −0.02 | −0.02 | −0.06 | +0.00 | +0.02 | −0.28 | −0.01 | −0.07 |
| all three | no value matrix | −0.01 | −0.06 | −0.02 | −0.05 | −0.13 | −0.14 | −0.38 | +0.02 | −0.11 | +0.19 | −0.28 | +0.02 | −0.03 |
| all three | value matrix kept | +0.00 | −0.05 | −0.02 | −0.08 | −0.12 | −0.18 | −0.30 | −0.05 | −0.11 | −0.08 | −0.16 | +0.01 | −0.05 |
| width 256 | no value matrix | +0.02 | −0.01 | +0.00 | +0.02 | +0.60 | −0.26 | −0.00 | +0.26 | −0.03 | +0.00 | +0.03 | +0.01 | −0.14 |
| width 256 | value matrix kept | +0.00 | +0.02 | +0.00 | −0.01 | +0.05 | −0.11 | −0.03 | −0.02 | +0.07 | +0.00 | −0.15 | −0.01 | +0.12 |

- With 3x training on 64 facts, normal attention learns every fact (fact loss 0.02), and only "look up a key, then recall" keeps a clear gain.
- Without the value matrix, counting got worse in 3 situations and better in 2: noisy, no rule.
- Without the value matrix at width 256, key → value lookup (recall) got much worse (+0.60). With the value matrix kept it did not (+0.05).
- Noise between seeds is about 0.02–0.05 on the 13-task mean.

Command: `cd toy && LAYERS=4 ./ablation.sh depth4 slots 0` (arms: `base`, `slots`, `keepv`; any knob from the table in front). Results go to `toy/ablation_runs.jsonl`. 4 layers take about 9 s per run, width 256 about 25 s, 3x training about 18 s.
