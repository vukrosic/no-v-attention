# Where it works and which layers need what

## Where it helps and where it does not

I changed the training one thing at a time to make it more like a large language model. [RESULTS.md](RESULTS.md) has every task and seed.

![Where the slots help](situations.png)

## Which layers need the value matrix

The first layer's input is only the token, so its value is already a fixed vector per token: a value-embedding table there repeats it. Later layers have read the context, so a table there brings back the plain token, which the context has blurred.

![Which layers get a value-embedding table](layers.png)

In a 4-layer model, the second layer is the place for the table, and a table in the first layer alone was worse than no table (it also learns 32 times faster than the rest, which hurt). Splitting the layers worked when the early layers use only stored vectors and the later layers keep the value matrix to read the context. When I removed the value matrix but kept slots in every layer, no layer clearly needed it in a narrow model; in a wide model, removing it from the early layers hurt more than from the last ones, but within the noise.

This fits [ROME](https://arxiv.org/abs/2202.05262), which found facts stored in the middle layers, and [nanochat](https://github.com/karpathy/nanochat/discussions/481), which found tables in alternating layers worked best and warned against early layers only. [Bank of Values](https://arxiv.org/abs/2606.02780) replaced the values in the last third of the layers; in my split test, stored vectors only in the last layers did worse than normal attention.

## What this predicts for large language models

Large models see far more rare facts than they can memorize, and the gain held in the toy setting closest to that. **My guess: keep the value matrix and add context-chosen slots in layers after the first, starting early**, which helped in every situation and never hurt. To test it on a real model: train a size ladder with and without slots, split the loss by token frequency (the gain should sit on rare tokens), and record which slot each layer picks.
