# Removing the attention value matrix

**In attention, the value matrix turns each earlier token into the vector that gets copied. I removed it. Instead, each token keeps a few learned value vectors, and the context picks which one to use. The model trained to a lower loss than with the value matrix, and each stored vector learned to hold a different kind of information about its token.**

![Value embedding with context-chosen rows vs the value matrix](main_result.png)

## The core idea

When a token looks back at earlier tokens, attention copies a vector from each earlier token. That vector is called the **value**.

Normally the value comes from the **value matrix**: it turns what the model knows about the earlier token at that moment into the vector that gets copied.

Recent models (modded-nanogpt, nanochat) add a **value embedding**: a learned table with one vector per token in the vocabulary, added to the value. It works like a dictionary: look up the token, get its stored vector.

My question: if the value embedding is good enough, is the value matrix needed at all?

## What I changed

1. **Gate:** depending on the sentence, the model can scale the value embedding up or down for each token (nanochat does this).
2. **Slots:** each token gets 4 rows in the value embedding instead of 1, and the layer input picks which row to use. The MoME paper calls these rows slots.
3. **No value matrix:** the value is now only the chosen value-embedding row.

Together they reach 0.996 against 1.084 for normal attention, better on all 4 seeds. When the 4 slots are averaged evenly instead of chosen, half the gain is gone, so the gain comes from choosing a row, not from having more weights. Spending the same weights on 3 more feed-forward experts gave nothing.

## What each slot learns

![Second layer: which slot each task uses](slots.png)

In the second layer, the task decides the slot. One slot holds the token itself, used by every task that copies a token. Another holds the token's common fact, and another its rare fact.

The first layer cannot do this. Its input is only the token, before any context is read, so its slot choice is fixed per token. Slots in the second layer alone keep almost all the gain.

## Where it helps and where it does not

I changed the training one thing at a time to make it more like a large language model.

![Where the slots help](situations.png)

- **Good at memorizing many facts**, most of all when most facts are rare, as in real text.
- **Good at copying the exact token**: sorting and "look up a key, then recall it" improve even with few facts.
- **Not good at information that comes from the context.** The value embedding only knows the token. Without the value matrix, a wide model got much worse at lookup inside the sentence.
- **No gain when the model already has room**: a wide model, or long training on few facts, memorizes everything anyway.

## What this predicts for large language models

Large models see far more rare facts than they can memorize, and the gain held in the toy setting closest to that. **My guess: keep the value matrix and add context-chosen slots in layers after the first**, which helped in every situation and never hurt. To test it on a real model: train a size ladder with and without slots, split the loss by token frequency (the gain should sit on rare tokens), and record which slot each layer picks.

## Is this new?

Mostly no. [MoME](https://arxiv.org/abs/2609.15126) adds context-chosen slots on top of the value matrix. [Bank of Values](https://arxiv.org/abs/2606.02780) and [Memory Attention](https://arxiv.org/abs/2609.28399) replace values with a value embedding. [nanochat](https://github.com/karpathy/nanochat/discussions/481) gates the value embedding.

Not found in published work: slots with no value matrix in every layer, a measurement of what each slot holds, and the map of where it helps and where it does not.

## Setup

A small transformer trained for 5–20 seconds on a MacBook CPU, on 13 small tasks mixed into one stream: predicting the next word of text, copying, facts, key → value lookup, sorting, counting and multi-step tasks. [RESULTS.md](RESULTS.md) has every number and command, [toy/](toy/) the code and [chart.py](chart.py) the charts.
