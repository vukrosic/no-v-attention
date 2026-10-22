# Removing the attention value matrix

**In attention, the value matrix turns each earlier token into the vector that gets copied. I removed it. Instead, each token keeps a few learned value vectors, and the context picks which one to use. The model trained to a lower loss than with the value matrix, and each stored vector learned to hold a different kind of information about its token.**

- **Good at:** memorizing many rare facts, and copying the exact token.
- **Bad at:** information that comes from the context, which still needs the value matrix.
- **Little gain** when the model already has room to memorize everything.
- **Where to put it:** in the early layers after the first, with the value matrix kept in the later layers.
- **Safest version:** keep the value matrix and add the stored vectors on top; it never hurt.

![Value embedding with context-chosen rows vs the value matrix](main_result.png)

## The core idea

When a token looks back at earlier tokens, attention copies a vector from each earlier token. That vector is called the **value**.

Normally the value comes from the **value matrix**: it turns what the model knows about the earlier token at that moment into the vector that gets copied.

Recent models (modded-nanogpt, nanochat) add a **value embedding**: a learned table with one vector per token in the vocabulary, added to the value. It works like a dictionary: look up the token, get its stored vector.

My question: if the value embedding is good enough, is the value matrix needed at all?

## More

- [METHOD.md](METHOD.md): what I changed and what each stored vector learns.
- [WHERE-IT-WORKS.md](WHERE-IT-WORKS.md): where it helps, which layers need the value matrix, and what it predicts for large language models.
- [RESULTS.md](RESULTS.md): every number, seed and command. Code in [toy/](toy/), charts from [chart.py](chart.py).

## Is this new?

Mostly no: [MoME](https://arxiv.org/abs/2609.15126), [Bank of Values](https://arxiv.org/abs/2606.02780), [Memory Attention](https://arxiv.org/abs/2609.28399) and [nanochat](https://github.com/karpathy/nanochat/discussions/481) each do a part. Not found in published work: no value matrix in every layer, what each stored vector holds, and the map of where it helps.

Setup: a small transformer trained for 5–40 seconds on a MacBook CPU, on 13 small tasks such as next-word prediction, copying, facts, lookup and sorting.
