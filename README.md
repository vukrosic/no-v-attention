# Delete attention value matrices from your LLM (from some layers)

## Abstract

**Some layers of a language model do not need the attention value matrix.**

**A learned table can replace it.** Each layer has its own table. For every token in the vocabulary, the table stores 4 vectors, and the sentence picks which one to use. For example, "bank" could have one vector for a river bank and another for a money bank.

**The table is better at rare facts and at copying a word exactly**, such as repeating a name that appeared earlier in the text.

**Keep the value matrix in the later layers.** By then each token carries information gathered from the sentence, such as which words came before it. A fixed table cannot hold that.

- **Good at:** memorizing many rare facts, and copying the exact token.
- **Bad at:** information that comes from the context, which still needs the value matrix.
- **Little gain** when the model already has room to memorize everything.
- **Where to put it:** in the early layers after the first, with the value matrix kept in the later layers.
- **Safest version for quality:** keep the value matrix and add the stored vectors on top; the loss never got worse in my tests.
- **Cost:** each table holds vocabulary × 4 × width weights, tens of times more memory than the value matrix it replaces. Compute barely changes, because each token reads only one row of its table.

![Value embedding with context-chosen rows vs the value matrix](main_result.png)

## The core idea

Attention lets each token gather information from earlier tokens.

Each earlier token offers one vector, called its **value**.

The current token (the latest one) takes a weighted mix of these values. Earlier tokens it pays more attention to get more weight.

Normally the **value matrix** makes each value, from what the model knows about that earlier token in its sentence.

Recent models (modded-nanogpt, nanochat) also add a **value embedding** on top: a learned table with one stored vector per token, like a dictionary.

**In some layers, the value matrix can be replaced by a value embedding** with 4 stored vectors per token.

Each layer has its own table. The stored vectors are learned in training and do not depend on the sentence. The sentence only chooses which of the token's 4 vectors to use.

## More

- [METHOD.md](METHOD.md): what I changed and what each stored vector learns.
- [WHERE-IT-WORKS.md](WHERE-IT-WORKS.md): where it helps, which layers need the value matrix, and what it predicts for large language models.
- [RESULTS.md](RESULTS.md): every number, seed and command. Code in [toy/](toy/), charts from [chart.py](chart.py).

## Is this new?

Mostly no: [MoME](https://arxiv.org/abs/2609.15126), [Bank of Values](https://arxiv.org/abs/2606.02780), [Memory Attention](https://arxiv.org/abs/2609.28399) and [nanochat](https://github.com/karpathy/nanochat/discussions/481) each do a part. Not found in published work: no value matrix in every layer, what each stored vector holds, and the map of where it helps.

Setup: a small transformer trained for 5–40 seconds on a MacBook CPU, on 13 small tasks such as next-word prediction, copying, facts, lookup and sorting.
