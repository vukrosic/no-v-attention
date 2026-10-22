# How the method works

## What I changed

1. **Gate:** depending on the sentence, the model can scale the value embedding up or down for each token (nanochat does this).
2. **Slots:** each token gets 4 rows in the value embedding instead of 1, and the layer input picks which row to use. The MoME paper calls these rows slots.
3. **No value matrix:** the value is now only the chosen value-embedding row.

Together they reach 0.996 against 1.084 for normal attention, better on all 4 seeds. When the 4 slots are averaged evenly instead of chosen, half the gain is gone, so the gain comes from choosing a row, not from having more weights. Spending the same weights on 3 more feed-forward experts gave nothing.

## What each slot learns

![Second layer: which slot each task uses](slots.png)

In the second layer, the task decides the slot. One slot holds the token itself, used by every task that copies a token. Another holds the token's common fact, and another its rare fact.

The first layer cannot do this. Its input is only the token, before any context is read, so its slot choice is fixed per token. Slots in the second layer alone keep almost all the gain.
