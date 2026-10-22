"""train one sampled.py model (env knobs as usual), then zero the value matrix or the word table (value embedding) of one layer
after training and print how much each skill's loss rises: which skills the trained model routes through v vs the table
usage: env ... python3 vprobe.py SEED [STEPS] [LR]"""
import os, sys, copy
import torch, torch.nn.functional as F
import sampled, mix
SK = sampled.SKILLS


@torch.no_grad()
def score(m):
    ge = torch.Generator().manual_seed(999); L = {}
    for sk in SK:
        s, msk = sampled.gen(sk, 256, ge); L[sk] = F.cross_entropy(m(s[:, :-1])[msk], s[:, 1:][msk]).item()
    return L


@torch.no_grad()
def cut(m, what, j):
    m2 = copy.deepcopy(m)
    if what == "v": m2.layers[j].v.weight.zero_()
    else: m2.ve[str(j)].weight.zero_()
    return m2


if __name__ == "__main__":
    seed = int(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400; lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.028
    mix.run("whole", seed, steps, lr, every=steps, bs=32); m = mix.last_model; base = score(m)
    print("skill    base  " + "  ".join(f"{w}{j + 1}-off" for w in ("v", "table") for j in (0, 1)))
    rows = {f"{w}{j}": score(cut(m, w, j)) for w in ("v", "table") for j in (0, 1) if w == "v" or hasattr(m, "ve")}
    for sk in SK: print(f"{sk:8s} {base[sk]:5.2f} " + "  ".join(f"{rows[k][sk] - base[sk]:+7.2f}" for k in rows))
