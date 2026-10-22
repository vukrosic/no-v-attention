"""train one sampled.py model with VE_SLOTS > 1, then look at the slot picker: per layer and skill, how sharp the pick is,
and whether the picked slot is fixed per word or changes with context (same word, other skill or other position)
usage: env ... python3 sprobe.py SEED [STEPS]"""
import sys
from collections import Counter, defaultdict
import torch
import sampled, mix
SK = sampled.SKILLS


if __name__ == "__main__":
    seed = int(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    mix.run("whole", seed, steps, 0.028, every=steps, bs=32); m = mix.last_model
    vs = [(n, mod) for n, mod in m.named_modules() if n.endswith(".vs")]
    got = {}
    for n, mod in vs: mod.register_forward_hook(lambda mod, i, o, n=n: got.__setitem__(n, torch.softmax(o, -1)))
    ge = torch.Generator().manual_seed(999); recs = defaultdict(list)   # layer -> (word, skill, pos, argmax, maxw)
    with torch.no_grad():
        for sk in SK:
            s, _ = sampled.gen(sk, 256, ge); x = s[:, :-1]; m(x)
            for n, _ in vs:
                w = got[n]; a = w.argmax(-1); mx = w.max(-1).values
                for b in range(x.shape[0]):
                    for t in range(x.shape[1]):
                        if x[b, t] != sampled.PAD: recs[n].append((x[b, t].item(), sk, t, a[b, t].item(), mx[b, t].item()))
    for n, _ in vs:
        r = recs[n]; K = m.ve[list(m.ve.keys())[0]].weight.shape[1] // sampled.suite.D
        use = Counter(z[3] for z in r); share = [use[k] / len(r) for k in range(K)]
        byw = defaultdict(Counter); bywk = defaultdict(Counter)
        for wd, sk, t, a, _ in r: byw[wd][a] += 1; bywk[(wd, sk)][a] += 1
        fixw = sum(c.most_common(1)[0][1] for c in byw.values()) / len(r)
        fixwk = sum(c.most_common(1)[0][1] for c in bywk.values()) / len(r)
        print(f"{n}: slots used {' '.join(f'{p:.2f}' for p in share)}  mean top weight {sum(z[4] for z in r) / len(r):.2f}  "
              f"same slot as the word's usual {fixw:.2f}  same slot within word+skill {fixwk:.2f}")
        for sk in SK:
            rs = [z for z in r if z[1] == sk]; u = Counter(z[3] for z in rs)
            print(f"   {sk:8s} top weight {sum(z[4] for z in rs) / len(rs):.2f}  slots {' '.join(f'{u[k] / len(rs):.2f}' for k in range(K))}")
