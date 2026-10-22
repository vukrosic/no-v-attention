# per-layer value-matrix map: layer_table.py TAG -> loss change vs base by task group, mean ± standard error over seeds
import json, collections, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "map4"
d = collections.defaultdict(dict)
for l in open("ablation_runs.jsonl"):
    r = json.loads(l)
    if r["tag"] == tag: d[r["arm"]][r["seed"]] = r["last"]
m = lambda x: sum(x) / len(x)
sk = list(d["base"][0])
G = {"all": sk, "context": ["recall", "rrecall", "bind", "chain"], "facts": ["factc", "factr", "rfact"], "copy": ["copy", "sort", "back2"]}
def diff(a, ks):
    ds = [m([d[a][s][k][0] for k in ks]) - m([d["base"][s][k][0] for k in ks]) for s in d[a] if s in d["base"]]
    se = (sum((x - m(ds)) ** 2 for x in ds) / max(len(ds) - 1, 1) / len(ds)) ** .5
    return m(ds), se, len(ds)
print("arm            " + "  ".join(f"{g:>13s}" for g in G) + "  seeds")
for a in sorted(d, key=lambda a: diff(a, sk)[0]):
    if a == "base": continue
    cells = [diff(a, G[g]) for g in G]
    print(f"{a:14s} " + "  ".join(f"{x:+.3f}±{e:.3f}" for x, e, n in cells) + f"  {cells[0][2]}")
