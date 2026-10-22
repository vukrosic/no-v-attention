"""print each run in sampled_runs.jsonl: tags, seed, wloss, wloss without the noisy count skill (nc), and per-skill loss"""
import json, sys
SK = ["chain", "copy", "factc", "back2", "recall", "sort", "factr", "count"]; W = [1 / (i + 1) for i in range(8)]
BASE = {"mode", "msign", "moe", "moe_k", "moe_f", "emb_mult", "head_mult", "emb_b1", "exp_mom", "emb_warm", "warmup", "attn_gate", "ve", "ve_mult", "canon", "canon_k", "canon_mult", "canon_pos", "canon_layers", "heads",
        "seed", "steps", "lr", "frac", "wloss", "chain_excess", "rare_mass", "rare_true", "last", "off"}
for l in open("sampled_runs.jsonl").readlines()[int(sys.argv[1]) if len(sys.argv) > 1 else 0:]:
    r = json.loads(l); L = r["last"]; nc = sum(W[i] * L[s][0] for i, s in enumerate(SK[:7])) / sum(W[:7])
    tag = " ".join(f"{k}={v}" for k, v in r.items() if k not in BASE) + (f" heads={r['heads']}" if r.get("heads") not in ("2", None) else "") + ("" if r.get("ve") else " PLAIN")
    print(f"{tag or 'base':42s} s{r['seed']} lr {r['lr']}  wloss {r['wloss']:.3f}  nc {nc:.3f}   " + " ".join(f"{s[:4]}:{L[s][0]:.2f}" for s in SK) + "".join(f"  {s}:{L[s][0]:.2f}/acc {L[s][1]:.2f}" for s in L if s not in SK))
