"""Sampled softmax (modded-nanogpt record, Sep 2026) on a language-like toy with a real vocabulary.
Training scores each position only against a candidate list: every correct next word in its group of G sequences,
plus K shared random words. Eval always uses the full softmax over all words, like the record's validation.

toy: 1024 content words with Zipf frequency (word 0 most common, word 1023 rarest), 8 skills with Zipf weights
  chain   text from a fixed word chain, each word has 8 likely next words (common words likelier)  (word statistics)
  copy    a random segment repeats                                                              (induction)
  factc   64 subjects -> fixed object, objects are common words                                   (knowledge, common answers)
  back2   the token 2 back repeats
  recall  4 key-value pairs, then key -> value
  sort    6 words, separator, sorted
  factr   64 subjects -> fixed object, objects are rare words (512-1023)                          (knowledge, rare answers)
  count   6 tokens from {0..3}, how many 0s

usage: G=4 K=64 FULL0=0 FULL1=0 LOGQ=0 python3 sampled.py full|samp SEED [STEPS] [LR]    appends to sampled_runs.jsonl
"""
import json, os, sys, time
import torch, torch.nn.functional as F
import suite, mix

NV = int(os.environ.get("NV", 1024))
SEP, PAD, L = NV, NV + 1, 16
SKILLS = ["chain", "copy", "factc", "back2", "recall", "sort", "factr", "count"]
EXTRA = [k for k in os.environ.get("EXTRA", "").split(",") if k]   # extra skills where attention must move context, not a word: hop2, sum3
HOPK = int(os.environ.get("HOPK", 246))
TN = int(os.environ.get("TN", 3))   # trip: triples per sequence
BN, BA = int(os.environ.get("BN", 3)), int(os.environ.get("BA", 1))   # bind: items, answer words
SG, SD, SM = int(os.environ.get("SG", 3)), int(os.environ.get("SD", 3)), int(os.environ.get("SM", 10))   # sum3: groups, digits per group, digit range
RK, RN = int(os.environ.get("RK", 246)), int(os.environ.get("RN", 4))   # rrecall key pool size and pairs per sequence   # hop2 words come from words 10..10+HOPK
EXTRA_W = float(os.environ.get("EXTRA_W", 0.5))                    # raw mix weight of each extra skill (rank 2 = 0.5)
SKILLS = SKILLS + EXTRA
suite.V, suite.T = NV + 2 + len(SKILLS), L
W = torch.tensor([1 / (i + 1) if i < 8 else EXTRA_W for i in range(len(SKILLS))]); W /= W.sum()   # natural skill frequency (eval weights)
MIX = float(os.environ.get("MIX", 1))                                           # training mix: skill weight 1/rank^MIX (1 = natural, 0 = equal)
WT = torch.tensor([1 / (i + 1) ** MIX if i < 8 else EXTRA_W for i in range(len(SKILLS))]); WT /= WT.sum()
P = 1 / torch.arange(1, NV + 1).float(); P /= P.sum()       # word frequency (Zipf)
g0 = torch.Generator().manual_seed(11)
NEXT = torch.multinomial(P.expand(NV, NV), 8, generator=g0)  # each word's 8 possible next words, common words likelier
CHAIN = torch.zeros(NV, NV); CHAIN.scatter_(1, NEXT, (1 / torch.arange(1, 9).float()).expand(NV, 8)); CHAIN /= CHAIN.sum(1, keepdim=True)
NF = int(os.environ.get("NF", 64))                           # number of fact subjects (64 default)
SUBJ = torch.randperm(NV, generator=g0)[:NF]                 # the NF subject words
FACTZIPF = os.environ.get("FACTZIPF", "0") == "1"              # fact j asked with weight 1/(j+1) instead of all 64 equally
FZ = 1 / torch.arange(1, NF + 1).float()
FACTK = int(os.environ.get("FACTK", 512))                     # at most FACTK distinct answers per fact skill (fewer = facts share answers)
FACTC = torch.randint(64, (NF,), generator=g0) % FACTK       # common answers: words 0-63
FACTR = NV // 2 + torch.randint(NV // 2, (NF,), generator=g0) % FACTK  # rare answers: top half of the vocabulary (512-1023 at NV 1024)
FREL = int(os.environ.get("FREL", 4))                         # frel: relations per subject (each subject has FREL different answers)
FRT = NV // 2 + torch.randint(NV // 2, (FREL, NF), generator=g0)   # answer for (relation, subject)
if os.environ.get("FACTSWAP") == "1": FACTC, FACTR = FACTR, FACTC  # control: the frequent fact skill gets the rare answer words
pos = torch.arange(L)
CHAIN2 = os.environ.get("CHAIN2", "0") == "1"
OPT = os.environ.get("OPT", "whole")                          # whole = Muon on matrices; adam = AdamW on everything                  # chain text where the next word depends on the last two words (like phrases)


def chain_rows(s, t):
    """row of CHAIN that gives the next-word chances after position t of sequences s"""
    return (s[:, t - 1] * 31 + s[:, t]) % NV if CHAIN2 and t >= 2 else s[:, t]


def gen(skill, n, g):
    w = lambda *shape: torch.multinomial(P, n * shape[-1], replacement=True, generator=g).view(n, shape[-1])  # Zipf words
    s = torch.full((n, L + 1), PAD); s[:, 0] = NV + 2 + SKILLS.index(skill); m = torch.zeros(n, L, dtype=torch.bool)
    if skill == "chain":
        s[:, 1] = w(n, 1)[:, 0]
        for t in range(1, L): s[:, t + 1] = torch.multinomial(CHAIN[chain_rows(s, t)], 1, generator=g)[:, 0]
        m[:, 1:] = True
    elif skill == "copy":
        k = torch.randint(3, 8, (n, 1), generator=g); a = w(n, L); p = pos.expand(n, L)
        s[:, 1:] = a.gather(1, torch.where((p >= k) & (p < 2 * k), p - k, p)); m = (p > k) & (p < 2 * k)
    elif skill in ("factc", "factr"):
        j = torch.multinomial(FZ, n, replacement=True, generator=g) if FACTZIPF else torch.randint(NF, (n,), generator=g); s[:, 1], s[:, 2], s[:, 3] = SUBJ[j], SEP, (FACTC if skill == "factc" else FACTR)[j]; m[:, 2] = True
    elif skill == "back2":
        s[:, 1:] = w(n, 2).repeat(1, L // 2); m[:, 2:] = True
    elif skill == "recall":
        ks = torch.rand(n, 256, generator=g).argsort(1)[:, :4]; vs = w(n, 4); j = torch.randint(4, (n, 1), generator=g)
        s[:, 1:9:2], s[:, 2:9:2] = ks, vs; s[:, 9] = ks.gather(1, j)[:, 0]; s[:, 10] = vs.gather(1, j)[:, 0]; m[:, 9] = True
    elif skill == "sort":
        a = w(n, 6); s[:, 1:7], s[:, 7], s[:, 8:14] = a, SEP, a.sort(1).values; m[:, 7:13] = True
    elif skill == "hop2":    # two-step lookup: pairs [b c] [x y], then [a b] [z x]; query a -> c or z -> y (random); a one-step guess is right only half the time
        k = 10 + torch.rand(n, HOPK, generator=g).argsort(1)[:, :6]; a, b, c, z, x, y = k.T
        f1, f2, f3 = (torch.rand(n, generator=g) < 0.5 for _ in range(3))
        s[:, 1:5] = torch.stack([torch.where(f1, b, x), torch.where(f1, c, y), torch.where(f1, x, b), torch.where(f1, y, c)], 1)
        s[:, 5:9] = torch.stack([torch.where(f2, a, z), torch.where(f2, b, x), torch.where(f2, z, a), torch.where(f2, x, b)], 1)
        s[:, 9], s[:, 10] = torch.where(f3, a, z), torch.where(f3, c, y); m[:, 9] = True
    elif skill == "sum3":    # 3 groups [key d1 d2 d3], query key -> (d1+d2+d3) mod 10; digits are words 0-9
        k = 10 + torch.rand(n, 246, generator=g).argsort(1)[:, :SG]; d = torch.randint(SM, (n, SG, SD), generator=g); j = torch.randint(SG, (n,), generator=g)
        e = SG * (SD + 1) + 1; s[:, 1:e] = torch.cat([k[:, :, None], d], 2).view(n, e - 1); s[:, e] = k.gather(1, j[:, None])[:, 0]
        s[:, e + 1] = d[torch.arange(n), j].sum(1) % 10; m[:, e] = True
    elif skill == "rfact":   # look up then recall a fact: [key subject] x3, query key -> that subject's fact (common answer); guess without lookup = 1/3
        ks = 10 + torch.rand(n, 246, generator=g).argsort(1)[:, :3]; js = torch.rand(n, NF, generator=g).argsort(1)[:, :3]; q = torch.randint(3, (n, 1), generator=g)
        s[:, 1:7:2], s[:, 2:7:2] = ks, SUBJ[js]; s[:, 7] = ks.gather(1, q)[:, 0]; s[:, 8] = FACTC[js.gather(1, q)[:, 0]]; m[:, 7] = True
    elif skill == "agree":   # long-range agreement: subject, 3-10 filler words, SEP -> word 0 or 1 by the subject's class (half the subjects each); guess = 1/2
        j = torch.randint(NF, (n,), generator=g); f = torch.randint(3, 11, (n,), generator=g); a = w(n, L)
        s[:, 1] = SUBJ[j]; p = pos.expand(n, L)
        s[:, 2:] = torch.where(p[:, :-1] < f[:, None], a[:, :-1], PAD); s[torch.arange(n), f + 2] = SEP; s[torch.arange(n), f + 3] = j % 2; m[torch.arange(n), f + 2] = True
    elif skill == "rrecall": # reverse recall: [value key] x4, query key -> the word BEFORE it; the answer is not at the matched word, so layer 1 must first copy it there
        ks = 10 + torch.rand(n, RK, generator=g).argsort(1)[:, :RN]; vs = w(n, RN); j = torch.randint(RN, (n, 1), generator=g)
        s[:, 1:2 * RN + 1:2], s[:, 2:2 * RN + 1:2] = vs, ks; s[:, 2 * RN + 1] = ks.gather(1, j)[:, 0]; s[:, 2 * RN + 2] = vs.gather(1, j)[:, 0]; m[:, 2 * RN + 1] = True
    elif skill == "bind":    # attribute binding: [color size noun] x3, query noun -> one word naming its color+size (16 combos); colors 1-4, sizes 5-8 ("red small ball")
        nn_ = 30 + torch.rand(n, 226, generator=g).argsort(1)[:, :BN]; c = torch.randint(4, (n, BN), generator=g); z = torch.randint(4, (n, BN), generator=g); j = torch.randint(BN, (n, 1), generator=g)
        e = 3 * BN + 1; s[:, 1:e] = torch.stack([1 + c, 5 + z, nn_], 2).view(n, 3 * BN); s[:, e] = nn_.gather(1, j)[:, 0]
        if BA == 2: s[:, e + 1], s[:, e + 2] = 1 + c.gather(1, j)[:, 0], 5 + z.gather(1, j)[:, 0]; m[:, e:e + 2] = True   # answer in two words: color, then size
        else: s[:, e + 1] = 40 + 4 * c.gather(1, j)[:, 0] + z.gather(1, j)[:, 0]; m[:, e] = True
    elif skill == "trip":    # [a key b] x TN, query key -> a (the word before the key); matching the key's neighbour b avoids the query matching itself, but a must be carried over to b's spot first
        ks = 10 + torch.rand(n, RK, generator=g).argsort(1)[:, :TN]; aa, bb = w(n, TN), w(n, TN); j = torch.randint(TN, (n, 1), generator=g)
        s[:, 1:3 * TN + 1] = torch.stack([aa, ks, bb], 2).view(n, 3 * TN); e = 3 * TN + 1; s[:, e] = ks.gather(1, j)[:, 0]; s[:, e + 1] = aa.gather(1, j)[:, 0]; m[:, e] = True
    elif skill == "prev":    # [value key] x RN, query = the key's partner word (key + 120, so the query never matches itself) -> the value before that key
        ks = 10 + torch.rand(n, 120, generator=g).argsort(1)[:, :RN]; vs = w(n, RN); j = torch.randint(RN, (n, 1), generator=g)
        s[:, 1:2 * RN + 1:2], s[:, 2:2 * RN + 1:2] = vs, ks; s[:, 2 * RN + 1] = ks.gather(1, j)[:, 0] + 120; s[:, 2 * RN + 2] = vs.gather(1, j)[:, 0]; m[:, 2 * RN + 1] = True
    elif skill == "frel":    # [relation subject SEP] -> that subject's answer for that relation; relations are words 1..FREL, so one subject word has FREL meanings
        r = torch.randint(FREL, (n,), generator=g); j = torch.randint(NF, (n,), generator=g)
        s[:, 1], s[:, 2], s[:, 3], s[:, 4] = 1 + r, SUBJ[j], SEP, FRT[r, j]; m[:, 3] = True
    elif skill == "count":
        a = torch.randint(4, (n, 6), generator=g); s[:, 1:7], s[:, 7], s[:, 8] = a, SEP, (a == 0).sum(1); m[:, 7] = True
    return s, m


def batch(n, g):
    counts = torch.multinomial(WT, n, replacement=True, generator=g).bincount(minlength=len(SKILLS))
    parts = [gen(sk, c.item(), g) for sk, c in zip(SKILLS, counts) if c > 0]
    s = torch.cat([p[0] for p in parts]); m = torch.cat([p[1] for p in parts])
    perm = torch.randperm(len(s), generator=g); s, m = s[perm], m[perm]   # groups mix skills, like chunks of real text
    return s[:, :-1], s[:, 1:], m


G, K = int(os.environ.get("G", 4)), int(os.environ.get("K", 64))
FULL0, FULL1 = int(os.environ.get("FULL0", 0)), int(os.environ.get("FULL1", 0))
LOGQ, NEG = os.environ.get("LOGQ", "0") == "1", os.environ.get("NEG", "uniform")
seen = torch.ones(suite.V)                                       # running count of each word as a target (for LOGQ)
frac = []


def sampled_loss(model, x, y, msk, i, steps):
    logits = model(x)
    if i < FULL0 or i >= steps - FULL1: return F.cross_entropy(logits[msk], y[msk])
    b, V = x.shape[0], logits.shape[-1]; grp = torch.arange(b) // G
    inc = torch.zeros(grp.max() + 1, V, dtype=torch.bool)
    inc[grp[:, None].expand_as(y)[msk], y[msk]] = True             # every correct word in the group is a candidate
    if K:
        neg = torch.randint(NV, (K,)) if NEG == "uniform" else torch.multinomial(P, K, replacement=True)
        inc[:, neg] = True                                          # shared random words
    frac.append(inc.float().sum(1).mean().item() / V)
    lg = logits.masked_fill(~inc[grp][:, None, :], float("-inf"))
    if LOGQ:                                                        # subtract log(chance the word is in the list)
        seen.index_add_(0, y[msk], torch.ones(msk.sum())); p = seen / seen.sum()
        n_t = msk.sum().item() / (grp.max().item() + 1); q = 1 - (1 - p) ** n_t
        qn = torch.zeros(V); qn[:NV] = 1 - (1 - 1 / NV) ** K if NEG == "uniform" else 1 - (1 - P) ** K
        q = q + qn * (1 - q)
        lg = lg - q.clamp_min(1e-6).log()
    return F.cross_entropy(lg[msk], y[msk])


mix.SKILLS, mix.W, mix.gen, mix.batch = SKILLS, W, gen, batch


def drop_loss(model, x, y, msk, i, steps):
    """cross-entropy over tokens the model has not solved yet: tokens with target probability > DROP_P are left out"""
    l = F.cross_entropy(model(x)[msk], y[msk], reduction="none")
    keep = (l.detach() < -torch.log(torch.tensor(float(os.environ["DROP_P"])))).logical_not()
    return (l * keep).sum() / keep.sum().clamp(min=1)


@torch.no_grad()
def extra(model):
    """full-softmax diagnostics: chain loss above its floor, and how much probability the model puts on rare words
    (512-1023) where the true next word is common vs the true chance"""
    s, m = gen("chain", 512, torch.Generator().manual_seed(999)); x, y = s[:, :-1], s[:, 1:]
    pr = model(x).softmax(-1)[m][:, :NV]; true = CHAIN[torch.stack([chain_rows(s, t) for t in range(L)], 1)[m]]
    floor = -(true * true.clamp_min(1e-12).log()).sum(1).mean().item()
    loss = -pr.gather(1, y[m][:, None]).clamp_min(1e-12).log().mean().item()
    return dict(chain_excess=round(loss - floor, 3), rare_mass=round(pr[:, 512:].sum(1).mean().item(), 4), rare_true=round(true[:, 512:].sum(1).mean().item(), 4))


if __name__ == "__main__":
    mode, seed = sys.argv[1], int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300; lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    mix.LOSS = sampled_loss if mode == "samp" else drop_loss if "DROP_P" in os.environ else None
    if suite.HBIAS in ("log", "loginit"):        # target counts from 200 natural-mix batches of a separate generator, add-one smoothed
        gc, cnt = torch.Generator().manual_seed(777), torch.ones(suite.V)
        for _ in range(200): _, yb, mb = batch(32, gc); cnt += torch.bincount(yb[mb], minlength=suite.V)[: suite.V]
        lp = (cnt / cnt.sum()).log(); suite.HB_INIT = lp - lp.mean()
    t0 = time.time(); rows = mix.run(OPT, seed, steps, lr, every=steps, bs=int(os.environ.get("BS", 32)))
    last = rows[-1][1]; wl = sum(w.item() * last[sk][0] for w, sk in zip(W, SKILLS)); ex = extra(mix.last_model)
    cfg = dict(mode=mode, G=G, K=K, FULL0=FULL0, FULL1=FULL1, LOGQ=int(LOGQ), NEG=NEG) if mode == "samp" else dict(mode=mode)
    if suite.ENGRAM: cfg.update(engram=os.environ["ENGRAM"], eg_rows=suite.EG_ROWS, eg_gate=int(suite.EG_GATE), tag="1gram" if suite.EG_1GRAM else "")
    if CHAIN2: cfg["chain2"] = 1
    if MIX != 1: cfg["mix"] = MIX
    if OPT != "whole": cfg["opt"] = OPT
    if os.environ.get("FACTSWAP") == "1": cfg["factswap"] = 1
    if suite.MSIGN != "svd": cfg["msign"] = suite.MSIGN
    if suite.MOE: cfg.update(moe=suite.MOE, moe_k=suite.MOE_K, moe_f=suite.MOE_F)
    if mix.QK_MAN != "none": cfg["qk_man"] = mix.QK_MAN
    if mix.VO_MAN != "none": cfg["vo_man"] = mix.VO_MAN
    if mix.QK_SCALE != 1: cfg["qk_scale"] = mix.QK_SCALE
    if mix.VO_SCALE != 1: cfg["vo_scale"] = mix.VO_SCALE
    if mix.QK_MULT != 1: cfg["qk_mult"] = mix.QK_MULT
    if suite.EMB_STD != 1: cfg["emb_std"] = suite.EMB_STD
    if suite.D != 64: cfg["width"] = suite.D
    if suite.HEAD_STD != 1: cfg["head_std"] = suite.HEAD_STD
    for k in ("VO_MULT", "MLP_MULT", "ROUTER_MULT", "EMB_MULT", "HEAD_MULT", "NORM_MULT"):
        if getattr(mix, k) != 1: cfg[k.lower()] = getattr(mix, k)
    if mix.MLP_MAN != "none": cfg["mlp_man"] = mix.MLP_MAN
    if mix.MLP_SCALE != 1: cfg["mlp_scale"] = mix.MLP_SCALE
    if mix.MAN_SCALE != 1: cfg["man_scale"] = mix.MAN_SCALE
    for k in ("ATTN_SCALE", "EMB_NORM", "ZERO_OUT", "HASH0", "EMB_B1", "EXP_MOM", "FREQ_LR", "EMB_WARM", "EMB_DECAY", "DROP_P", "SEED_TAG", "HEAD_B1", "REST_B1", "ATT_MOM", "WARMUP", "HEAD_MUON", "HEAD_ROW", "ATTN_GATE", "VE", "SHARED", "MOE_K", "VE_SHARE", "VE_LAYERS", "EVAL_SEED", "VE_MULT", "VE_B1", "VE_STD", "KE", "QE", "VE_GATE", "VE_SLOTS", "VS_FIX", "VS_L", "NF", "FREL", "RE", "GE", "BIGRAM", "ENGRAM", "VRES", "CANON", "X0", "CANON_K", "CANON_STD", "CANON_MULT", "CANON_POS", "CANON_LAYERS", "KANNEAL", "ROUTE_PREV", "ROUTE_X0", "GF_L", "GF_A", "GF_ONLY", "LAZY", "MUON_ON", "ADAM_MAT_LR", "HBIAS", "HB_MULT", "LAYER_MULT", "HEADS", "NV", "ADAM_B2", "CON_HEADS", "HX", "HX_ON", "HLR", "HMOM", "HX_SCALE", "HX_K", "MAN_EVERY", "SEQ_OV", "QKNORM", "KV_HEADS", "KV_MULT", "FUSE", "FUSE_SCALE", "FREEZE", "V_ZERO", "V_ZERO_L", "AG_MULT", "V_MULT", "AG_RANK", "EXTRA", "EXTRA_W", "HOPK", "RK", "RN", "TN", "BN", "BA", "SG", "SD", "SM"):
        if k in os.environ: cfg[k.lower()] = os.environ[k]
    off = {}
    if suite.ENGRAM:                                                # same model with the lookup switched off: which skills live in it
        m = mix.last_model; m.eg_off = True
        with torch.no_grad():
            for sk in SKILLS:
                s_, m_ = gen(sk, 128, torch.Generator().manual_seed(999)); o = m(s_[:, :-1]); y_ = s_[:, 1:]
                off[sk] = (round(F.cross_entropy(o[m_], y_[m_]).item(), 3), round((o.argmax(-1)[m_] == y_[m_]).float().mean().item(), 2))
        m.eg_off = False
    fr = sum(frac) / len(frac) if frac else 1.0
    print(f"{' '.join(f'{k}={v}' for k, v in cfg.items())} seed {seed} steps {steps} lr {lr} {time.time() - t0:.1f}s  list {fr:.3f} of vocab  "
          f"wloss {wl:.3f}  chain_excess {ex['chain_excess']}  rare_mass {ex['rare_mass']} (true {ex['rare_true']})")
    print("  " + " ".join(f"{sk}:{last[sk][0]:.2f}/{last[sk][1]:.2f}" for sk in SKILLS))
    if off: print("  lookup off: " + " ".join(f"{sk}:{off[sk][0]:.2f}/{off[sk][1]:.2f}" for sk in SKILLS))
    with open("sampled_runs.jsonl", "a") as f:
        f.write(json.dumps(dict(cfg, seed=seed, steps=steps, lr=lr, frac=fr, wloss=wl, **ex, last=last, off=off)) + "\n")
