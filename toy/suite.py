"""Toy suite: six few-second tasks that each copy one thing a large LM does, trained with Muon where Q,K use
whole-matrix Muon, per-head Muon, or the lopsidedness dial (per-head directions, singular values ^ a, same size).

tasks
  back    predict the token 4 places back        (fixed-position head, like attending to the previous word)
  copy    a random segment repeats; continue it  (induction / in-context copying)
  recall  8 key-value pairs, then a key -> value (retrieval from context)
  add     a b = (a+b) mod 31                     (arithmetic, needs the MLP)
  sort    10 tokens, separator, sorted tokens    (algorithm over the whole context)
  markov  text from a fixed random Zipf chain    (language statistics, loss has a floor)

Model (modern by default, like current LMs): 2 layers, width 64, 4 heads, pre-norm RMSNorm, RoPE, QK-norm,
SwiGLU MLP, no biases. Muon on q,k,v,o and MLP matrices, AdamW on the rest.
Old toy: ROPE=0 (learned positions) MLP=gelu QKNORM=0.
usage: python3 suite.py TASK MODE [seed] [steps]     MODE = whole | perhead | pow:A
prints the eval loss every 10 steps as one json line (also appended to runs.jsonl)
"""
import json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
KE, VRES, CANON, X0 = (os.environ.get(k, "0") == "1" for k in ("KE", "VRES", "CANON", "X0"))
QE = os.environ.get("QE", "0") == "1"
VE_GATE = os.environ.get("VE_GATE", "0") == "1"   # nanochat: value table scaled per head by 2*sigmoid(W h), W zero init
VE_SLOTS = int(os.environ.get("VE_SLOTS", 1))       # MoME-style: each word has VE_SLOTS table rows, mixed by softmax(W h) of the layer input
VS_L = os.environ.get("VS_L", "01")                  # which layers get the slots (others: one row per word)
RE, GE, BIGRAM = (os.environ.get(k, "0") == "1" for k in ("RE", "GE", "BIGRAM"))   # zero-init word tables: RE added to the residual before each layer, GE added to the attention-gate logits, BIGRAM adds V logits per current word   # query table: each layer adds its own word table to q  # key table, value residual, causal conv, embedding shortcut
CK = int(os.environ.get("CANON_K", 4))
KANNEAL = int(os.environ.get("KANNEAL", 0))   # top-k goes linearly from all experts to MOE_K over this many training steps
ROUTE_PREV, ROUTE_X0 = os.environ.get("ROUTE_PREV", "0") == "1", os.environ.get("ROUTE_X0", "0") == "1"
ATTN_SCALE = float(os.environ.get("ATTN_SCALE", 1))   # attention logits x this (after QK-norm the logits are small)
V, D, T = 64, int(os.environ.get("WIDTH", 64)), 32   # WIDTH: model width (default 64); MLP and expert sizes scale with it
NL = int(os.environ.get("LAYERS", 2))                 # LAYERS: number of transformer layers (default 2)
H = int(os.environ.get("HEADS", "4")); KVH = int(os.environ.get("KV_HEADS", 0)) or H   # KV_HEADS < HEADS: grouped-query attention (k,v heads shared)  # attention heads; width stays 64, so more heads = smaller heads
MULT = {k: float(v) for k, v in (kv.split("=") for kv in os.environ.get("MULT", "").split(",") if kv)}  # lr multiplier per matrix kind, e.g. MULT=q=2,k=2
ROPE = os.environ.get("ROPE", "1") == "1"  # RoPE on q,k (default); ROPE=0 = learned position table (older numbers)
SWIGLU = os.environ.get("MLP", "swiglu") == "swiglu"  # SwiGLU MLP (default); MLP=gelu = older numbers
MOE, MOE_K = int(os.environ.get("MOE", 0)), int(os.environ.get("MOE_K", 2))  # mixture of experts: MOE experts (0 = dense MLP), top-MOE_K per token
MOE_F = int(os.environ.get("MOE_F", 176 * D // 64 // max(1, MOE_K)))  # expert hidden size; default keeps the active MLP size of the dense model
MATS = ("q", "k", "v", "o", "gate", "up", "down") if SWIGLU else ("q", "k", "v", "o", "up", "down")  # Muon matrices per layer
ATTN_GATE = os.environ.get("ATTN_GATE", "0") == "1"   # sigmoid gate on the attention output, from the block input (Qwen gated attention)
VE = os.environ.get("VE", "0") == "1"                 # value embeddings: each layer adds its own token table to v
VE_SHARE = os.environ.get("VE_SHARE", "0") == "1"     # control: v gets the input embedding itself (no new parameters)
VE_LAYERS = [int(i) for i in os.environ.get("VE_LAYERS", ",".join(map(str, range(NL)))).split(",")]   # which layers get value embeddings
SHARED = os.environ.get("SHARED", "0") == "1"         # MoE: one extra expert every token always uses (DeepSeek shared expert)
AG_RANK = int(os.environ.get("AG_RANK", 0))
if ATTN_GATE: MATS = MATS + ("ag",)


def rope(x, base=10000):
    s, d = x.shape[-2], x.shape[-1]
    th = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
    ang = torch.outer(torch.arange(s).float(), th); c, sn = ang.cos(), ang.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * c - b * sn, b * c + a * sn], -1).flatten(-2)


SCHED = os.environ.get("SCHED", "1") == "1"  # warmup 30 steps, linear decay to 0 over the second half (default); SCHED=0 = constant lr (older numbers)


WARMUP = int(os.environ.get("WARMUP", 30))  # warmup steps of the shared schedule


def lr_at(i, steps):
    return 1.0 if not SCHED else min(1.0, (i + 1) / WARMUP) * min(1.0, 2 * (1 - i / steps))


QKN = os.environ.get("QKNORM", "1") == "1"  # RMSNorm on each head's q and k, as in current LMs (default on; QKNORM=0 for the older numbers)


class Layer(nn.Module):
    def __init__(s):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(D, D if n_ in "qo" else KVH * (D // H), bias=False) for n_ in "qkvo")   # same init order as before GQA
        if ATTN_GATE: s.ag = nn.Linear(D, AG_RANK or D, bias=False)
        if ATTN_GATE and AG_RANK: s.ag2 = nn.Linear(AG_RANK, D, bias=False)   # low-rank gate: D -> AG_RANK -> D (ag2 on AdamW)
        s.ids = s.vex = s.kex = s.qex = s.gex = s.vprev = s.vout = None
        if CANON:
            s.cv = nn.Conv1d(D, D, CK, groups=D, bias=False); nn.init.normal_(s.cv.weight, std=float(os.environ.get("CANON_STD", 0.1)))
            s.cv2 = nn.Conv1d(D, D, CK, groups=D, bias=False); nn.init.normal_(s.cv2.weight, std=float(os.environ.get("CANON_STD", 0.1)))
            s.cpos = ""
        F_ = 176 * D // 64 if SWIGLU else 4 * D  # SwiGLU hidden ~8/3 D keeps the MLP size of a 4D GELU MLP
        if MOE: s.moe = MoE(); s.gate, s.up, s.down = s.moe.gate, s.moe.up, s.moe.down
        else:
            if SWIGLU: s.gate = nn.Linear(D, F_, bias=False)
            s.up, s.down = nn.Linear(D, F_, bias=False), nn.Linear(F_, D, bias=False)
        s.n1, s.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        if VE_GATE: s.vg = nn.Linear(D, H, bias=False); nn.init.zeros_(s.vg.weight)
        s.ns = VE_SLOTS
        if VE_SLOTS > 1: s.vs = nn.Linear(D, VE_SLOTS, bias=False); nn.init.zeros_(s.vs.weight); s.vs.weight.requires_grad_(os.environ.get("VS_FIX", "0") != "1")   # VS_FIX: slots mixed evenly (control: extra rows, no context choice)
        s.qn, s.kn = (nn.RMSNorm(D // H), nn.RMSNorm(D // H)) if QKN else (nn.Identity(), nn.Identity())

    def forward(s, x):
        b, t, _ = x.shape
        h = s.n1(x)
        hc = h + s.cv(F.pad(h.transpose(1, 2), (CK - 1, 0))).transpose(1, 2) if CANON and s.cpos else h   # causal depthwise conv over the last CK tokens
        if CANON and "a" in s.cpos: h = hc
        q, k, v = (m(hc if CANON and p_ in s.cpos else h).view(b, t, -1, D // H).transpose(1, 2) for m, p_ in ((s.q, "q"), (s.k, "k"), (s.v, "v")))
        if s.qex is not None: q = q + s.qex.view(b, t, H, D // H).transpose(1, 2)
        if s.kex is not None: k = k + s.kex.view(b, t, H, D // H).transpose(1, 2)
        q, k = s.qn(q), s.kn(k)
        if ROPE: q, k = rope(q), rope(k)
        vex = s.vex
        if vex is not None and s.ns > 1: vex = (vex.view(b, t, s.ns, D) * torch.softmax(s.vs(h), -1)[..., None]).sum(2) * s.ns ** 0.5
        if vex is not None and VE_GATE: vex = (vex.view(b, t, H, D // H) * 2 * torch.sigmoid(s.vg(h))[..., None]).view(b, t, D)
        if s.vex is not None: v = v + vex[..., : KVH * (D // H)].view(b, t, KVH, D // H).transpose(1, 2)
        if s.vprev is not None: v = v + s.vprev
        s.vout = v
        if KVH != H: k, v = k.repeat_interleave(H // KVH, 1), v.repeat_interleave(H // KVH, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=ATTN_SCALE / (D // H) ** 0.5).transpose(1, 2).reshape(b, t, D)
        if ATTN_GATE: a = a * torch.sigmoid((s.ag2(s.ag(h)) if AG_RANK else s.ag(h)) + (0 if s.gex is None else s.gex))
        x = x + s.o(a)
        h = s.n2(x)
        if CANON and "c" in s.cpos: h = h + s.cv2(F.pad(h.transpose(1, 2), (CK - 1, 0))).transpose(1, 2)   # same, before the MLP
        if MOE: return x + s.moe(h)
        return x + s.down(F.silu(s.gate(h)) * s.up(h) if SWIGLU else F.gelu(s.up(h)))


class Experts(nn.Module):
    """MOE weight matrices of shape out x inp stacked in one E x out x inp tensor (Muon orthogonalizes each expert's slice)"""
    def __init__(s, out, inp):
        super().__init__(); s.weight = nn.Parameter(torch.empty(MOE + SHARED, out, inp).uniform_(-inp ** -0.5, inp ** -0.5))


class MoE(nn.Module):
    """SwiGLU experts, softmax router (AdamW), top-k, gates renormalized over the chosen experts (Mixtral style).
    Load balance without an aux loss (DeepSeek-V3): a bias added to the router scores only for choosing experts, nudged up for
    under-used experts and down for over-used ones after every training forward. Each expert runs only on its own tokens."""
    def __init__(s):
        super().__init__()
        s.router = nn.Linear(D, MOE, bias=False)
        s.gate, s.up, s.down = Experts(MOE_F, D), Experts(MOE_F, D), Experts(D, MOE_F)
        s.register_buffer("bias", torch.zeros(MOE)); s.load = torch.zeros(MOE); s.hashed = False
        s.calls, s.x0 = 0, None
        if ROUTE_PREV: s.router2 = nn.Linear(D, MOE, bias=False)

    def forward(s, h):
        x = h.reshape(-1, D); r_in = s.x0.reshape(-1, D) if ROUTE_X0 else x
        logits = s.router(r_in)
        if ROUTE_PREV: logits = logits + s.router2(F.pad(h, (0, 0, 1, 0))[:, :-1].reshape(-1, D))   # previous token's hidden state
        p = logits.softmax(-1)
        k = MOE_K
        if KANNEAL and torch.is_grad_enabled(): s.calls += 1; k = max(MOE_K, round(MOE - (MOE - MOE_K) * s.calls / KANNEAL))
        top = (p.detach() + s.bias).topk(k, -1).indices
        g = torch.zeros_like(p).scatter(1, top, p.gather(1, top)); g = g / g.sum(-1, keepdim=True)
        if s.hashed:                                                             # fixed per-token experts (Hash Layers), equal gates
            a = s.ids % MOE; top = torch.stack([a, (a + 1 + (s.ids * 7 + 3) % (MOE - 1)) % MOE], -1)[:, :MOE_K]
            g = torch.zeros_like(p).scatter(1, top, 1.0 / MOE_K)
        if torch.is_grad_enabled() and not s.hashed:
            load = (g > 0).float().mean(0) * MOE / k; s.load = load      # 1 = fair share
            s.bias += 0.01 * torch.sign(1 - load)
        out = torch.zeros_like(x)
        for e in range(MOE):                                                     # each expert only on the tokens routed to it
            i = (top == e).any(-1).nonzero()[:, 0]
            if len(i) == 0: continue
            xe = x[i]; a = F.silu(xe @ s.gate.weight[e].T) * (xe @ s.up.weight[e].T)
            out = out.index_add(0, i, (a @ s.down.weight[e].T) * g[i, e, None])
        if SHARED: out = out + (F.silu(x @ s.gate.weight[MOE].T) * (x @ s.up.weight[MOE].T)) @ s.down.weight[MOE].T
        return out.view_as(h)


ENGRAM = [int(i) for i in os.environ.get("ENGRAM", "").split(",") if i]  # Engram (DeepSeek, arXiv 2601.07372) before these layers, "" = off; paper's best single spot = layer 2 = index 1
EG_ROWS, EG_HEADS, EG_DIM = int(os.environ.get("EG_ROWS", 4099)), int(os.environ.get("EG_HEADS", 2)), int(os.environ.get("EG_DIM", 16))
EG_GATE = os.environ.get("EG_GATE", "1") == "1"
EG_1GRAM = os.environ.get("EG_1GRAM", "0") == "1"   # control: every head hashes only the current token (same size, no n-gram memory)


class Engram(nn.Module):
    """hashed memory of the last 2 and 3 tokens: EG_HEADS hash heads per n, each a table of EG_ROWS rows (prime);
    the looked-up vectors are concatenated, projected to key and value, gated by sigmoid(norm(h) . norm(key) / sqrt(D)),
    then a short causal depthwise conv (kernel 4, dilation 3, zero init) is added on top. Tables get 5x Adam lr."""
    def __init__(s, seed):
        super().__init__()
        s.tab = nn.Embedding(2 * EG_HEADS * EG_ROWS, EG_DIM); s.tab.weight.lr_mult = 5   # all hash heads' tables in one
        mul = torch.randint(1, 2 ** 20, (2 * EG_HEADS, 3), generator=torch.Generator().manual_seed(seed)) * 2 + 1
        mul[:EG_HEADS, 0] = 0                             # first EG_HEADS heads: 2-grams (oldest token ignored), the rest: 3-grams
        if EG_1GRAM: mul[:, :2] = 0
        s.register_buffer("mul", mul[:, :, None, None]); s.register_buffer("off", (torch.arange(2 * EG_HEADS) * EG_ROWS)[:, None, None])
        s.wk, s.wv = (nn.Linear(2 * EG_HEADS * EG_DIM, D, bias=False) for _ in range(2))
        s.hn, s.kn, s.vn = nn.RMSNorm(D), nn.RMSNorm(D), nn.RMSNorm(D)
        s.conv = nn.Parameter(torch.zeros(4, D))           # depthwise causal conv, kernel 4, dilation 3, zero init (as 4 shifted adds: faster on CPU)

    def forward(s, x, h):
        p = F.pad(x, (2, 0), value=V)                    # id V = before the start
        z = (p[:, :-2] * s.mul[:, 0]) ^ (p[:, 1:-1] * s.mul[:, 1]) ^ (p[:, 2:] * s.mul[:, 2])   # multiplicative-XOR hash, heads x b x t
        es = s.tab(z % EG_ROWS + s.off)                   # heads x b x t x EG_DIM
        e = es.permute(1, 2, 0, 3).flatten(2); k, v = s.wk(e), s.wv(e)
        if EG_GATE: v = torch.sigmoid((s.hn(h) * s.kn(k)).sum(-1, keepdim=True) / D ** 0.5) * v
        u = F.pad(s.vn(v), (0, 0, 9, 0)); t = x.shape[1]
        return F.silu(sum(s.conv[j] * u[:, 9 - 3 * j:9 - 3 * j + t] for j in range(4))) + v


EMB_STD, HEAD_STD = float(os.environ.get("EMB_STD", 1)), float(os.environ.get("HEAD_STD", 1))
EMB_NORM = os.environ.get("EMB_NORM", "0") == "1"     # RMSNorm right after the embedding lookup
ZERO_OUT = os.environ.get("ZERO_OUT", "0") == "1"     # attention out (o) and MLP/expert down start at zero
HBIAS = os.environ.get("HBIAS", "")  # output-logit bias: learn = trained vector (zero init), log = fixed log unigram frequency of targets, loginit = trained, starts at log frequency
HB_INIT = None                         # log-frequency vector, set by the caller for log / loginit
HASH0 = os.environ.get("HASH0", "0") == "1"           # layer 0's MoE picks experts from the token id (fixed hash), equal gates


class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, D)
        if not ROPE: s.pos = nn.Parameter(torch.randn(T, D) * 0.1)
        s.layers = nn.ModuleList([Layer() for _ in range(NL)])
        s.norm, s.head = nn.RMSNorm(D), nn.Linear(D, V, bias=False)
        s.eg = nn.ModuleDict({str(i): Engram(i) for i in ENGRAM}); s.eg_off = False
        with torch.no_grad():
            s.emb.weight.mul_(EMB_STD); s.head.weight.mul_(HEAD_STD)
            if ZERO_OUT:
                for l in s.layers: l.o.weight.zero_(); l.down.weight.zero_()
        s.en = nn.RMSNorm(D) if EMB_NORM else nn.Identity()
        if VE:
            for j, l in enumerate(s.layers):
                if str(j) not in VS_L and l.ns > 1: l.ns = 1; del l.vs
            s.ve = nn.ModuleDict({str(j): nn.Embedding(V, D * s.layers[j].ns) for j in VE_LAYERS})
            with torch.no_grad(): [e.weight.mul_(float(os.environ.get("VE_STD", 1))) for e in s.ve.values()]
        for nm_, on, w_ in (("re", RE, D), ("ge", GE, D)):
            if on: setattr(s, nm_, nn.ModuleDict({str(j): nn.Embedding(V, w_) for j in range(len(s.layers))})); [nn.init.zeros_(e.weight) for e in getattr(s, nm_).values()]
        if BIGRAM: s.bg = nn.Embedding(V, V); nn.init.zeros_(s.bg.weight)
        if QE: s.qe = nn.ModuleDict({str(j): nn.Embedding(V, D) for j in range(len(s.layers))})
        if KE: s.ke = nn.ModuleDict({str(j): nn.Embedding(V, D) for j in range(len(s.layers))})
        if X0: s.x0g = nn.Parameter(torch.ones(len(s.layers)))
        if CANON:
            for j, l in enumerate(s.layers): l.cpos = os.environ.get("CANON_POS", "a") if str(j) in os.environ.get("CANON_LAYERS", "01") else ""
        if HBIAS == "log": s.register_buffer("hb", HB_INIT.clone())
        elif HBIAS: s.hb = nn.Parameter(HB_INIT.clone() if HBIAS == "loginit" else torch.zeros(V))
        if HASH0 and MOE: s.layers[0].moe.hashed = True   # init scale x default (emb N(0,1), head U(+-1/sqrt(D)))   # eg_off: skip the lookup (ablation at eval)

    def forward(s, x):
        h = s.en(s.emb(x)) + (0 if ROPE else s.pos[: x.shape[1]])
        if HASH0 and MOE: s.layers[0].moe.ids = x.reshape(-1)
        e = s.emb(x) if VE_SHARE else None
        for j, l in enumerate(s.layers):
            l.vex = (e if VE_SHARE else s.ve[str(j)](x)) if (VE or VE_SHARE) and j in VE_LAYERS else None
        x0 = h
        if ROUTE_X0 and MOE:
            for l in s.layers: l.moe.x0 = s.emb(x)
        for i, l in enumerate(s.layers):
            if str(i) in s.eg and not s.eg_off: h = h + s.eg[str(i)](x, h)
            if KE: l.kex = s.ke[str(i)](x)
            if QE: l.qex = s.qe[str(i)](x)
            if GE: l.gex = s.ge[str(i)](x)
            if RE: h = h + s.re[str(i)](x)
            if VRES and i: l.vprev = s.layers[i - 1].vout
            if X0 and i: h = h + s.x0g[i] * x0
            h = l(h)
        o = s.head(s.norm(h))
        if BIGRAM: o = o + s.bg(x)
        return o + s.hb if HBIAS else o


g0 = torch.Generator().manual_seed(123)
CHAIN = torch.zeros(V, V)                         # markov: each token has 6 likely next tokens, Zipf weights
for i in range(V): CHAIN[i, torch.randperm(V, generator=g0)[:6]] = 1 / torch.arange(1, 7).float()
CHAIN /= CHAIN.sum(1, keepdim=True)


def batch(task, n, g):
    """input x, target y (same shape), mask of scored positions"""
    m = torch.zeros(n, T, dtype=torch.bool)
    if task == "back":
        x = torch.randint(V, (n, T), generator=g); y = torch.roll(x, 4, 1); m[:, 4:] = True
        return x, y, m
    if task == "copy":
        s = torch.randint(V, (n, T + 1), generator=g); k = torch.randint(6, 15, (n,), generator=g)
        st = (torch.rand(n, generator=g) * (T + 1 - 2 * k + 1).float()).long()
        for i in range(n):
            a, kk = st[i].item(), k[i].item(); s[i, a + kk:a + 2 * kk] = s[i, a:a + kk]; m[i, a + kk:a + 2 * kk - 1] = True
        return s[:, :-1], s[:, 1:], m
    if task == "recall":                          # keys 0-31, values 32-63; sequence k v k v ... k
        x = torch.zeros(n, 17, dtype=torch.long); y = torch.zeros_like(x); m = torch.zeros(n, 17, dtype=torch.bool)
        for i in range(n):
            ks = torch.randperm(32, generator=g)[:8]; vs = 32 + torch.randint(32, (8,), generator=g)
            x[i, 0:16:2], x[i, 1:16:2] = ks, vs; j = torch.randint(8, (1,), generator=g).item(); x[i, 16] = ks[j]; y[i, 16] = vs[j]
        m[:, 16] = True
        return x, y, m
    if task == "add":                             # tokens 0-30 numbers, 31 '='
        a, b = torch.randint(31, (n,), generator=g), torch.randint(31, (n,), generator=g)
        x = torch.stack([a, b, torch.full_like(a, 31)], 1); y = torch.zeros_like(x); y[:, 2] = (a + b) % 31
        m = torch.zeros(n, 3, dtype=torch.bool); m[:, 2] = True
        return x, y, m
    if task == "sort":                            # tokens 0-31, separator 63
        a = torch.randint(32, (n, 10), generator=g); s = torch.cat([a, torch.full((n, 1), 63), a.sort(1).values], 1)
        m = torch.zeros(n, 20, dtype=torch.bool); m[:, 10:] = True
        return s[:, :-1], s[:, 1:], m
    if task == "markov":
        s = torch.zeros(n, T + 1, dtype=torch.long); s[:, 0] = torch.randint(V, (n,), generator=g)
        for t in range(T): s[:, t + 1] = torch.multinomial(CHAIN[s[:, t]], 1, generator=g)[:, 0]
        m[:, :] = True
        return s[:, :-1], s[:, 1:], m


MSIGN = os.environ.get("MSIGN", "svd")  # svd = exact (older numbers); ns = Muon's 5-step quintic Newton-Schulz (as in real Muon, 3x faster here)


def msign(g):
    if MSIGN == "ns":
        a, b, c = 3.4445, -4.7750, 2.0315; t = g.shape[-2] > g.shape[-1]
        X = g.mT if t else g; X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(5): A = X @ X.mT; X = a * X + (b * A + c * A @ A) @ X
        return X.mT if t else X
    U, _, Vh = torch.linalg.svd(g, full_matrices=False)
    return U @ Vh * (g.norm(dim=(-2, -1), keepdim=True) > 0)   # a zero gradient (an expert no token chose) gives no step


def qk_step(u, mode):
    if mode == "whole": return msign(u)
    ph = torch.cat([msign(c) for c in u.chunk(H, 0)], 0)
    if mode == "perhead": return ph
    U, S, Vh = torch.linalg.svd(ph, full_matrices=False)  # pow:a = per-head directions, singular values ^ a, same size
    d = U @ torch.diag(S ** float(mode[4:])) @ Vh
    return d * (ph.norm() / d.norm())


def run(task, mode, seed=0, steps=300, lr=0.03, every=10):
    torch.manual_seed(seed); model = Model()
    names = [(l, n) for l in range(NL) for n in MATS]
    mats = [getattr(model.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    alr, fast = float(os.environ.get("ADAM_LR", "0.01")), [w for w in os.environ.get("ADAM2", "").split(",") if w]  # ADAM2=pos,emb: those Adam params get 2x lr
    rest = [(n, p) for n, p in model.named_parameters() if id(p) not in ids]
    adam = torch.optim.AdamW([dict(params=[p for n, p in rest if any(w in n for w in fast)], lr=2 * alr),
                              dict(params=[p for n, p in rest if not any(w in n for w in fast)], lr=alr)], betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]
    g, xe, ye, me = torch.Generator().manual_seed(seed), *batch(task, 256, torch.Generator().manual_seed(999))
    curve = []
    for i in range(steps + 1):
        if i % every == 0:
            with torch.no_grad(): curve.append(round(F.cross_entropy(model(xe)[me], ye[me]).item(), 4))
        if i == steps: break
        x, y, msk = batch(task, 32, g)
        loss = F.cross_entropy(model(x)[msk], y[msk])
        f = lr_at(i, steps)
        for gr in adam.param_groups: gr["lr"] = gr.setdefault("base", gr["lr"]) * f
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            for p, b, (l, n) in zip(mats, buf, names):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                st = qk_step(u, mode) if n in ("q", "k") else msign(u)
                p.add_(st, alpha=-lr * f * MULT.get(n, 1) * max(1, p.shape[-2] / p.shape[-1]) ** 0.5)
    return curve


if __name__ == "__main__":
    task, mode = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0; steps = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    t0 = time.time(); c = run(task, mode, seed, steps)
    r = dict(task=task, mode=mode, seed=seed, steps=steps, sec=round(time.time() - t0, 1), curve=c)
    with open(__file__.rsplit("/", 1)[0] + "/runs.jsonl", "a") as f: f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "curve"}), "loss", c[0], c[len(c) // 4], c[len(c) // 2], c[-1])
