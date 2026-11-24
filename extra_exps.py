#!/usr/bin/env python
"""Extra experiments for the connectome-TKG NeuroAI paper (Witvliet developmental C. elegans).
Modes
  budget     : Selective vs RandomMask R-GCN at fixed edge budgets, real vs degree-preserving rewired inputs
  rolling    : rolling-origin evaluation (test stage T=5..8; train 2..T-2, val T-1) + heuristic-LR combination
  gates      : what do the learned gates keep? per-synapse gate analysis on test inputs
  geometry   : Poincare curvature probe (real/rewired, hyperbolic vs Euclidean, curvature-init sensitivity)
  summary    : statistics for budget / rolling / gates CSVs
  geosummary : statistics over all geometry CSVs
Ranking: mrr/h1/h10 use tie-aware (mean) ranks; mrr_opt/h1_opt/h10_opt use optimistic ranks (only strictly
greater scores count), which is what the original rank_scores() does.
"""
import argparse, os, math, random, copy, glob
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu, rankdata, pearsonr
from scipy.stats import t as tdist

STAGES = list(range(1, 9))
RELS = ["chemical", "electrical"]
HEURS = ["PA", "CN", "AA", "Recip"]


# ----------------------------------------------------------------------------- data (same as connectome_tkg.py)
def load(data_dir):
    frames = {}
    for s in STAGES:
        d = pd.read_excel(os.path.join(data_dir, f"witvliet_2020_{s}.xlsx"))
        d.columns = [c.strip().lower() for c in d.columns]
        frames[s] = d
    nodes = sorted(set().union(*[set(d.pre) | set(d.post) for d in frames.values()]))
    idx = {n: i for i, n in enumerate(nodes)}
    E = {}
    for s, d in frames.items():
        e = {}
        for p, q, t, w in d[["pre", "post", "type", "synapses"]].itertuples(index=False):
            if p == q:
                continue
            r = RELS.index(t)
            u, v = idx[p], idx[q]
            e[(u, r, v)] = e.get((u, r, v), 0) + int(w)
            if r == 1:
                e[(v, r, u)] = e.get((v, r, u), 0) + int(w)
        E[s] = e
    return nodes, E


def new_edge_queries(E, t):
    seen = set().union(*[set(E[s]) for s in STAGES if s < t])
    return [(u, r, v) for (u, r, v) in E[t] if (u, r, v) not in seen and not (r == 1 and u > v)]


def filter_mask(E, t, n):
    excl = {}
    prev = set().union(*[set(E[s]) for s in STAGES if s < t])
    for (u, r, v) in set(E[t]) | prev:
        excl.setdefault((u, r), set()).add(v)
    return excl


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def rewire(G, seed, mult=10):
    """Degree-preserving rewiring: chemical as directed graph, electrical as undirected (kept symmetric)."""
    rng = np.random.default_rng(seed)
    out = {}
    ed = [(u, v) for (u, r, v) in G if r == 0]
    w = [G[(u, 0, v)] for (u, v) in ed]
    S, m = set(ed), len(ed)
    for _ in range(mult * m):
        i, j = rng.integers(m, size=2)
        if i == j:
            continue
        (a, b), (c, d) = ed[i], ed[j]
        if a == d or c == b or (a, d) in S or (c, b) in S:
            continue
        S.discard((a, b)); S.discard((c, d)); S.add((a, d)); S.add((c, b))
        ed[i], ed[j] = (a, d), (c, b)
    for (u, v), ww in zip(ed, w):
        out[(u, 0, v)] = ww
    pr = sorted({(min(u, v), max(u, v)) for (u, r, v) in G if r == 1})
    w = [G[(u, 1, v)] for (u, v) in pr]
    S, m = set(pr), len(pr)
    for _ in range(mult * m):
        i, j = rng.integers(m, size=2)
        if i == j:
            continue
        (a, b), (c, d) = pr[i], pr[j]
        if rng.random() < 0.5:
            c, d = d, c
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if a == d or c == b or e1 == e2 or e1 in S or e2 in S:
            continue
        S.discard(pr[i]); S.discard(pr[j]); S.add(e1); S.add(e2)
        pr[i], pr[j] = e1, e2
    for (u, v), ww in zip(pr, w):
        out[(u, 1, v)] = ww; out[(v, 1, u)] = ww
    return out


def build(E, t, n, G, dev):
    """Target stage t (new-synapse queries), input graph G (= stage t-1 snapshot, possibly rewired)."""
    qs = new_edge_queries(E, t)
    excl = filter_mask(E, t, n)
    M = torch.zeros(len(qs), n, dtype=torch.bool)
    for i, (u, r, v) in enumerate(qs):
        M[i, list(excl.get((u, r), set()) - {v}) + [u]] = True
    syn = list(G.keys())
    src, dst, typ, cnt, sid = [], [], [], [], []
    for k, (u, r, v) in enumerate(syn):
        w = float(G[(u, r, v)])
        if r == 0:   # chemical: forward + inverse message types
            src += [u, v]; dst += [v, u]; typ += [0, 1]; cnt += [w, w]; sid += [k, k]
        else:        # electrical: already symmetric
            src.append(u); dst.append(v); typ.append(2); cnt.append(w); sid.append(k)
    src, dst, typ = torch.tensor(src), torch.tensor(dst), torch.tensor(typ)
    deg = torch.zeros(3, n)
    for tt in range(3):
        mm = typ == tt
        deg[tt].index_add_(0, dst[mm], torch.ones(int(mm.sum())))
    A = torch.zeros(n, n)
    for (u, r, v) in syn:
        A[u, v] = 1.0
    U = ((A + A.T) > 0).float()
    dg = U.sum(1)
    CN = U @ U
    inv = torch.where(dg > 1, 1.0 / torch.log(dg.clamp(min=2)), torch.zeros_like(dg))
    AA = (U * inv[None, :]) @ U
    PA = dg[:, None] * dg[None, :]
    REC = A.T.contiguous()                      # Recip(u,v)=1 if v->u exists (any relation)
    qu = torch.tensor([q[0] for q in qs]); qr = torch.tensor([q[1] for q in qs]); qv = torch.tensor([q[2] for q in qs])
    H = torch.stack([torch.log1p(PA), torch.log1p(CN), AA, REC], -1)[qu]
    g = dict(qu=qu, qr=qr, qv=qv, M=M, src=src, dst=dst, typ=typ, cnt=torch.tensor(cnt),
             sid=torch.tensor(sid), deg=deg.clamp(min=1), H=H)
    g = {k: v.to(dev) for k, v in g.items()}
    g["syn"] = syn
    return g


def make_D(E, n, ts, dev, rw_seed=None):
    return {t: build(E, t, n, rewire(E[t - 1], rw_seed * 100 + t) if rw_seed is not None else E[t - 1], dev)
            for t in ts}


# ----------------------------------------------------------------------------- models
class Net(nn.Module):
    def __init__(self, n, d, kind, use_h, nl=2):
        super().__init__()
        self.kind, self.use_h, self.nl = kind, use_h, nl
        self.emb = nn.Embedding(n, d); nn.init.normal_(self.emb.weight, std=0.1)
        if kind != "distmult":
            self.selfl = nn.ModuleList([nn.Linear(d, d) for _ in range(nl)])
            self.rel = nn.ModuleList([nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(3)]) for _ in range(nl)])
        if kind == "selective":
            self.gnet = nn.Sequential(nn.Linear(2 * d + 4, 64), nn.ReLU(), nn.Linear(64, 1))
            nn.init.constant_(self.gnet[2].bias, 1.0)          # start with most edges kept
        self.R = nn.Parameter(torch.ones(2, d) + 0.1 * torch.randn(2, d))
        if use_h:
            self.hl = nn.Linear(4, 1)

    def gate_logits(self, g):
        e = self.emb.weight
        x = torch.cat([e[g["src"]], e[g["dst"]], F.one_hot(g["typ"], 3).float(), torch.log1p(g["cnt"])[:, None]], 1)
        return self.gnet(x).squeeze(-1)

    def edge_w(self, g, q, gen):
        m, dev = g["src"].numel(), g["src"].device
        if self.kind in ("distmult", "rgcn"):
            return torch.ones(m, device=dev), None
        if self.kind == "random":
            k = max(1, int(round(q * m)))
            w = torch.zeros(m, device=dev)
            w[torch.randperm(m, generator=gen)[:k].to(dev)] = 1.0
            return w, None
        p = torch.sigmoid(self.gate_logits(g))
        if q is None:
            hard = (p > 0.5).float()
        else:
            hard = torch.zeros_like(p)
            hard[torch.topk(p, max(1, int(round(q * m)))).indices] = 1.0
        return hard + p - p.detach(), p                       # straight-through hard mask

    def encode(self, g, w):
        h = self.emb.weight
        if self.kind == "distmult":
            return h
        for l in range(self.nl):
            out = self.selfl[l](h)
            for tt in range(3):
                mm = g["typ"] == tt
                if int(mm.sum()) == 0:
                    continue
                d_ = g["dst"][mm]
                msg = self.rel[l][tt](h[g["src"][mm]]) * w[mm, None] / g["deg"][tt][d_][:, None]
                out = out.index_add(0, d_, msg)
            h = F.relu(out) if l < self.nl - 1 else out
        return h

    def run(self, g, q, gen):
        w, p = self.edge_w(g, q, gen)
        h = self.encode(g, w)
        sc = (h[g["qu"]] * self.R[g["qr"]]) @ h.T
        if self.use_h:
            sc = sc + self.hl(g["H"]).squeeze(-1)
        return sc.masked_fill(g["M"], -1e9), w, p


@torch.no_grad()
def metrics(sc, g):
    tg = sc.gather(1, g["qv"][:, None])
    gt = (sc > tg).sum(1).float()
    eq = (sc == tg).sum(1).float() - 1
    rm, ro = gt + 1 + eq / 2, gt + 1
    f = lambda x: float(x.float().mean())
    return dict(mrr=f(1 / rm), h1=f(rm <= 1), h10=f(rm <= 10), mrr_opt=f(1 / ro), h1_opt=f(ro <= 1), h10_opt=f(ro <= 10))


def run_model(kind, use_h, D, tr, va, te, seed, dev, a, q=None):
    set_seed(seed)
    n = D[tr[0]]["M"].shape[1]
    net = Net(n, a.dim, kind, use_h).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    best, state = -1.0, copy.deepcopy(net.state_dict())
    for ep in range(a.epochs):
        net.train(); opt.zero_grad()
        loss = 0.0
        for t in tr:
            sc, w, p = net.run(D[t], q, gen)
            loss = loss + F.cross_entropy(sc, D[t]["qv"])
            if p is not None and q is None:
                loss = loss + a.lam * p.mean()
        loss.backward(); opt.step()
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                sc, _, _ = net.run(D[va], q, torch.Generator().manual_seed(10 ** 6 + seed))
                mv = metrics(sc, D[va])["mrr"]
            if mv > best:
                best, state = mv, copy.deepcopy(net.state_dict())
    net.load_state_dict(state); net.eval()
    rows = []
    with torch.no_grad():
        for t in te:
            sc, w, _ = net.run(D[t], q, torch.Generator().manual_seed(10 ** 6 + seed))
            r = metrics(sc, D[t])
            r.update(model=kind + ("+H" if use_h else ""), kind=kind, use_h=int(use_h),
                     q=-1.0 if q is None else q, seed=seed, stage=t, kept=float((w > 0.5).float().mean()))
            rows.append(r)
    return rows, net


def heuristic_rows(D, te):
    rows = []
    for t in te:
        for j, name in enumerate(HEURS):
            r = metrics(D[t]["H"][..., j].masked_fill(D[t]["M"], -1e9), D[t])
            r.update(model=name, kind="heuristic", use_h=0, q=-1.0, seed=-1, stage=t, kept=1.0)
            rows.append(r)
    return rows


def heur_lr_rows(D, tr, va, te, seed, a):
    """Listwise linear combination of the 4 heuristics (trained with CE over all candidates)."""
    set_seed(seed)
    lin = nn.Linear(4, 1).to(D[tr[0]]["H"].device)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    sc_of = lambda g: lin(g["H"]).squeeze(-1).masked_fill(g["M"], -1e9)
    best, state = -1.0, copy.deepcopy(lin.state_dict())
    for ep in range(300):
        opt.zero_grad()
        loss = sum(F.cross_entropy(sc_of(D[t]), D[t]["qv"]) for t in tr)
        loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                mv = metrics(sc_of(D[va]), D[va])["mrr"]
            if mv > best:
                best, state = mv, copy.deepcopy(lin.state_dict())
    lin.load_state_dict(state)
    rows = []
    with torch.no_grad():
        for t in te:
            r = metrics(sc_of(D[t]), D[t])
            r.update(model="HeurLR", kind="heurlr", use_h=0, q=-1.0, seed=seed, stage=t, kept=1.0,
                     weights=";".join(f"{x:.3f}" for x in lin.weight.view(-1).tolist()))
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------- modes
def save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def mode_budget(E, n, a, dev):
    tr, va, te = [2, 3, 4, 5], 6, [7, 8]
    rows, path = [], os.path.join(a.out, f"budget_h{''.join(map(str, a.use_h))}.csv")
    Dreal = make_D(E, n, tr + [va] + te, dev)
    for rw in (0, 1):
        for seed in a.seeds:
            D = Dreal if rw == 0 else make_D(E, n, tr + [va] + te, dev, rw_seed=seed)
            for h in a.use_h:
                for q in a.budgets:
                    for kind in ("selective", "random"):
                        rr, _ = run_model(kind, h, D, tr, va, te, seed, dev, a, q=q)
                        for r in rr:
                            r["rewired"] = rw
                        rows += rr
                save(rows, path)
                print(f"[budget] rewired={rw} seed={seed} h={h} done", flush=True)
    save(rows, path)


def mode_rolling(E, n, a, dev):
    rows, path = [], os.path.join(a.out, "rolling.csv")
    for T in (5, 6, 7, 8):
        tr, va, te = list(range(2, T - 1)), T - 1, [T]
        D = make_D(E, n, tr + [va] + te, dev)
        rows += heuristic_rows(D, te)
        for seed in a.seeds:
            rows += heur_lr_rows(D, tr, va, te, seed, a)
            for h in a.use_h:
                for kind in ("distmult", "rgcn"):
                    rr, _ = run_model(kind, h, D, tr, va, te, seed, dev, a); rows += rr
                qs = a.gate_q if a.gate_q > 0 else None
                rr, _ = run_model("selective", h, D, tr, va, te, seed, dev, a, q=qs); rows += rr
                kept = float(np.mean([r["kept"] for r in rr]))
                rr, _ = run_model("random", h, D, tr, va, te, seed, dev, a, q=kept)
                for r in rr:
                    r["model"] = "RandomMatched" + ("+H" if h else "")
                rows += rr
            save(rows, path)
            print(f"[rolling] T={T} seed={seed} done", flush=True)
    save(rows, path)


def mode_gates(E, n, nodes, a, dev):
    tr, va, te = [2, 3, 4, 5], 6, [7, 8]
    D = make_D(E, n, tr + [va] + te, dev)
    cls = None
    if a.classes:
        c = pd.read_csv(a.classes)
        c.columns = [x.strip().lower() for x in c.columns]
        cls = dict(zip(c["name"].astype(str), c["class"].astype(str)))
    rows, path = [], os.path.join(a.out, "gates.csv")
    for h in a.use_h:
        for seed in a.seeds:
            _, net = run_model("selective", h, D, tr, va, te, seed, dev, a, q=a.gate_q if a.gate_q > 0 else None)
            with torch.no_grad():
                for t in te:
                    g, s = D[t], t - 1
                    p = torch.sigmoid(net.gate_logits(g))
                    ns = len(g["syn"])
                    ps = torch.zeros(ns, device=p.device).index_add_(0, g["sid"], p)
                    cs = torch.zeros(ns, device=p.device).index_add_(0, g["sid"], torch.ones_like(p))
                    ps = (ps / cs).cpu().numpy()
                    keep_thr = np.quantile(ps, 1 - a.gate_q) if a.gate_q > 0 else 0.5
                    dout = {}
                    for (u, r, v) in E[s]:
                        dout[u] = dout.get(u, 0) + 1
                    for k, (u, r, v) in enumerate(g["syn"]):
                        if r == 1 and u > v:
                            continue   # one row per gap junction
                        rec = ((v, 0, u) in E[s]) if r == 0 else (((u, 0, v) in E[s]) or ((v, 0, u) in E[s]))
                        row = dict(use_h=h, seed=seed, input_stage=s, pre=nodes[u], post=nodes[v], relation=RELS[r],
                                   count=E[s][(u, r, v)], persistence=sum((u, r, v) in E[x] for x in STAGES if x <= s),
                                   reciprocated=int(rec), survives=int((u, r, v) in E[t]),
                                   deg_pre=dout.get(u, 0), deg_post=dout.get(v, 0), gate=float(ps[k]),
                                   kept=int(ps[k] >= keep_thr))
                        if cls is not None:
                            row["class_pre"] = cls.get(nodes[u], "NA"); row["class_post"] = cls.get(nodes[v], "NA")
                        rows.append(row)
            save(rows, path)
            print(f"[gates] h={h} seed={seed} done", flush=True)
    save(rows, path)


# ----------------------------------------------------------------------------- geometry
def snorm(x):
    return torch.sqrt((x * x).sum(-1) + 1e-10)


def mob_add(x, y, c):
    xy = (x * y).sum(-1, keepdim=True); x2 = (x * x).sum(-1, keepdim=True); y2 = (y * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    return num / (1 + 2 * c * xy + c * c * x2 * y2).clamp_min(1e-15)


def expmap0(v, c):
    sc = c.sqrt(); nv = snorm(v)[..., None]
    return torch.tanh(sc * nv) * v / (sc * nv)


def pdist(x, y, c):
    sc = c.sqrt()
    return 2 / sc * torch.atanh((sc * snorm(mob_add(-x, y, c))).clamp(max=1 - 1e-5))


class Geo(nn.Module):
    def __init__(self, n, d, hyp, c_init):
        super().__init__()
        self.hyp = hyp
        self.x = nn.Parameter(torch.randn(n, d) * 0.1)
        self.pc = nn.Parameter(torch.full((2,), math.log(math.expm1(c_init))))
        self.pt = nn.Parameter(torch.zeros(2))

    def curv(self):
        return F.softplus(self.pc) + 1e-5

    def scores(self, qu, qr):
        out = []
        for r in (0, 1):
            if self.hyp:
                c = self.curv()[r]; X = expmap0(self.x, c)
                d = pdist(X[qu][:, None, :], X[None, :, :], c)
            else:
                d = snorm(self.x[qu][:, None, :] - self.x[None, :, :])
            out.append(-F.softplus(self.pt[r] + 1.0) * d)
        return torch.where(qr[:, None] == 0, out[0], out[1])


def geo_run(Gs, n, dim, hyp, c_init, seed, dev, a):
    set_seed(seed)
    syn = sorted({(u, r, v) if r == 0 else (min(u, v), 1, max(u, v)) for (u, r, v) in Gs})
    perm = np.random.default_rng(seed).permutation(len(syn))
    test = set(syn[i] for i in perm[:max(1, int(0.1 * len(syn)))])
    dirs = lambda k: [k] if k[1] == 0 else [(k[0], 1, k[2]), (k[2], 1, k[0])]
    trq = [d for k in syn if k not in test for d in dirs(k)]
    teq = [d for k in test for d in dirs(k)]
    true = {}
    for (u, r, v) in Gs:
        true.setdefault((u, r), set()).add(v)

    def pack(qs):
        M = torch.zeros(len(qs), n, dtype=torch.bool)
        for i, (u, r, v) in enumerate(qs):
            M[i, list(true[(u, r)] - {v}) + [u]] = True
        t = lambda j: torch.tensor([q[j] for q in qs], device=dev)
        return dict(qu=t(0), qr=t(1), qv=t(2), M=M.to(dev))

    TR, TE = pack(trq), pack(teq)
    m = Geo(n, dim, hyp, c_init).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.geo_lr)
    for _ in range(a.geo_epochs):
        opt.zero_grad()
        F.cross_entropy(m.scores(TR["qu"], TR["qr"]).masked_fill(TR["M"], -1e9), TR["qv"]).backward()
        opt.step()
    with torch.no_grad():
        mt = metrics(m.scores(TE["qu"], TE["qr"]).masked_fill(TE["M"], -1e9), TE)["mrr"]
        c = m.curv().tolist()
    return c, mt


def fanout(Gs, r):
    od = {}
    for (u, rr, v) in Gs:
        if rr == r:
            od[u] = od.get(u, 0) + 1
    return float(np.mean(list(od.values()))) if od else 0.0


def mode_geometry(E, n, a, dev):
    rows, path = [], os.path.join(a.out, f"geometry_{a.tag}.csv")
    for dim in a.dims:
        for s in STAGES:
            for graph in a.graphs:
                for seed in a.seeds:
                    Gs = E[s] if graph == "real" else rewire(E[s], seed * 1000 + s)
                    for geom in a.geoms:
                        c, mt = geo_run(Gs, n, dim, geom == "hyp", a.c_init, seed, dev, a)
                        for r in (0, 1):
                            nsyn = len({(u, v) if r == 0 else (min(u, v), max(u, v)) for (u, rr, v) in Gs if rr == r})
                            rows.append(dict(tag=a.tag, dim=dim, stage=s, relation=RELS[r], graph=graph, geom=geom,
                                             c_init=a.c_init, seed=seed, curv=c[r] if geom == "hyp" else np.nan,
                                             mrr=mt, fanout=fanout(Gs, r), n_edges=nsyn))
                save(rows, path)
                print(f"[geometry:{a.tag}] dim={dim} stage={s} graph={graph} done", flush=True)
    save(rows, path)


# ----------------------------------------------------------------------------- statistics
def wil(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) < 2 or np.allclose(x, y):
        return float("nan")
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return float("nan")


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)
    r, _ = pearsonr(res(rx), res(ry))
    df_ = len(x) - 3
    tt = r * math.sqrt(df_ / max(1e-12, 1 - r * r))
    return r, 2 * tdist.sf(abs(tt), df_)


def summarize(a):
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    for f in sorted(glob.glob(os.path.join(a.out, "budget_h*.csv"))):
        df = pd.read_csv(f)
        print(f"\n===== BUDGET ({os.path.basename(f)}) =====")
        print(df.groupby(["rewired", "use_h", "q", "kind"])[["mrr", "mrr_opt", "kept"]].agg(["mean", "std"]).round(4))
        print("\nSelective vs Random at matched budget (paired over seed x stage):")
        for (rw, h, q), sub in df.groupby(["rewired", "use_h", "q"]):
            s_ = sub[sub.kind == "selective"].sort_values(["seed", "stage"]).mrr.values
            r_ = sub[sub.kind == "random"].sort_values(["seed", "stage"]).mrr.values
            print(f"  rewired={rw} H={h} q={q:.2f}: dMRR={np.mean(s_) - np.mean(r_):+.4f}  p={wil(s_, r_):.3g}  n={len(s_)}")
        print("\nReal vs rewired (Selective) at each budget:")
        for (h, q), sub in df[df.kind == "selective"].groupby(["use_h", "q"]):
            x = sub[sub.rewired == 0].sort_values(["seed", "stage"]).mrr.values
            y = sub[sub.rewired == 1].sort_values(["seed", "stage"]).mrr.values
            print(f"  H={h} q={q:.2f}: real-rewired dMRR={np.mean(x) - np.mean(y):+.4f}  p={wil(x, y):.3g}")
    f = os.path.join(a.out, "rolling.csv")
    if os.path.exists(f):
        df = pd.read_csv(f)
        print("\n===== ROLLING-ORIGIN (test stage = T) =====")
        for col in ("mrr", "mrr_opt"):
            print(f"\n{col} (mean over seeds):")
            print(df.pivot_table(index="model", columns="stage", values=col, aggfunc="mean").round(4))
        print("\nMRR std over seeds:")
        print(df.pivot_table(index="model", columns="stage", values="mrr", aggfunc="std").round(4))
        print("\nPaired tests pooled over T=5..8 (seed x stage):")
        def pair(m1, m2):
            x = df[df.model == m1].sort_values(["stage", "seed"])
            y = df[df.model == m2].sort_values(["stage", "seed"])
            if len(x) == 0 or len(y) == 0:
                return
            if y.seed.iloc[0] == -1:   # deterministic heuristic: broadcast to seeds
                y = x[["stage", "seed"]].merge(y.drop(columns="seed"), on="stage")
            print(f"  {m1:18s} vs {m2:18s}: dMRR={x.mrr.mean() - y.mrr.mean():+.4f}  p={wil(x.mrr.values, y.mrr.values):.3g}  n={len(x)}")
        for m1, m2 in [("HeurLR", "Recip"), ("HeurLR", "AA"), ("selective+H", "HeurLR"), ("rgcn+H", "HeurLR"),
                       ("distmult+H", "HeurLR"), ("selective", "rgcn"), ("selective", "RandomMatched"),
                       ("selective+H", "rgcn+H"), ("selective+H", "RandomMatched+H"), ("rgcn", "distmult")]:
            pair(m1, m2)
        if "weights" in df.columns:
            w = df[df.model == "HeurLR"].weights.dropna().str.split(";", expand=True).astype(float)
            w.columns = HEURS
            print("\nHeurLR weights (mean over runs):", w.mean().round(3).to_dict())
    f = os.path.join(a.out, "gates.csv")
    if os.path.exists(f):
        df = pd.read_csv(f)
        if "kept" not in df.columns:
            df["kept"] = (df.gate > 0.5).astype(int)
        df["count_q"] = df.groupby(["use_h", "seed", "input_stage"])["count"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 4, labels=False) + 1)
        df["persist_bin"] = np.where(df.persistence >= df.input_stage, "all-prior-stages",
                                     np.where(df.persistence == 1, "new-at-input", "intermittent"))
        print("\n===== GATES =====")
        for h, sub in df.groupby("use_h"):
            print(f"\n--- use_h={h}: overall kept={sub.kept.mean():.3f}")
            for col in ["relation", "count_q", "persist_bin", "reciprocated", "survives"] + \
                       [c for c in ("class_pre", "class_post") if c in sub.columns]:
                print(sub.groupby(col)[["gate", "kept"]].mean().round(3).assign(n=sub.groupby(col).size()))
            for col in ("count", "persistence", "deg_pre", "deg_post"):
                rh = [spearmanr(g_[col], g_.gate)[0] for _, g_ in sub.groupby(["seed", "input_stage"])]
                ps = [spearmanr(g_[col], g_.gate)[1] for _, g_ in sub.groupby(["seed", "input_stage"])]
                print(f"  gate~{col}: mean rho={np.nanmean(rh):+.3f}  frac p<.05={np.mean(np.array(ps) < .05):.2f}")
            pv = [mannwhitneyu(g_[g_.survives == 1].gate, g_[g_.survives == 0].gate).pvalue
                  for _, g_ in sub.groupby(["seed", "input_stage"]) if g_.survives.nunique() == 2]
            print(f"  gate higher for synapses that survive to next stage? frac Mann-Whitney p<.05 = {np.mean(np.array(pv) < .05):.2f}")


def geosummary(a):
    fs = glob.glob(os.path.join(a.out, "geometry_*.csv"))
    if not fs:
        print("no geometry CSVs"); return
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["tag"] = df.tag.replace({"d4": "main"})
    hyp = df[df.geom == "hyp"]
    agg = hyp.groupby(["tag", "graph", "c_init", "dim", "relation", "stage"]).agg(
        curv=("curv", "mean"), sd=("curv", "std"), fanout=("fanout", "first"),
        n_edges=("n_edges", "first"), mrr=("mrr", "mean")).reset_index()
    print("===== LEARNED CURVATURE =====")
    print(agg.round(4).to_string(index=False))
    print("\n===== CORRELATIONS (Spearman; partial = controlling for log #edges) =====")
    for (tag, graph, ci, dim), sub in agg.groupby(["tag", "graph", "c_init", "dim"]):
        r, p = spearmanr(sub.curv, sub.fanout)
        print(f"\n[{tag} | {graph} | c_init={ci} | d={dim}] pooled curv~fanout rho={r:+.3f} (p={p:.3g}, n={len(sub)})")
        for rel, s2 in sub.groupby("relation"):
            r1, p1 = spearmanr(s2.curv, s2.fanout); r2, p2 = spearmanr(s2.curv, s2.stage)
            r3, p3 = partial_spearman(s2.curv.values, s2.fanout.values, np.log(s2.n_edges.values))
            print(f"   {rel:10s} curv~fanout {r1:+.3f} (p={p1:.3g}) | curv~stage {r2:+.3f} (p={p2:.3g}) | "
                  f"partial curv~fanout|edges {r3:+.3f} (p={p3:.3g})")
    real = df[(df.graph == "real") & (df.tag == "main")]
    if set(real.geom) >= {"hyp", "euc"}:
        print("\n===== HYPERBOLIC vs EUCLIDEAN link-prediction MRR (real, paired over stage x seed) =====")
        pv = real[real.relation == "chemical"].pivot_table(index=["dim", "stage", "seed"], columns="geom", values="mrr").dropna()
        for dim, s in pv.groupby(level="dim"):
            print(f"  d={dim}: hyp={s.hyp.mean():.4f}  euc={s.euc.mean():.4f}  d={s.hyp.mean() - s.euc.mean():+.4f}  p={wil(s.hyp, s.euc):.3g}  n={len(s)}")
    key = ["dim", "relation", "stage"]
    m = agg[agg.tag == "main"]
    rw = agg[agg.graph == "rewired"]
    if len(rw):
        j = m[m.graph == "real"].merge(rw, on=key, suffixes=("_real", "_rw"))
        print("\n===== REAL vs REWIRED curvature (per dim; paired over relation x stage) =====")
        for dim, s in j.groupby("dim"):
            print(f"  d={dim}: real={s.curv_real.mean():.4f}  rewired={s.curv_rw.mean():.4f}  p={wil(s.curv_real, s.curv_rw):.3g}  n={len(s)}")
    oth = agg[(agg.graph == "real") & (agg.tag != "main")]
    if len(oth):
        print("\n===== CURVATURE-INIT SENSITIVITY =====")
        for (tag, ci), o in oth.groupby(["tag", "c_init"]):
            j = m[m.graph == "real"].merge(o, on=key, suffixes=("_main", "_alt"))
            for dim, s in j.groupby("dim"):
                r, p = spearmanr(s.curv_main, s.curv_alt)
                print(f"  [{tag}, c_init={ci}] d={dim}: mean c main={s.curv_main.mean():.4f} alt={s.curv_alt.mean():.4f}  "
                      f"rank agreement rho={r:+.3f} (p={p:.3g})")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["budget", "rolling", "gates", "geometry", "summary", "geosummary"])
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="extra")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--use_h", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dim", type=int, default=64); ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--gate_q", type=float, default=0.4, help="edge budget for Selective in rolling/gates (<=0: threshold gates at 0.5)")
    ap.add_argument("--classes", default=None, help="optional CSV with columns name,class")
    ap.add_argument("--graphs", nargs="+", default=["real"]); ap.add_argument("--geoms", nargs="+", default=["hyp"])
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16]); ap.add_argument("--c_init", type=float, default=0.05)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--geo_epochs", type=int, default=400); ap.add_argument("--geo_lr", type=float, default=0.01)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "summary":
        return summarize(a)
    if a.mode == "geosummary":
        return geosummary(a)
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    nodes, E = load(a.data)
    n = len(nodes)
    print(f"nodes={n}  edges/stage={[len(E[s]) for s in STAGES]}  device={dev}", flush=True)
    {"budget": lambda: mode_budget(E, n, a, dev), "rolling": lambda: mode_rolling(E, n, a, dev),
     "gates": lambda: mode_gates(E, n, nodes, a, dev), "geometry": lambda: mode_geometry(E, n, a, dev)}[a.mode]()


if __name__ == "__main__":
    main()
