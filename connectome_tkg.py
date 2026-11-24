"""Developmental C. elegans connectome (Witvliet et al. 2021, 8 stages) as a temporal KG.
Exp A (forecast): rank synapses NEW at stage k+1 from stage k. Heuristics PA/CN/AA/Recip;
  DistMult, R-GCN, Selective R-GCN (learned edge gates), Random-mask R-GCN; +H = hybrid with
  heuristic features; every GNN also on degree-preserving rewired inputs. Train 2-5, val 6, test 7-8.
Exp B (geometry): per stage, Poincare model with one learnable curvature per relation.
Usage: python connectome_tkg.py --data data --out results --seeds 0 1 2 --device cuda:0 --exp both
"""
import argparse, json, math, os, random, time
import numpy as np, pandas as pd, networkx as nx, torch, torch.nn as nn, torch.nn.functional as F

STAGES = list(range(1, 9))
RELS = ["chemical", "electrical"]


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


def rank_scores(scores, queries, excl):
    ranks = []
    for i, (u, r, v) in enumerate(queries):
        s = scores[i].clone()
        bad = list(excl.get((u, r), set()) - {v}) + [u]
        s[bad] = -float("inf")
        ranks.append(int((s > s[v]).sum().item()) + 1)
    return np.array(ranks)


def metrics(ranks):
    return dict(MRR=float(np.mean(1.0 / ranks)), H1=float(np.mean(ranks <= 1)),
                H3=float(np.mean(ranks <= 3)), H10=float(np.mean(ranks <= 10)), n=int(len(ranks)))


def heuristic_scores(E_in, queries, n, name):
    A = np.zeros((2, n, n))
    for (u, r, v), w in E_in.items():
        A[r, u, v] = 1
    Au = ((A[0] + A[0].T + A[1]) > 0).astype(float)
    deg = Au.sum(1)
    out_deg = A[0].sum(1) + A[1].sum(1)
    in_deg = A[0].sum(0) + A[1].sum(0)
    S = np.zeros((len(queries), n))
    for i, (u, r, v) in enumerate(queries):
        if name == "PA":
            S[i] = out_deg[u] * in_deg + 1e-9 * deg
        elif name == "CN":
            S[i] = Au[u] @ Au + 1e-6 * deg
        elif name == "AA":
            S[i] = (Au[u] / np.log(np.maximum(deg, 2))) @ Au + 1e-6 * deg
        elif name == "Recip":
            S[i] = A[0][:, u] + A[1][u] + 1e-3 * (Au[u] @ Au) + 1e-6 * deg
    return torch.tensor(S)


def rewire(E_in, seed):
    rng = random.Random(seed)
    out = {}
    chem = [(u, v, w) for (u, r, v), w in E_in.items() if r == 0]
    edges = [(u, v) for u, v, _ in chem]
    es = set(edges)
    for _ in range(10 * len(edges)):
        i, j = rng.randrange(len(edges)), rng.randrange(len(edges))
        (a, b), (c, d) = edges[i], edges[j]
        if len({a, b, c, d}) < 4 or (a, d) in es or (c, b) in es:
            continue
        es -= {(a, b), (c, d)}; es |= {(a, d), (c, b)}
        edges[i], edges[j] = (a, d), (c, b)
    for (u, v), (_, _, w) in zip(edges, chem):
        out[(u, 0, v)] = w
    G = nx.Graph()
    G.add_edges_from({(min(u, v), max(u, v)) for (u, r, v) in E_in if r == 1})
    if G.number_of_edges() > 3:
        nx.double_edge_swap(G, nswap=5 * G.number_of_edges(), max_tries=10 ** 6, seed=seed)
    for u, v in G.edges():
        out[(u, 1, v)] = 1; out[(v, 1, u)] = 1
    return out


def graph_tensors(E_in, device):
    src, dst, rel, w = [], [], [], []
    for (u, r, v), c in E_in.items():
        lw = math.log1p(c)
        if r == 0:
            src += [u, v]; dst += [v, u]; rel += [0, 1]; w += [lw, lw]
        else:
            src.append(u); dst.append(v); rel.append(2); w.append(lw)
    t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    return t(src, torch.long), t(dst, torch.long), t(rel, torch.long), t(w, torch.float)


class RGCN(nn.Module):
    def __init__(self, n, d=64, layers=2, mode="dense", n_msg_rel=3, drop=0.2):
        super().__init__()
        self.mode, self.emb = mode, nn.Embedding(n, d)
        self.W = nn.ModuleList([nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_msg_rel)]) for _ in range(layers)])
        self.self_ = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.rel_emb = nn.Embedding(n_msg_rel, 16)
        self.gate = nn.Sequential(nn.Linear(2 * d + 16 + 1, 64), nn.ReLU(), nn.Linear(64, 1))
        self.dec = nn.Parameter(torch.ones(2, d))
        self.drop = drop
        self.fixed_mask = None
        self.hw = nn.Parameter(torch.zeros(2, 4))
        nn.init.normal_(self.emb.weight, std=0.1)
        nn.init.constant_(self.gate[-1].bias, 2.0)

    def edge_gates(self, g):
        src, dst, rel, w = g
        if self.mode in ("dense", "nomp"):
            return torch.ones_like(w)
        if self.mode == "random":
            return self.fixed_mask
        h0 = self.emb.weight
        return torch.sigmoid(self.gate(torch.cat([h0[src], h0[dst], self.rel_emb(rel), w[:, None]], 1))).squeeze(1)

    def encode(self, g):
        h = self.emb.weight
        if self.mode == "nomp":
            return h, torch.ones(1, device=h.device)
        src, dst, rel, w = g
        gate = self.edge_gates(g)
        for W, S in zip(self.W, self.self_):
            agg = torch.zeros_like(h)
            deg = torch.zeros(h.size(0), device=h.device)
            for r in range(len(W)):
                m = rel == r
                msg = W[r](h[src[m]]) * (gate[m] * (1 + w[m]))[:, None]
                agg.index_add_(0, dst[m], msg)
                deg.index_add_(0, dst[m], gate[m])
            h = F.dropout(F.relu(S(h) + agg / deg.clamp(min=1)[:, None]), self.drop, self.training) + h
        return h, gate

    def score(self, h, queries, feats=None):
        u = torch.tensor([q[0] for q in queries], device=h.device)
        r = torch.tensor([q[1] for q in queries], device=h.device)
        s = (h[u] * self.dec[r]) @ h.T
        if feats is not None:
            s = s + (feats[u] * self.hw[r][:, None, :]).sum(-1)
        return s


def heuristic_feats(E_in, n, device):
    A = torch.zeros(2, n, n)
    for (u, r, v) in E_in:
        A[r, u, v] = 1
    Au = ((A[0] + A[0].T + A[1]) > 0).float()
    deg = Au.sum(1)
    f = torch.stack([A[0].T + A[1], torch.log1p(Au @ Au), (Au / torch.log(deg.clamp(min=2))) @ Au,
                     torch.log1p((A[0].sum(1) + A[1].sum(1))[:, None] * (A[0].sum(0) + A[1].sum(0))[None, :])], -1)
    f = (f - f.mean((0, 1))) / f.std((0, 1)).clamp(min=1e-6)
    return f.to(device)


def train_eval_gnn(nodes, E, mode, seed, device, rewired=False, lam=1e-2, epochs=300, keep_frac=None, hybrid=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    n = len(nodes)
    inp = {k: (rewire(E[k], seed * 100 + k) if rewired else E[k]) for k in STAGES}
    graphs = {k: graph_tensors(inp[k], device) for k in STAGES}
    feats = {k: heuristic_feats(inp[k], n, device) if hybrid else None for k in STAGES}
    Q = {t: new_edge_queries(E, t) for t in STAGES[1:]}
    X = {t: filter_mask(E, t, n) for t in STAGES[1:]}
    model = RGCN(n, mode=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)

    def masks(t):
        M = torch.zeros(len(Q[t]), n, dtype=torch.bool, device=device)
        for i, (u, r, v) in enumerate(Q[t]):
            M[i, list(X[t].get((u, r), set()) - {v}) + [u]] = True
        y = torch.tensor([q[2] for q in Q[t]], device=device)
        return M, y

    M = {t: masks(t) for t in STAGES[1:]}
    train_t, test_t = [2, 3, 4, 5], [7, 8]
    rmasks = ({k: (torch.rand(graphs[k][0].numel(), device=device) < keep_frac).float() for k in STAGES}
              if mode == "random" else None)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); opt.zero_grad(); loss = 0
        for t in train_t:
            if mode == "random":
                model.fixed_mask = rmasks[t - 1]
            h, gate = model.encode(graphs[t - 1])
            s = model.score(h, Q[t], feats[t - 1]).masked_fill(M[t][0], -1e9)
            loss = loss + F.cross_entropy(s, M[t][1])
            if mode == "selective":
                loss = loss + lam * gate.mean()
        loss.backward(); opt.step()
        if ep % 10 == 9:
            model.eval()
            with torch.no_grad():
                if mode == "random":
                    model.fixed_mask = rmasks[5]
                h, _ = model.encode(graphs[5])
                r = rank_scores(model.score(h, Q[6], feats[5]).cpu(), Q[6], X[6])
                mrr = float(np.mean(1 / r))
            if mrr > best:
                best, best_state = mrr, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    res, kept = {}, []
    with torch.no_grad():
        for t in test_t:
            if mode == "random":
                model.fixed_mask = rmasks[t - 1]
            h, gate = model.encode(graphs[t - 1])
            res[t] = metrics(rank_scores(model.score(h, Q[t], feats[t - 1]).cpu(), Q[t], X[t]))
            kept.append(float((gate > 0.5).float().mean()))
    return res, float(np.mean(kept)), best


def run_forecast(nodes, E, seeds, device, out):
    rows = []
    n = len(nodes)
    for t in [7, 8]:
        Q, X = new_edge_queries(E, t), filter_mask(E, t, n)
        for h in ["PA", "CN", "AA", "Recip"]:
            m = metrics(rank_scores(heuristic_scores(E[t - 1], Q, n, h), Q, X))
            rows.append(dict(model=h, rewired=False, seed=-1, stage=t, kept=1.0, **m))
    pd.DataFrame(rows).to_csv(os.path.join(out, "forecast_heuristics.csv"), index=False)
    names = {"nomp": "DistMult", "dense": "RGCN", "selective": "Selective-RGCN", "random": "RandomMask-RGCN"}
    for seed in seeds:
        rows, sel_keep = [], {}
        for hy in [False, True]:
            for mode in ["nomp", "dense", "selective", "random"]:
                for rw in [False, True]:
                    if mode == "nomp" and rw:
                        continue
                    t0 = time.time()
                    res, kept, val = train_eval_gnn(nodes, E, mode, seed, device, rewired=rw,
                                                    keep_frac=sel_keep.get((hy, rw)), hybrid=hy)
                    if mode == "selective":
                        sel_keep[(hy, rw)] = max(kept, 0.05)
                    for t, m in res.items():
                        rows.append(dict(model=names[mode] + ("+H" if hy else ""), rewired=rw, seed=seed,
                                         stage=t, kept=kept, val_mrr=val, **m))
                    print(f"[seed {seed}] {mode:9s} hybrid={hy!s:5s} rewired={rw!s:5s} kept={kept:.2f} "
                          f"MRR7={res[7]['MRR']:.3f} MRR8={res[8]['MRR']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, f"forecast_seed{seed}.csv"), index=False)


def artanh(x):
    x = x.clamp(-1 + 1e-5, 1 - 1e-5)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def expmap0(v, c):
    sc = c.sqrt()
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    return torch.tanh(sc * n) * v / (sc * n)


def mobius_add(x, y, c):
    xy = (x * y).sum(-1, keepdim=True); x2 = (x * x).sum(-1, keepdim=True); y2 = (y * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    return num / (1 + 2 * c * xy + c * c * x2 * y2).clamp_min(1e-10)


def pdist(x, y, c):
    sc = c.sqrt().squeeze(-1)
    return 2 / sc * artanh(sc * mobius_add(-x, y, c).norm(dim=-1))


class CurvKG(nn.Module):
    def __init__(self, n, n_rel=2, d=8):
        super().__init__()
        self.e = nn.Embedding(n, d); self.r = nn.Embedding(n_rel, d)
        self.b = nn.Embedding(n, 1); self.c_raw = nn.Parameter(torch.zeros(n_rel))
        nn.init.normal_(self.e.weight, std=1e-2); nn.init.normal_(self.r.weight, std=1e-2)
        nn.init.zeros_(self.b.weight)

    def curv(self):
        return F.softplus(self.c_raw) + 1e-3

    def forward(self, u, r, v):
        c = self.curv()[r][..., None]
        hu = mobius_add(expmap0(self.e(u), c), expmap0(self.r(r), c), c)
        hv = expmap0(self.e(v), c)
        return -pdist(hu, hv, c) ** 2 + self.b(u).squeeze(-1) + self.b(v).squeeze(-1)


def structure_stats(E_s, n, rng, samples=3000):
    st = {}
    for r in range(2):
        G = nx.DiGraph() if r == 0 else nx.Graph()
        G.add_edges_from([(u, v) for (u, rr, v) in E_s if rr == r])
        outd = [d for _, d in (G.out_degree() if r == 0 else G.degree()) if d > 0]
        U = G.to_undirected() if r == 0 else G
        cc = U.subgraph(max(nx.connected_components(U), key=len))
        nodes = list(cc.nodes()); sp = dict(nx.all_pairs_shortest_path_length(cc))
        deltas = []
        for _ in range(samples):
            a, b, c_, d = rng.sample(nodes, 4)
            s = sorted([sp[a][b] + sp[c_][d], sp[a][c_] + sp[b][d], sp[a][d] + sp[b][c_]])
            deltas.append((s[2] - s[1]) / 2)
        diam = max(max(x.values()) for x in sp.values())
        st[RELS[r]] = dict(fanout=float(np.mean(outd)), delta_max=float(np.max(deltas)),
                           delta_mean_rel=float(np.mean(deltas) / max(diam, 1)), n_edges=G.number_of_edges())
    return st


def run_geometry(nodes, E, seeds, device, out, dims=(4, 8, 16)):
    dims = tuple(int(x) for x in os.environ.get("GEOM_DIMS", ",".join(map(str, dims))).split(","))
    n, rows = len(nodes), []
    rng = random.Random(0)
    stats = {s: structure_stats(E[s], n, rng) for s in STAGES}
    for seed in seeds:
        for d in dims:
            for s in STAGES:
                torch.manual_seed(seed)
                trip = torch.tensor([k for k in E[s] if not (k[1] == 1 and k[0] > k[2])], device=device)
                m = CurvKG(n, d=d).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=1e-2)
                for ep in range(400):
                    u, r, v = trip[:, 0], trip[:, 1], trip[:, 2]
                    neg = torch.randint(0, n, (len(trip), 32), device=device)
                    pos = m(u, r, v)
                    ng = m(u[:, None].expand_as(neg), r[:, None].expand_as(neg), neg)
                    loss = F.softplus(-pos).mean() + F.softplus(ng).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                c = m.curv().detach().cpu().numpy()
                for r in range(2):
                    rows.append(dict(seed=seed, dim=d, stage=s, relation=RELS[r], curvature=float(c[r]),
                                     **stats[s][RELS[r]]))
            print(f"[geom seed {seed} d={d}] chem c={[round(x['curvature'],3) for x in rows[-16::2]]}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, f"geometry_seed{seed}.csv"), index=False)
        rows = []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exp", choices=["forecast", "geometry", "both"], default="both")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nodes, E = load(a.data)
    print(f"{len(nodes)} cells; edges per stage: {[len(E[s]) for s in STAGES]}; "
          f"new-edge queries: {[len(new_edge_queries(E, t)) for t in STAGES[1:]]}", flush=True)
    if a.exp in ("forecast", "both"):
        run_forecast(nodes, E, a.seeds, a.device, a.out)
    if a.exp in ("geometry", "both"):
        run_geometry(nodes, E, a.seeds, a.device, a.out)
