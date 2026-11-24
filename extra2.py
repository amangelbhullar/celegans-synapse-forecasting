#!/usr/bin/env python
"""Follow-up experiments (imports extra_exps.py; run from aaai27_ws/neuroai).
  contact : multiplex-contact ablation of the Recip rule + exact random-guess MRR baseline (CPU, seconds)
  arch    : fixed split, budgets q in {0.1,0.4,1.0}: residual+dropout R-GCN, 1-layer R-GCN,
            and a FIXED random mask (one mask for the whole run, as in Implementation A)
  summary : tables + paired tests for arch
"""
import sys, os, glob, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import extra_exps as X

BaseNet = X.Net


class Net2(BaseNet):
    def __init__(self, n, d, kind, use_h, nl=2, res=False, drop=0.0, fseed=0):
        super().__init__(n, d, "random" if kind == "randomfixed" else kind, use_h, nl)
        self.fixed, self.res, self.drop, self.fseed, self._masks = kind == "randomfixed", res, drop, fseed, {}

    def edge_w(self, g, q, gen):
        if self.fixed:                      # one mask per stage, drawn once, reused in training and evaluation
            key = id(g)
            if key not in self._masks:
                m = g["src"].numel(); k = max(1, int(round(q * m)))
                w = torch.zeros(m, device=g["src"].device)
                w[torch.randperm(m, generator=torch.Generator().manual_seed(self.fseed * 7919 + m))[:k].to(w.device)] = 1.0
                self._masks[key] = w
            return self._masks[key], None
        return super().edge_w(g, q, gen)

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
                out = out.index_add(0, d_, self.rel[l][tt](h[g["src"][mm]]) * w[mm, None] / g["deg"][tt][d_][:, None])
            if self.res:
                h = F.dropout(F.relu(out), self.drop, self.training) + h
            else:
                h = F.relu(out) if l < self.nl - 1 else out
        return h


CFGS = {"base": dict(nl=2, res=False, drop=0.0), "res": dict(nl=2, res=True, drop=0.1), "l1": dict(nl=1, res=False, drop=0.0)}


def mode_contact(a):
    nodes, E = X.load(a.data); n = len(nodes)
    rows = []
    for T in (5, 6, 7, 8):
        G = E[T - 1]
        g = X.build(E, T, n, G, torch.device("cpu"))
        A = torch.zeros(2, n, n)
        for (u, r, v) in G:
            A[r, u, v] = 1.0
        U = ((A.sum(0) + A.sum(0).T) > 0).float(); deg = U.sum(1); CN = U @ U
        qu, qr = g["qu"], g["qr"]
        rev_chem = A[0].T[qu]                     # v->u chemical exists
        elec = A[1][qu]                           # u-v gap junction exists
        same_dir_other = A[0][qu]                 # u->v chemical exists (relevant for new gap junctions)
        tb = 1e-3 * CN[qu] + 1e-6 * deg[None, :]
        variants = {
            "reverse chemical only": rev_chem + tb,
            "gap junction only": elec + tb,
            "any prior chemical contact (u->v or v->u)": rev_chem + same_dir_other + tb,
            "Recip (reverse chem + gap junction)": rev_chem + elec + tb,
            "CN + degree tie-break only": tb,
        }
        for name, S in variants.items():
            for rel in (0, 1, None):
                m = torch.ones_like(qr, dtype=torch.bool) if rel is None else (qr == rel)
                if int(m.sum()) == 0:
                    continue
                gg = {k: (v[m] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == qr.shape[0] else v) for k, v in g.items()}
                r_ = X.metrics(S[m].masked_fill(gg["M"], -1e9), gg)
                rows.append(dict(stage=T, score=name, queries=["chemical", "electrical", "all"][2 if rel is None else rel],
                                 n=int(m.sum()), mrr=r_["mrr"], h10=r_["h10"]))
        k = (~g["M"]).sum(1).double()                        # candidates per query
        H = torch.tensor([np.sum(1.0 / np.arange(1, int(x) + 1)) for x in k.tolist()], dtype=torch.double)
        rows.append(dict(stage=T, score="random guess (expected)", queries="all", n=len(k), mrr=float((H / k).mean()),
                         h10=float(torch.clamp(10 / k, max=1).mean())))
        # fraction of new synapses that had prior contact
        rows.append(dict(stage=T, score="share of new synapses with prior contact", queries="all", n=len(k),
                         mrr=float(((rev_chem + elec + same_dir_other).gather(1, g["qv"][:, None]) > 0).float().mean()), h10=np.nan))
    df = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True); df.to_csv(os.path.join(a.out, "contact.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_columns", 20)
    print("MRR (tie-aware) by test stage; 'share' rows report a fraction, not MRR")
    print(df.pivot_table(index=["queries", "score"], columns="stage", values="mrr").round(4))


def mode_arch(a):
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    nodes, E = X.load(a.data); n = len(nodes)
    tr, va, te = [2, 3, 4, 5], 6, [7, 8]
    D = X.make_D(E, n, tr + [va] + te, dev)
    args = argparse.Namespace(dim=64, lr=0.01, epochs=a.epochs, lam=0.05)
    rows, path = [], os.path.join(a.out, f"arch_{'_'.join(a.cfgs)}_h{''.join(map(str, a.use_h))}.csv")
    for cfg in a.cfgs:
        for seed in a.seeds:
            X.Net = lambda n_, d_, kind, use_h, _c=CFGS[cfg], _s=seed: Net2(n_, d_, kind, use_h, fseed=_s, **_c)
            for h in a.use_h:
                rr, _ = X.run_model("rgcn", h, D, tr, va, te, seed, dev, args)
                for r in rr: r.update(cfg=cfg, q=1.0)
                rows += rr
                for q in a.budgets:
                    kinds = ("selective", "random", "randomfixed") if q < 1 else ("selective",)
                    for kind in kinds:
                        rr, _ = X.run_model(kind, h, D, tr, va, te, seed, dev, args, q=q)
                        for r in rr: r.update(cfg=cfg)
                        rows += rr
            X.Net = BaseNet
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"[arch] cfg={cfg} seed={seed} done", flush=True)


def wil(x, y):
    return X.wil(x, y)


def mode_summary(a):
    fs = glob.glob(os.path.join(a.out, "arch_*.csv"))
    if not fs:
        print("no arch CSVs"); return
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pd.set_option("display.width", 200)
    print("===== ARCH: MRR mean (fixed split, stages 7-8) =====")
    print(df.pivot_table(index=["cfg", "use_h", "kind"], columns="q", values="mrr", aggfunc="mean").round(4))
    print("\nPaired tests (seed x stage):")
    for (cfg, h, q), s in df[df.q < 1].groupby(["cfg", "use_h", "q"]):
        def v(k): return s[s.kind == k].sort_values(["seed", "stage"]).mrr.values
        for k1, k2 in [("selective", "random"), ("selective", "randomfixed"), ("random", "randomfixed")]:
            x, y = v(k1), v(k2)
            if len(x) and len(x) == len(y):
                print(f"  cfg={cfg:4s} H={h} q={q:.1f}  {k1:10s} vs {k2:11s}: dMRR={x.mean() - y.mean():+.4f}  p={wil(x, y):.3g}  n={len(x)}")
    ref = "extra/ref_fixed.csv"
    if os.path.exists(ref):
        r = pd.read_csv(ref)
        print("\nFull-graph R-GCN (q=1) vs DistMult (no message passing), same split:")
        for (cfg, h), s in df[(df.kind == "rgcn")].groupby(["cfg", "use_h"]):
            dm = r[r.model == ("distmult+H" if h else "distmult")].sort_values(["seed", "stage"])
            x = s.sort_values(["seed", "stage"])
            if len(dm) == len(x):
                print(f"  cfg={cfg:4s} H={h}: rgcn={x.mrr.mean():.4f}  distmult={dm.mrr.mean():.4f}  p={wil(x.mrr.values, dm.mrr.values):.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["contact", "arch", "summary"])
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="extra2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--use_h", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.4])
    ap.add_argument("--cfgs", nargs="+", default=["base", "res", "l1"])
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    {"contact": mode_contact, "arch": mode_arch, "summary": mode_summary}[a.mode](a)
