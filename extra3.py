#!/usr/bin/env python
"""HardConcrete-gate control: the published hard-concrete SparsityController (sd_gnn/sdgnn.py) as the edge gate of the
same R-GCN, vs density-matched random masks (resampled and fixed). Fixed split (train 2-5, val 6, test 7-8)."""
import sys, os, glob, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
import extra_exps as X
import extra2 as X2
from hardconcrete_gate import SparsityController


class SDNet(X2.Net2):
    def __init__(self, n, d, kind, use_h, tau=0.5, **kw):
        super().__init__(n, d, "rgcn" if kind == "sdgnn" else kind, use_h, **kw)
        self.is_sd, self.tau = kind == "sdgnn", tau
        if self.is_sd:
            self.ctrl = SparsityController(d)
            self.rel_emb = nn.Embedding(3, d)

    def edge_w(self, g, q, gen):
        if not self.is_sd:
            return super().edge_w(g, q, gen)
        e = self.emb.weight
        out = self.ctrl(e[g["src"]], self.rel_emb(g["typ"]), e[g["dst"]], tau=self.tau, hard=True, train_mode=self.training)
        mask, prob = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, torch.sigmoid(out))
        return mask.view(-1).float(), prob.view(-1).float()


def run(a):
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    nodes, E = X.load(a.data); n = len(nodes)
    tr, va, te = [2, 3, 4, 5], 6, [7, 8]
    D = X.make_D(E, n, tr + [va] + te, dev)
    args = argparse.Namespace(dim=64, lr=0.01, epochs=a.epochs, lam=a.lam)
    os.makedirs(a.out, exist_ok=True)
    rows, path = [], os.path.join(a.out, "hardconcrete.csv")
    base = X.Net
    for seed in a.seeds:
        X.Net = lambda n_, d_, kind, use_h, _s=seed: SDNet(n_, d_, kind, use_h, tau=a.tau, fseed=_s, **X2.CFGS["base"])
        for h in a.use_h:
            rr, _ = X.run_model("sdgnn", h, D, tr, va, te, seed, dev, args)
            for r in rr: r["model"] = "HardConcrete-gate" + ("+H" if h else "")
            rows += rr
            kept = float(np.clip(np.mean([r["kept"] for r in rr]), 0.02, 1.0))
            for kind, name in (("random", "RandomMatched"), ("randomfixed", "RandomFixedMatched")):
                rr2, _ = X.run_model(kind, h, D, tr, va, te, seed, dev, args, q=kept)
                for r in rr2: r["model"] = name + ("+H" if h else "")
                rows += rr2
        X.Net = base
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[sdgnn] seed={seed} done  (last kept={kept:.3f})", flush=True)


def summary(a):
    df = pd.read_csv(os.path.join(a.out, "hardconcrete.csv"))
    pd.set_option("display.width", 200)
    print("===== Hard-concrete gate (fixed split, stages 7-8) =====")
    print(df.groupby("model")[["mrr", "kept"]].agg(["mean", "std"]).round(4))
    for h in sorted(df.use_h.unique()):
        sfx = "+H" if h else ""
        def v(m): return df[df.model == m + sfx].sort_values(["seed", "stage"]).mrr.values
        s = v("HardConcrete-gate")
        for m in ("RandomMatched", "RandomFixedMatched"):
            r = v(m)
            if len(s) == len(r) and len(s):
                print(f"  H={h}  HardConcrete-gate vs {m:18s}: dMRR={s.mean() - r.mean():+.4f}  p={X.wil(s, r):.3g}  n={len(s)}")
    ref = "extra/ref_fixed.csv"
    if os.path.exists(ref):
        rf = pd.read_csv(ref)
        print("  reference (same split): DistMult=%.4f  DistMult+H=%.4f  HeurLR=%.4f" % tuple(
            rf[rf.model == m].mrr.mean() for m in ("distmult", "distmult+H", "HeurLR")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["run", "summary"])
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="extra3")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--use_h", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    run(a) if a.mode == "run" else summary(a)
