"""Budget figure: MRR vs fraction of edges kept, with DistMult / HeurLR reference lines.
  python make_fig_budget.py --refs   # trains DistMult(+H) and HeurLR on fixed split -> extra/ref_fixed.csv
  python make_fig_budget.py          # draws fig_budget.pdf / fig_budget.png"""
import os, sys, argparse
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_refs():
    sys.argv = ["x"]
    import torch, extra_exps as X
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nodes, E = X.load("data"); n = len(nodes)
    D = X.make_D(E, n, [2, 3, 4, 5, 6, 7, 8], dev)
    a = argparse.Namespace(dim=64, lr=0.01, epochs=300, lam=0.05)
    rows = []
    for seed in range(10):
        for h in (0, 1):
            rr, _ = X.run_model("distmult", h, D, [2, 3, 4, 5], 6, [7, 8], seed, dev, a); rows += rr
        rows += X.heur_lr_rows(D, [2, 3, 4, 5], 6, [7, 8], seed, a)
        print(f"refs seed {seed} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv("extra/ref_fixed.csv", index=False)
    print(df.groupby("model").mrr.agg(["mean", "std"]).round(4))


if "--refs" in sys.argv:
    make_refs(); sys.exit(0)

df = pd.concat([pd.read_csv("extra/budget_h0.csv"), pd.read_csv("extra/budget_h1.csv")], ignore_index=True)
ref = pd.read_csv("extra/ref_fixed.csv") if os.path.exists("extra/ref_fixed.csv") else None
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
style = {("selective", 0): dict(color="#1f4e79", ls="-", marker="o", label="Learned gates, real"),
         ("random", 0): dict(color="#1f4e79", ls="--", marker="s", label="Random mask, real"),
         ("selective", 1): dict(color="#9dc3e6", ls="-", marker="o", label="Learned gates, rewired"),
         ("random", 1): dict(color="#9dc3e6", ls="--", marker="s", label="Random mask, rewired")}
for ax, h in zip(axes, (0, 1)):
    sub = df[df.use_h == h]
    for (kind, rw), st in style.items():
        g = sub[(sub.kind == kind) & (sub.rewired == rw)].groupby("q").mrr
        m, se = g.mean(), g.std() / np.sqrt(g.count())
        ax.errorbar(m.index, m.values, yerr=se.values, ms=3.5, lw=1.4, capsize=2, **st)
    if ref is not None:
        dm = ref[ref.model == ("distmult+H" if h else "distmult")].mrr.mean()
        hl = ref[ref.model == "HeurLR"].mrr.mean()
        ax.axhline(dm, color="#7f7f7f", ls=":", lw=1.3, label="DistMult (no message passing)")
        ax.axhline(hl, color="#c55a11", ls="-.", lw=1.3, label="HeurLR (heuristics only)")
    ax.set_title("R-GCN" if h == 0 else "R-GCN + heuristic features", fontsize=9)
    ax.set_xlabel("fraction of edges kept", fontsize=8.5)
    ax.tick_params(labelsize=7.5); ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("MRR (new synapses)", fontsize=8.5)
h0, l0 = axes[0].get_legend_handles_labels()
l0 = [("DistMult(+H) (no message passing)" if l.startswith("DistMult") else l) for l in l0]
fig.legend(h0, l0, fontsize=6.8, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09))
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig("fig_budget.pdf", bbox_inches="tight"); fig.savefig("fig_budget.png", dpi=200, bbox_inches="tight")
print("wrote fig_budget.pdf / fig_budget.png" + ("" if ref is not None else "  (no reference lines: run with --refs first)"))
