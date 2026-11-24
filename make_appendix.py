"""Generate appendix_tables.tex from the result CSVs. Run from aaai27_ws/neuroai:  python make_appendix.py"""
import os, glob, sys
import numpy as np, pandas as pd

sys.argv = sys.argv[:1]
OUT = "appendix_tables.tex"
parts = []


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}".replace("0.", ".", 1)


def table(caption, label, colspec, header, rows, star=False, size="\\small"):
    env = "table*" if star else "table"
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    return (f"\\begin{{{env}}}[t]\n\\centering\n{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


# ---------------- A. dataset statistics
try:
    import extra_exps as X
    nodes, E = X.load("data")
    rows = []
    for s in X.STAGES:
        chem = sum(1 for (u, r, v) in E[s] if r == 0)
        elec = sum(1 for (u, r, v) in E[s] if r == 1 and u < v)
        syn = sum(w for (u, r, v), w in E[s].items() if r == 0 or u < v)
        if s > 1:
            q = X.new_edge_queries(E, s)
            nc, ne = sum(1 for x in q if x[1] == 0), sum(1 for x in q if x[1] == 1)
        else:
            nc = ne = "--"
        rows.append([s, f"{chem:,}", f"{elec:,}", f"{syn:,}", f"{nc:,}" if s > 1 else nc, f"{ne:,}" if s > 1 else ne])
    parts.append(table(
        "Dataset statistics per developmental stage: chemical synapse pairs (directed), gap-junction pairs (undirected), "
        "total synapse count, and the number of \\emph{new} chemical and electrical synapses used as forecasting queries "
        f"({len(nodes)} neurons in total).",
        "tab:app-data", "rrrrrr",
        "Stage & Chem.\\ pairs & Gap-junction pairs & Synapses & New chem. & New elec.", rows))
except Exception as e:
    print("skip dataset stats:", e)

# ---------------- B. full budget sweep
fs = [f for f in ("extra/budget_h0.csv", "extra/budget_h1.csv") if os.path.exists(f)]
if fs:
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    rows = []
    for h in sorted(df.use_h.unique()):
        for rw in (0, 1):
            for kind, name in (("selective", "Learned gate"), ("random", "Random mask")):
                sub = df[(df.use_h == h) & (df.rewired == rw) & (df.kind == kind)]
                if len(sub) == 0:
                    continue
                g = sub.groupby("q").mrr
                cells = [f"{f3(m)}$\\pm${f3(s)}" for m, s in zip(g.mean(), g.std())]
                rows.append([("yes" if h else "no"), ("rewired" if rw else "real"), name] + cells)
    qs = sorted(df.q.unique())
    parts.append(table(
        "Full edge-budget sweep (fixed split, stages 7--8; mean$\\pm$std over 10 seeds $\\times$ 2 stages). Columns give the fraction of edges kept.",
        "tab:app-budget", "lll" + "c" * len(qs),
        "+H & Input & Mask & " + " & ".join(f"$q{{=}}{q:g}$" for q in qs), rows, star=True, size="\\scriptsize"))

# ---------------- C. full rolling results
if os.path.exists("extra/rolling.csv"):
    df = pd.read_csv("extra/rolling.csv")
    order = ["PA", "CN", "AA", "Recip", "HeurLR", "distmult", "rgcn", "selective", "RandomMatched",
             "distmult+H", "rgcn+H", "selective+H", "RandomMatched+H"]
    nice = {"Recip": "Binary reciprocity", "distmult": "DistMult", "rgcn": "R-GCN", "selective": "Selective-RGCN (40\\%)",
            "RandomMatched": "RandomMask (40\\%)", "distmult+H": "DistMult+H", "rgcn+H": "R-GCN+H",
            "selective+H": "Selective-RGCN+H", "RandomMatched+H": "RandomMask+H"}
    rows = []
    for m in order:
        sub = df[df.model == m]
        if len(sub) == 0:
            continue
        cells = []
        for t in (5, 6, 7, 8):
            x = sub[sub.stage == t]
            cells.append(f3(x.mrr.mean()) + ("" if x.seed.iloc[0] == -1 or len(x) < 2 else f"$\\pm${f3(x.mrr.std())}"))
        h10 = [f3(sub[sub.stage == t].h10.mean()) for t in (5, 6, 7, 8)]
        rows.append([nice.get(m, m)] + cells + h10)
        if m == "Recip" and os.path.exists("extra2/contact.csv"):
            c = pd.read_csv("extra2/contact.csv")
            c = c[(c.score == "Recip (reverse chem + gap junction)") & (c.queries == "all")]
            if len(c):
                rows.append(["Recip (reciprocity, CN tie-break)"] + [f3(c[c.stage == t].mrr.mean()) for t in (5, 6, 7, 8)]
                            + [f3(c[c.stage == t].h10.mean()) for t in (5, 6, 7, 8)])
    parts.append(table(
        "Rolling-origin results: MRR (mean$\\pm$std over 10 seeds; heuristics are deterministic) and Hits@10 for each held-out stage.",
        "tab:app-rolling", "l" + "c" * 8,
        "Model & MRR $T{=}5$ & 6 & 7 & 8 & H@10 $T{=}5$ & 6 & 7 & 8", rows, star=True, size="\\scriptsize"))
    if "weights" in df.columns:
        w = df[df.model == "HeurLR"].dropna(subset=["weights"])
        w = pd.concat([w[["stage"]].reset_index(drop=True),
                       w.weights.str.split(";", expand=True).astype(float).reset_index(drop=True)], axis=1)
        w.columns = ["stage", "PA", "CN", "AA", "Recip"]
        g = w.groupby("stage").mean()
        rows = [[int(t)] + [f"{v:+.2f}" for v in g.loc[t]] for t in g.index]
        parts.append(table("Learned HeurLR weights per test stage (mean over 10 seeds; features: log PA, log CN, AA, binary reciprocity).",
                           "tab:app-heurlr", "lcccc", "Test stage & PA & CN & AA & Reciprocity", rows))

# ---------------- D. gate breakdown
if os.path.exists("extra/gates.csv"):
    df = pd.read_csv("extra/gates.csv")
    if "kept" not in df.columns:
        df["kept"] = (df.gate > 0.5).astype(int)
    df["count_q"] = df.groupby(["use_h", "seed", "input_stage"])["count"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=False) + 1)
    df["persist"] = np.where(df.persistence >= df.input_stage, "all prior stages",
                             np.where(df.persistence == 1, "new at input stage", "intermittent"))
    groups = [("relation", "Relation"), ("count_q", "Count quartile"), ("persist", "Persistence"),
              ("reciprocated", "Reciprocated"), ("survives", "Survives to next stage")]
    rows = []
    for col, lab in groups:
        first = True
        for val, sub in df.groupby(col):
            k = [f3(sub[sub.use_h == h].kept.mean()) for h in (0, 1)]
            gm = [f3(sub[sub.use_h == h].gate.mean()) for h in (0, 1)]
            v = {0: "no", 1: "yes"}.get(val, val) if col in ("reciprocated", "survives") else (f"Q{val}" if col == "count_q" else val)
            rows.append([lab if first else "", v, k[0], gm[0], k[1], gm[1], f"{len(sub) // 2:,}"])
            first = False
    parts.append(table(
        "Complete gate breakdown (Selective-RGCN at a 40\\% budget; inputs at stages 6 and 7; 10 seeds). Kept = fraction of synapses retained; gate = mean gate score.",
        "tab:app-gates", "llccccr",
        "Group & Value & Kept & Gate & Kept (+H) & Gate (+H) & $n$", rows))

# ---------------- E. learned curvature per stage
gf = glob.glob("extra/geometry_*.csv")
if gf:
    g = pd.concat([pd.read_csv(f) for f in gf], ignore_index=True)
    g["tag"] = g.tag.replace({"d4": "main"})
    h = g[(g.geom == "hyp") & (g.tag == "main") & (g.graph == "real")]
    if len(h):
        agg = h.groupby(["dim", "relation", "stage"]).curv.agg(["mean", "std"]).reset_index()
        fan = h.groupby(["relation", "stage"]).fanout.first()
        rows = []
        for s in sorted(agg.stage.unique()):
            r = [int(s), f"{fan.loc[('chemical', s)]:.1f}", f"{fan.loc[('electrical', s)]:.1f}"]
            for d in sorted(agg.dim.unique()):
                for rel in ("chemical", "electrical"):
                    x = agg[(agg.dim == d) & (agg.relation == rel) & (agg.stage == s)]
                    r.append(f3(x["mean"].iloc[0]) if len(x) else "--")
            rows.append(r)
        dims = sorted(agg.dim.unique())
        hdr = "Stage & Fan-out chem. & Fan-out elec. & " + " & ".join(f"$d{{=}}{d}$ chem. & $d{{=}}{d}$ elec." for d in dims)
        parts.append(table("Learned curvature per stage and relation (real graphs, initial curvature 0.05; mean over seeds), with mean fan-out.",
                           "tab:app-curv", "rcc" + "cc" * len(dims), hdr, rows, star=True, size="\\scriptsize"))

open(OUT, "w").write("% auto-generated by make_appendix.py\n" + "\n".join(parts))
print(f"wrote {OUT} with {len(parts)} tables")
