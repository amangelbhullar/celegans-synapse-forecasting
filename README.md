# What Predicts the Next Synapse? Local Wiring Rules versus Graph Neural Networks across *C. elegans* Development

Code for the NeuroAI 2.0 @ AAAI-27 submission. The eight stage-wise *C. elegans* connectomes of Witvliet et al. (2021) are treated as a temporal knowledge graph: neurons are entities, chemical synapses and gap junctions are relations, and developmental stages are timestamps. The task is to forecast which synapses appear at the next stage.

## Contents

| File | Purpose |
|---|---|
| `connectome_tkg.py` | Implementation A: initial pipeline (heuristics, DistMult, R-GCN, regularised gates, geometry probe) |
| `extra_exps.py` | Implementation B: budget sweep, rolling-origin evaluation, gate analysis, geometry probe with controls |
| `extra2.py` | Contact ablation; architecture controls (residual, 1-layer); fixed vs. resampled random masks |
| `extra3.py` + `hardconcrete_gate.py` | Hard-concrete (L0) gate vs. density-matched random masks |
| `make_fig_budget.py` | Figure 1 (MRR vs. edge budget, with DistMult / HeurLR reference lines) |
| `make_appendix.py` | Generates the appendix tables from result CSVs |
| `results/` | Result CSVs and summary files behind every number in the paper |

## Data

Download the eight stage-wise connectivity tables of Witvliet et al. (2021), *Connectomes across development reveal principles of brain maturation*, Nature 596:257–261 (supplementary data). Save them as `data/witvliet_2020_{1..8}.xlsx`, each with columns `pre, post, type, synapses`. The data are not redistributed here.

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing the paper

```bash
python extra_exps.py --mode budget  --use_h 0 --device cuda:0 --out results
python extra_exps.py --mode budget  --use_h 1 --device cuda:1 --out results
python extra_exps.py --mode rolling           --device cuda:2 --out results
python extra_exps.py --mode gates             --device cuda:0 --out results
python extra_exps.py --mode geometry --dims 8 16 --geoms hyp euc --tag main --out results
python extra_exps.py --mode geometry --dims 4 --geoms hyp euc --seeds $(seq -s ' ' 0 19) --tag d4 --out results
python extra_exps.py --mode geometry --graphs rewired --dims 8 16 --tag rewired --out results
python extra_exps.py --mode geometry --dims 8 16 --c_init 0.5 --tag init05 --out results
python extra2.py --mode contact --out results
python extra2.py --mode arch --cfgs base res l1 --out results
python extra3.py --mode run --out results
python extra_exps.py --mode summary --out results && python extra_exps.py --mode geosummary --out results
python make_fig_budget.py --refs && python make_fig_budget.py && python make_appendix.py
```

All learned models use 10 seeds (20 for the d=4 geometry probe). Ranking is filtered and tie-aware: `mrr` uses mean ranks for ties, and `mrr_opt` uses optimistic ranks.

## License

MIT (code only; the connectome data remain under their original terms).
