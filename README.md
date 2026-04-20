# Urban Transport Emergence Simulation

A simulation of self-organising transport networks on a 2D city grid. Zero-weight edges — *train tracks* — emerge from repeated optimisation of random journeys, without any global planner.

## Algorithm

The city is an **N×N lattice graph**. Each edge carries a travel-time weight (default 1). A fraction of edges are pre-set to weight 0 (`--track`), representing randomly placed initial track. The total weight across all edges is conserved throughout.

Each iteration:

1. Sample `--mutual` independent origin-destination pairs.
2. Find the shortest weighted path between each pair (Dijkstra).
3. Take the **intersection** of all path edge sets — edges shared by every path simultaneously.
4. From this intersection, either decrease **one** random edge (`--all 0`) or **every** edge (`--all 1`) by 1, floored at 0.
5. To conserve total weight, increment the same number of randomly chosen edges that have room to grow (weight < cap) by 1 each.

Over many iterations, edges that appear on many shortest paths simultaneously accumulate weight reductions and eventually reach 0 — forming persistent fast-transit corridors.

## Physical interpretation

### Kawasaki dynamics

The update rule is a form of **conserved-order-parameter dynamics**: weight is neither created nor destroyed, only redistributed. This is structurally identical to Kawasaki spin-exchange dynamics (Kawasaki, *Phys. Rev.* **145**, 224, 1966), except the moves are always accepted when they reduce path length (T = 0, greedy descent). The system cannot escape local minima by thermal fluctuation, so different random initial track placements (`--track`) lead to different metastable attractors.

### Edge betweenness centrality

The `--mutual N` parameter computes an on-the-fly proxy for **edge betweenness centrality**: an edge scores if it lies on the shortest path for multiple independent journeys simultaneously. Only edges passing this threshold are reinforced. This is precisely the selection pressure that concentrates weight reductions into high-traffic corridors rather than diffusing them uniformly.

### Relation to biological transport networks

The closest analogue in the literature is the *Physarum polycephalum* (slime mould) model. The organism thickens tubes that carry high flux and withdraws those with little flow, converging to efficient spanning networks:

> Tero, A., et al. "Rules for biologically inspired adaptive network design." *Science* **327**, 439–442 (2010).

A mathematical analysis of the underlying ODE system is in:

> Hu, D. & Cai, D. "Adaptation and optimization of biological transport networks." *Phys. Rev. Lett.* **111**, 138701 (2013).

### Loops vs trees: fluctuating demand

For fixed demand, optimal transport networks under concave cost functions are trees (Steiner trees), with no redundant loops — see the branched-transport literature:

> Xia, Q. "Optimal paths related to transport problems." *Commun. Contemp. Math.* **5**, 251 (2003).

However, when demand **fluctuates** (as here, with random O-D pairs each iteration), loops emerge as optimal because a single edge can serve multiple flow patterns. This was shown for vascular networks in:

> Katifori, E., Szöllősi, G. J. & Magnasco, M. O. "Damage and fluctuations induce loops in optimal transport networks." *Phys. Rev. Lett.* **104**, 048704 (2010).

### Expected network structure

| Regime | Structure |
|---|---|
| Low track density, `--mutual 1` | Sparse branching tree, fractal-like |
| Low track density, `--mutual` > 1 | Looped hierarchical network |
| High track density | Approximately regular grid |

At intermediate densities, allometric scaling (trunk lines feeding finer branches) is expected, consistent with:

> Banavar, J. R., Maritan, A. & Rinaldo, A. "Size and form in efficient transportation networks." *Nature* **399**, 130–132 (1999).

## Requirements

```
pip install numpy scipy matplotlib
```

## Usage

```bash
python simulate.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dim` | 20 | Grid side length N (city is N×N nodes) |
| `--cost` | 1.0 | Initial travel cost per edge |
| `--track` | 0.1 | Fraction of edges pre-set to weight 0 (initial random track) |
| `--all` | 0 | `0`: decrease one shared edge per iteration; `1`: decrease all shared edges |
| `--mutual` | 1 | Number of simultaneous paths; only edges shared by all are reinforced |
| `--iterations` | 10000 | Number of optimisation steps |
| `--frames` | 200 | Animation frames |

## Examples

```bash
# Default
python simulate.py

# Large city, aggressive optimisation
python simulate.py --dim 100 --iterations 10000 --track 0.2 --all 1 --mutual 2

# Observe loop formation with fluctuating demand
python simulate.py --dim 30 --mutual 3 --all 1 --track 0.15 --iterations 20000
```
