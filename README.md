# Urban Transport Emergence Simulation

A minimal simulation showing how fast-transit corridors — analogous to train lines — self-organise on a city grid through repeated optimisation of random journeys.

## How it works

A city is modelled as an **N×N lattice graph**. Each edge carries a travel-time weight (default 1). The simulation runs a loop:

1. Pick two random nodes (origin and destination).
2. Compute the **shortest weighted path** between them (Dijkstra).
3. Pick a **random edge on that path** with current weight *v*; reduce it to `max(v − 1, 0)`.
4. Pick a **random edge anywhere on the grid** that has room to grow (weight < cap) and increase it by the same amount, conserving the total weight.

Over many iterations, weight drains towards the most-used corridors, eventually forming zero-weight edges — **train tracks** — that short-circuit large parts of the city.

## Output

An animated figure with three panels:

| Panel | Description |
|-------|-------------|
| **City network** | Grid drawn in real time; black thick edges = weight 0 (train track), light edges = normal roads |
| **Average path length** | Running average of the shortest-path times computed during optimisation |
| **Total weight sum** | Should remain constant throughout — verifying conservation |

## Requirements

```
pip install networkx matplotlib numpy
```

## Usage

```bash
python simulate.py [--dim N] [--cost C] [--iterations K] [--frames F]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dim` | 20 | Grid side length (city is N×N nodes) |
| `--cost` | 1.0 | Initial travel cost per edge (any positive number) |
| `--iterations` | 10000 | Number of optimisation steps |
| `--frames` | 200 | Number of animation frames |

### Examples

```bash
# Default 20×20 city, 10 000 steps
python simulate.py

# Larger city, more iterations
python simulate.py --dim 30 --iterations 50000

# Non-unit initial cost
python simulate.py --dim 15 --cost 2.0 --iterations 20000
```

## Notes

- **Conservation**: the total weight is conserved exactly. The random edge chosen to absorb weight is always restricted to edges not already at maximum, guaranteeing no weight is lost.
- **Emergence**: with enough iterations relative to grid size (roughly `iterations ≳ N²`), clear linear corridors appear — qualitatively similar to the Physarum polycephalum (slime-mould) transport network model.
