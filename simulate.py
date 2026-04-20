#!/usr/bin/env python3
"""
Urban transport emergence simulation on a 2D grid graph.

A city is represented as an N×N lattice. Each edge carries a travel-time
weight. Repeated optimisation of random origin-destination journeys causes
zero-weight corridors — analogous to train lines — to self-organise.
"""

import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec

# Small offset so no explicit zeros appear in the CSR matrix.
# scipy.sparse.csgraph treats structural zeros as absent edges;
# adding _EPS makes every edge visible while leaving path selection unchanged
# (max correction per path = 400 * 1e-9 << 1 for any realistic grid).
_EPS = 1e-9


class GridGraph:
    """
    Fixed-topology N×N grid backed by a scipy CSR sparse matrix.

    Edge weights live in a plain numpy array `self.weights`; the CSR
    data array is kept in sync via O(1) in-place updates, so repeated
    Dijkstra calls never rebuild the matrix from scratch.
    """

    def __init__(self, n: int, cost: float, track_fraction: float):
        self.n = n
        self.N = n * n
        self.cap = float(cost)

        # Canonical edge list: u < v, row-major node index = r*n + c
        edges = []
        for r in range(n):
            for c in range(n):
                u = r * n + c
                if c + 1 < n:
                    edges.append((u, u + 1))       # horizontal
                if r + 1 < n:
                    edges.append((u, u + n))        # vertical

        self.n_edges = len(edges)
        self.edges = np.asarray(edges, dtype=np.int32)   # (n_edges, 2)
        self.weights = np.full(self.n_edges, self.cap, dtype=np.float64)

        # Randomly pre-place track
        n_track = round(track_fraction * self.n_edges)
        if n_track > 0:
            self.weights[np.random.choice(self.n_edges, n_track, replace=False)] = 0.0

        # (u, v) with u < v  →  edge index (for path reconstruction)
        self._lookup: dict[tuple[int, int], int] = {
            (int(u), int(v)): i for i, (u, v) in enumerate(edges)
        }

        # Build symmetric CSR (each undirected edge stored as u→v and v→u)
        u_arr = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        v_arr = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        w_arr = np.concatenate([self.weights, self.weights]) + _EPS
        self._csr = csr_matrix(
            (w_arr, (u_arr, v_arr)), shape=(self.N, self.N), dtype=np.float64
        )
        self._csr.sort_indices()

        # Pre-compute the position of every edge's two entries in csr.data
        # so set_weight() can update in O(1) without any search.
        self._pos = self._build_data_positions()

    def _build_data_positions(self) -> np.ndarray:
        pos = np.empty((self.n_edges, 2), dtype=np.int32)
        inds = self._csr.indices
        iptr = self._csr.indptr
        for i, (u, v) in enumerate(self.edges):
            s = iptr[u]
            pos[i, 0] = s + np.searchsorted(inds[s:iptr[u + 1]], v)
            s = iptr[v]
            pos[i, 1] = s + np.searchsorted(inds[s:iptr[v + 1]], u)
        return pos

    # ------------------------------------------------------------------ #

    def set_weight(self, idx: int, w: float) -> None:
        """Update one edge weight in both self.weights and the CSR matrix."""
        self.weights[idx] = w
        self._csr.data[self._pos[idx, 0]] = w + _EPS
        self._csr.data[self._pos[idx, 1]] = w + _EPS

    def shortest_path(self, src: int, dst: int) -> tuple[float, frozenset]:
        """
        Single-source Dijkstra via scipy (C implementation, O(E log V)).
        Returns (path_length, frozenset of edge indices on the path).
        """
        dist, pred = _sp_dijkstra(
            self._csr, directed=False, indices=src, return_predecessors=True
        )
        # Reconstruct edge indices by walking the predecessor array
        edge_indices: set[int] = set()
        node = dst
        while node != src:
            prev = int(pred[node])
            if prev < 0:
                return float("inf"), frozenset()
            u, v = (prev, node) if prev < node else (node, prev)
            edge_indices.add(self._lookup[(u, v)])
            node = prev
        return float(dist[dst]), frozenset(edge_indices)

    def eligible(self) -> np.ndarray:
        """Indices of edges with weight strictly below cap."""
        return np.where(self.weights < self.cap)[0]

    @property
    def total_weight(self) -> float:
        return float(self.weights.sum())

    def line_segments(self) -> list:
        """[(x1,y1),(x2,y2)] segments in (col, row) order for LineCollection."""
        segs = []
        for u, v in self.edges:
            r1, c1 = divmod(int(u), self.n)
            r2, c2 = divmod(int(v), self.n)
            segs.append([(c1, r1), (c2, r2)])
        return segs


# --------------------------------------------------------------------------- #


def run_simulation(
    n: int,
    cost: float,
    track: float,
    all_edges: bool,
    mutual: int,
    n_iter: int,
    n_frames: int,
):
    rng = np.random.default_rng()
    G = GridGraph(n, cost, track)
    node_pool = np.arange(G.N, dtype=np.int32)

    frame_interval = max(1, n_iter // n_frames)
    snapshots = [G.weights.copy()]
    snapshot_iters = [0]
    avg_path_lengths: list[float] = []
    total_sums: list[float] = []
    cum_path_len = 0.0
    total_sum = G.total_weight

    report_every = max(1, n_iter // 20)

    for i in range(n_iter):
        if i % report_every == 0:
            print(f"  {i:>{len(str(n_iter))}}/{n_iter} iterations …", end="\r", flush=True)

        # Find `mutual` shortest paths between independent random node pairs
        edge_sets: list[frozenset] = []
        iter_path_len = 0.0
        for _ in range(mutual):
            src, dst = rng.choice(node_pool, size=2, replace=False)
            plen, eset = G.shortest_path(int(src), int(dst))
            iter_path_len += plen
            edge_sets.append(eset)

        cum_path_len += iter_path_len / mutual
        avg_path_lengths.append(cum_path_len / (i + 1))

        # Only edges present in ALL paths are candidates for reinforcement
        common = list(edge_sets[0].intersection(*edge_sets[1:]))

        total_delta = 0.0
        if common:
            if not all_edges:
                # Decrease one randomly chosen common edge
                idx = int(common[int(rng.integers(len(common)))])
                v_old = G.weights[idx]
                v_new = max(v_old - 1.0, 0.0)
                total_delta = v_old - v_new
                G.set_weight(idx, v_new)
            else:
                # Decrease every common edge by 1
                for idx in common:
                    v_old = G.weights[idx]
                    v_new = max(v_old - 1.0, 0.0)
                    total_delta += v_old - v_new
                    G.set_weight(idx, v_new)

        total_sum -= total_delta

        # Redistribute: increment that many distinct eligible edges by 1
        if total_delta > 0:
            elig = G.eligible()
            n_pick = min(int(total_delta), len(elig))
            if n_pick > 0:
                chosen = rng.choice(elig, size=n_pick, replace=False)
                for idx in chosen:
                    G.set_weight(int(idx), G.weights[int(idx)] + 1.0)
                total_sum += n_pick

        total_sums.append(total_sum)

        if (i + 1) % frame_interval == 0:
            snapshots.append(G.weights.copy())
            snapshot_iters.append(i + 1)

    print()   # clear the progress line
    return G, snapshots, snapshot_iters, avg_path_lengths, total_sums


# --------------------------------------------------------------------------- #


def build_animation(
    G: GridGraph,
    n_iter: int,
    snapshots,
    snapshot_iters,
    avg_path_lengths,
    total_sums,
):
    segments = G.line_segments()
    cap = G.cap

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax_grid = fig.add_subplot(gs[0])
    ax_path = fig.add_subplot(gs[1])
    ax_sum  = fig.add_subplot(gs[2])

    ax_grid.set_xlim(-0.5, G.n - 0.5)
    ax_grid.set_ylim(-0.5, G.n - 0.5)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title(
        "City transport network\nblack = train track (weight 0)", fontsize=9
    )
    lc = LineCollection(segments, linewidths=1.0, colors="gray")
    ax_grid.add_collection(lc)
    iter_label = ax_grid.text(
        0.02, 0.98, "Iteration 0",
        transform=ax_grid.transAxes, va="top", ha="left", fontsize=8,
    )

    ax_path.set_xlim(1, n_iter)
    ax_path.set_ylim(0, max(avg_path_lengths) * 1.1 if avg_path_lengths else 1)
    ax_path.set_xlabel("Iteration")
    ax_path.set_ylabel("Running avg journey time")
    ax_path.set_title("Average path length")
    path_line, = ax_path.plot([], [], color="steelblue", lw=1.2)

    s_min, s_max = min(total_sums), max(total_sums)
    margin = max((s_max - s_min) * 0.1, 1.0)
    ax_sum.set_xlim(1, n_iter)
    ax_sum.set_ylim(s_min - margin, s_max + margin)
    ax_sum.set_xlabel("Iteration")
    ax_sum.set_ylabel("Total weight")
    ax_sum.set_title("Total weight sum\n(constant ⇒ conservation holds)")
    sum_line, = ax_sum.plot([], [], color="tomato", lw=1.2)

    def update(frame):
        weights = snapshots[frame]
        gray = np.where(weights == 0, 0.0, 0.25 + 0.65 * (weights / cap))
        colors = np.stack([gray, gray, gray, np.ones_like(gray)], axis=1)
        lws = np.where(weights == 0, 2.5, 0.7)
        lc.set_colors(colors)
        lc.set_linewidths(lws)

        it = snapshot_iters[frame]
        iter_label.set_text(f"Iteration {it:,}")

        end = min(it, len(avg_path_lengths))
        if end > 0:
            xs = np.arange(1, end + 1)
            path_line.set_data(xs, avg_path_lengths[:end])
            sum_line.set_data(xs, total_sums[:end])

        return lc, iter_label, path_line, sum_line

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=60, blit=True
    )
    return fig, anim


# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Urban transport emergence simulation on a 2D grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dim", type=int, default=20,
                        help="Grid side length N (city is N×N nodes)")
    parser.add_argument("--cost", type=float, default=1.0,
                        help="Initial travel cost per edge (any positive number)")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Number of optimisation iterations")
    parser.add_argument("--frames", type=int, default=200,
                        help="Number of animation frames to render")
    parser.add_argument("--track", type=float, default=0.1,
                        help="Fraction of edges initialised to weight 0 (train track)")
    parser.add_argument("--all", dest="all_edges", type=int, default=0,
                        choices=[0, 1],
                        help="0: decrease one common edge per iteration; "
                             "1: decrease all common edges and redistribute weight")
    parser.add_argument("--mutual", type=int, default=1,
                        help="Number of random paths per iteration; only edges shared "
                             "by all paths are eligible for optimisation")
    args = parser.parse_args()

    if args.dim < 2:
        parser.error("--dim must be at least 2")
    if args.cost <= 0:
        parser.error("--cost must be positive")
    if not 0.0 <= args.track <= 1.0:
        parser.error("--track must be between 0 and 1")
    if args.mutual < 1:
        parser.error("--mutual must be at least 1")

    print(
        f"Simulating {args.dim}×{args.dim} city  |  cost={args.cost}  |  "
        f"track={args.track:.0%}  |  all={args.all_edges}  |  mutual={args.mutual}  |  "
        f"{args.iterations:,} iterations  |  {args.frames} animation frames"
    )

    G, snapshots, snapshot_iters, avg_path_lengths, total_sums = run_simulation(
        args.dim, args.cost, args.track, bool(args.all_edges),
        args.mutual, args.iterations, args.frames,
    )

    print(
        f"Done.  Initial total weight: {total_sums[0]:.4f}  "
        f"Final: {total_sums[-1]:.4f}  "
        f"(diff = {abs(total_sums[-1] - total_sums[0]):.2e})"
    )

    fig, anim = build_animation(
        G, args.iterations, snapshots, snapshot_iters, avg_path_lengths, total_sums,
    )
    plt.show()


if __name__ == "__main__":
    main()
