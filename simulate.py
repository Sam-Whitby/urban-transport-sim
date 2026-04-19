#!/usr/bin/env python3
"""
Urban transport emergence simulation on a 2D grid graph.

A city is represented as an N×N lattice. Each edge carries a travel-time
weight (default 1). Repeated optimisation of random origin-destination journeys
causes zero-weight corridors — analogous to train lines — to self-organise.
"""

import argparse
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec


def run_simulation(n: int, cost: float, n_iter: int, n_frames: int):
    """Run optimisation and return snapshots plus per-iteration statistics."""
    G = nx.grid_2d_graph(n, n)
    for u, v in G.edges():
        G[u][v]['weight'] = float(cost)

    nodes = list(G.nodes())
    edges = list(G.edges())
    cap = float(cost)
    total_sum = len(edges) * cap

    frame_interval = max(1, n_iter // n_frames)

    snapshots = [np.array([G[u][v]['weight'] for u, v in edges], dtype=float)]
    snapshot_iters = [0]
    avg_path_lengths: list[float] = []
    total_sums: list[float] = []
    cum_path_len = 0.0

    for i in range(n_iter):
        src, dst = random.sample(nodes, 2)

        path = nx.shortest_path(G, src, dst, weight='weight')
        path_len = nx.shortest_path_length(G, src, dst, weight='weight')
        cum_path_len += path_len
        avg_path_lengths.append(cum_path_len / (i + 1))

        # Decrease a random edge on the shortest path
        path_edges = list(zip(path[:-1], path[1:]))
        eu, ev = random.choice(path_edges)
        v_old = G[eu][ev]['weight']
        v_new = max(v_old - 1.0, 0.0)
        delta = v_old - v_new          # 0 or 1
        G[eu][ev]['weight'] = v_new
        total_sum -= delta

        # Redistribute delta to a random edge that has room to grow (conserves total)
        if delta > 0:
            eligible = [(u, v) for u, v in edges if G[u][v]['weight'] < cap]
            if eligible:
                ru, rv = random.choice(eligible)
                G[ru][rv]['weight'] += delta  # guaranteed <= cap
                total_sum += delta

        total_sums.append(total_sum)

        if (i + 1) % frame_interval == 0:
            snapshots.append(
                np.array([G[u][v]['weight'] for u, v in edges], dtype=float)
            )
            snapshot_iters.append(i + 1)

    return edges, snapshots, snapshot_iters, avg_path_lengths, total_sums


def build_animation(n, cost, n_iter, edges, snapshots, snapshot_iters,
                    avg_path_lengths, total_sums):
    cap = float(cost)
    segments = [[(u[0], u[1]), (v[0], v[1])] for u, v in edges]

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax_grid = fig.add_subplot(gs[0])
    ax_path = fig.add_subplot(gs[1])
    ax_sum  = fig.add_subplot(gs[2])

    # --- Grid panel ---
    ax_grid.set_xlim(-0.5, n - 0.5)
    ax_grid.set_ylim(-0.5, n - 0.5)
    ax_grid.set_aspect('equal')
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title('City transport network\nblack = train track (weight 0)',
                      fontsize=9)

    lc = LineCollection(segments, linewidths=1.0, colors='gray')
    ax_grid.add_collection(lc)
    iter_label = ax_grid.text(0.02, 0.98, 'Iteration 0',
                              transform=ax_grid.transAxes,
                              va='top', ha='left', fontsize=8)

    # --- Average path length panel ---
    ax_path.set_xlim(1, n_iter)
    ax_path.set_ylim(0, max(avg_path_lengths) * 1.1 if avg_path_lengths else 1)
    ax_path.set_xlabel('Iteration')
    ax_path.set_ylabel('Running avg journey time')
    ax_path.set_title('Average path length')
    path_line, = ax_path.plot([], [], color='steelblue', lw=1.2)

    # --- Total weight panel ---
    s_min, s_max = min(total_sums), max(total_sums)
    margin = max((s_max - s_min) * 0.1, 1.0)
    ax_sum.set_xlim(1, n_iter)
    ax_sum.set_ylim(s_min - margin, s_max + margin)
    ax_sum.set_xlabel('Iteration')
    ax_sum.set_ylabel('Total weight')
    ax_sum.set_title('Total weight sum\n(constant ⇒ conservation holds)')
    sum_line, = ax_sum.plot([], [], color='tomato', lw=1.2)

    def update(frame):
        weights = snapshots[frame]

        # Black for weight=0 (train track), light gray for weight near cap
        gray = np.where(weights == 0, 0.0, 0.25 + 0.65 * (weights / cap))
        colors = np.stack([gray, gray, gray, np.ones_like(gray)], axis=1)
        lws = np.where(weights == 0, 2.5, 0.7)
        lc.set_colors(colors)
        lc.set_linewidths(lws)

        it = snapshot_iters[frame]
        iter_label.set_text(f'Iteration {it:,}')

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


def main():
    parser = argparse.ArgumentParser(
        description='Urban transport emergence simulation on a 2D grid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dim', type=int, default=20,
                        help='Grid side length N (city is N×N nodes)')
    parser.add_argument('--cost', type=float, default=1.0,
                        help='Initial travel cost per edge (any positive number)')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of optimisation iterations')
    parser.add_argument('--frames', type=int, default=200,
                        help='Number of animation frames to render')
    args = parser.parse_args()

    if args.dim < 2:
        parser.error('--dim must be at least 2')
    if args.cost <= 0:
        parser.error('--cost must be positive')

    print(
        f"Simulating {args.dim}×{args.dim} city  |  "
        f"cost={args.cost}  |  {args.iterations:,} iterations  |  "
        f"{args.frames} animation frames"
    )

    edges, snapshots, snapshot_iters, avg_path_lengths, total_sums = run_simulation(
        args.dim, args.cost, args.iterations, args.frames
    )

    print(
        f"Done.  Initial total weight: {total_sums[0]:.4f}  "
        f"Final: {total_sums[-1]:.4f}  "
        f"(diff = {abs(total_sums[-1] - total_sums[0]):.2e})"
    )

    fig, anim = build_animation(
        args.dim, args.cost, args.iterations,
        edges, snapshots, snapshot_iters, avg_path_lengths, total_sums,
    )
    plt.show()


if __name__ == '__main__':
    main()
