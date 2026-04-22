

import time

import osmnx as ox

from config import FN_NOTES, OUTPUT_FOLDER
from graph import load_mirpur_map, path_cost
from algo import (
    bfs, dfs, dls, iddfs,
    ucs, greedy_bfs, a_star, weighted_a_star,
    bidirectional_dijkstra, bidirectional_astar, ida_star,
)
from visu import (
    plot_route,
    plot_risk_heatmap,
    plot_all_routes_overlay,
    plot_complexities,
    plot_cost_comparison,
    plot_risk_metrics_distribution,
)



def build_algo_list(G, start, goal) -> list:
   
    return [
        ("BFS",        lambda: bfs(G, start, goal)),
        ("DFS",        lambda: dfs(G, start, goal)),
        ("DLS",        lambda: dls(G, start, goal, limit=40)),
        ("IDDFS",      lambda: iddfs(G, start, goal, max_depth=40)),
        ("UCS",        lambda: ucs(G, start, goal)),
        ("Greedy",     lambda: greedy_bfs(G, start, goal)),
        ("A*",         lambda: a_star(G, start, goal)),
        ("WeightedA*", lambda: weighted_a_star(G, start, goal, weight=1.5)),
        ("BiDi",       lambda: bidirectional_dijkstra(G, start, goal)),
        ("BiDiA*",     lambda: bidirectional_astar(G, start, goal)),
        ("IDA*",       lambda: ida_star(G, start, goal)),
    ]


def run_all() -> None:
    # ── 1. Load map ──────────────────────────────────────────────────────────
    G     = load_mirpur_map()
    start = ox.distance.nearest_nodes(G, 90.3545, 23.7997)
    goal  = ox.distance.nearest_nodes(G, 90.3683, 23.8069)
    print(f"\nStart: Mirpur-1  |  Goal: Mirpur-10\n")

    # ── 2. Run algorithms ────────────────────────────────────────────────────
    algos      = build_algo_list(G, start, goal)
    results    = []
    algo_paths = []

    for name, func in algos:
        print(f"Running {name} ...")
        t0                              = time.time()
        path, explored, max_fr, meet   = func()
        elapsed                         = (time.time() - t0) * 1000
        cost                            = path_cost(G, path) if path else float("inf")

        results.append({"name": name, "time": elapsed,
                         "nodes": explored, "max_frontier": max_fr, "cost": cost})
        algo_paths.append((name, path))

        status = f"{len(path)} hops, cost={cost:.1f}" if path else "NO PATH"
        print(f"  {name:<12} | {elapsed:7.1f} ms | {explored:5} explored "
              f"| frontier peak={max_fr:4} | {status}")

        if path:
            plot_route(G, path, name, start, goal, meet)

    # ── 3. Summary plots ─────────────────────────────────────────────────────
    print("\nGenerating summary plots ...")
    plot_complexities(results)
    plot_cost_comparison(results)
    plot_risk_heatmap(G, start, goal)
    plot_all_routes_overlay(G, algo_paths, start, goal)
    plot_risk_metrics_distribution(G)

    # ── 4. Console summary table ─────────────────────────────────────────────
    print("\n" + "=" * 97)
    print(f"{'Algorithm':<14} {'Time(ms)':>9} {'Explored':>10} {'PeakFrontier':>14} "
          f"{'PathCost':>12} ")
    print("=" * 97)
    for r in results:
        n = r["name"]
        print(f"{n:<14} {r['time']:>9.1f} {r['nodes']:>10} {r['max_frontier']:>14} "
              f"{r['cost']:>12.1f} ")
    print("=" * 97)
    print(f"\nAll outputs → {OUTPUT_FOLDER}")


if __name__ == "__main__":
    run_all()
