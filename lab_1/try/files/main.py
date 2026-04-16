"""
main.py — Mirpur City Pathfinding Assignment
=============================================
Run with:
    python main.py                        # default: node 0 → 11
    python main.py --src 0 --dst 11       # choose source/destination
    python main.py --profile female       # traveler profile
    python main.py --no-plot              # print results only, skip matplotlib
    python main.py --interactive          # pick nodes from printed list
    python main.py --algo astar           # run one algorithm only

Plots open live via plt.show() — no images are pre-generated.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mirpur_map import MirpurGraph, NODES
from heuristic import DEFAULT_WEIGHTS, PROFILES, describe_weights
from algorithms import bfs, dfs, ucs, greedy, astar

DIVIDER = "─" * 70

ALL_ALGOS = {
    "bfs":    ("BFS",    bfs),
    "dfs":    ("DFS",    dfs),
    "ucs":    ("UCS",    ucs),
    "greedy": ("Greedy", greedy),
    "astar":  ("A*",     astar),
}


def print_result(result, graph):
    status = "✓ FOUND" if result.found else "✗ NOT FOUND"
    print(f"\n  Algorithm    : {result.algorithm}")
    print(f"  Status       : {status}")
    print(f"  Nodes visited: {result.nodes_visited}")
    if result.found:
        print(f"  Path length  : {result.path_length} edges")
        print(f"  Total cost   : {result.total_cost:.3f}")
        path_names = " → ".join(graph.node_name(n) for n in result.path)
        print(f"  Path         : {path_names}")
    print(f"  Visit order  : {result.visited_order}")


def print_comparison(results):
    algo_type   = {"BFS": "Uninformed", "DFS": "Uninformed", "UCS": "Uninformed",
                   "Greedy": "Informed",  "A*":  "Informed"}
    optimal_map = {"BFS": "hop-optimal", "DFS": "No", "UCS": "Yes (cost)",
                   "Greedy": "No",        "A*":  "Yes (cost)"}

    print(f"\n{'='*70}")
    print(f"  {'ALGORITHM':<12} {'TYPE':<12} {'VISITED':>8} {'COST':>10} "
          f"{'EDGES':>6}  {'OPTIMAL':<14} FOUND")
    print(f"  {DIVIDER}")
    for r in results:
        cost_str  = f"{r.total_cost:.2f}" if r.found else "—"
        edge_str  = str(r.path_length)    if r.found else "—"
        found_str = "YES ✓"               if r.found else "NO ✗"
        print(f"  {r.algorithm:<12} {algo_type[r.algorithm]:<12} {r.nodes_visited:>8} "
              f"{cost_str:>10} {edge_str:>6}  {optimal_map[r.algorithm]:<14} {found_str}")
    print(f"{'='*70}")

    found = [r for r in results if r.found]
    if found:
        best_eff  = min(found, key=lambda r: r.nodes_visited)
        worst_eff = max(found, key=lambda r: r.nodes_visited)
        best_cost = min(found, key=lambda r: r.total_cost)
        print(f"\n  🏆 Most efficient (fewest nodes visited) : "
              f"{best_eff.algorithm} ({best_eff.nodes_visited} nodes)")
        print(f"  📉 Least efficient (most nodes visited)  : "
              f"{worst_eff.algorithm} ({worst_eff.nodes_visited} nodes)")
        print(f"  💰 Lowest cost path                      : "
              f"{best_cost.algorithm} (cost = {best_cost.total_cost:.2f})\n")


def run(source, goal, weights, algo_key="all", show_plots=True):
    graph = MirpurGraph()

    print(f"\n{'='*70}")
    print(f"  MIRPUR CITY PATHFINDING — AI SEARCH ASSIGNMENT")
    print(f"{'='*70}")
    print(f"  Source : [{source}] {graph.node_name(source)}")
    print(f"  Goal   : [{goal}]  {graph.node_name(goal)}")
    print()
    print(describe_weights(weights))

    # Choose which algorithms to run
    if algo_key == "all":
        to_run = list(ALL_ALGOS.values())
    else:
        to_run = [ALL_ALGOS[algo_key]]

    print(f"\n{'='*70}")
    print("  RUNNING ALGORITHMS")
    print(f"{'='*70}")

    results = []
    for name, fn in to_run:
        print(f"\n  ▶ Running {name}...")
        result = fn(graph, source, goal, weights)
        print_result(result, graph)
        results.append(result)

    if len(results) > 1:
        print(f"\n{'='*70}")
        print("  PERFORMANCE COMPARISON")
        print_comparison(results)

    if show_plots:
        from visualize import draw_city_map, draw_path, compare_results

        print("\n  [Showing city map...]")
        draw_city_map()

        for r in results:
            print(f"  [Showing path: {r.algorithm}...]")
            draw_path(r, source, goal)

        if len(results) > 1:
            print("  [Showing comparison chart...]")
            compare_results(results, source, goal)

    return results


def interactive_mode():
    graph = MirpurGraph()
    print("\n  NODES:")
    for nid, info in NODES.items():
        print(f"    [{nid:2d}] {info['name']}")
    src = int(input("\n  Enter source node ID: ").strip())
    dst = int(input("  Enter destination node ID: ").strip())
    return src, dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirpur Pathfinding — AI Assignment")
    parser.add_argument("--src",    type=int, default=0,
                        help="Source node ID (default: 0)")
    parser.add_argument("--dst",    type=int, default=11,
                        help="Destination node ID (default: 11 = Agargaon)")
    parser.add_argument("--profile", type=str, default="default",
                        choices=list(PROFILES.keys()),
                        help="Traveler profile")
    parser.add_argument("--algo",   type=str, default="all",
                        choices=["all"] + list(ALL_ALGOS.keys()),
                        help="Which algorithm to run (default: all)")
    parser.add_argument("--interactive", action="store_true",
                        help="Pick source/destination interactively")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plots, print only")
    args = parser.parse_args()

    weights = PROFILES[args.profile]
    src, dst = args.src, args.dst

    if args.interactive:
        src, dst = interactive_mode()

    run(source=src, goal=dst, weights=weights,
        algo_key=args.algo, show_plots=not args.no_plot)
