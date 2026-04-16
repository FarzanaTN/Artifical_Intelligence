"""
main.py — Mirpur OSMnx Pathfinding
====================================
Downloads REAL Mirpur road network from OpenStreetMap via OSMnx.
Runs all 12 search algorithms. Saves all images to ./output/

Usage:
    python main.py                          # default profile
    python main.py --profile female
    python main.py --profile elderly
    python main.py --profile rush_hour
    python main.py --profile budget
    python main.py --no-plot                # skip image saving

First run downloads the graph from OSM and caches to mirpur_graph.graphml.
Subsequent runs load from cache instantly.
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import CityGraph
from cost_heuristic import PROFILES, describe_weights
from algorithms import run_all
from visualize import generate_all, ALGO_TYPE, OPTIMAL

DIV = "─" * 72


def print_results(results, graph):
    print(f"\n{'='*72}")
    print("  ALL ALGORITHM RESULTS")
    print(f"{'='*72}")
    for r in results:
        print(f"\n{DIV}")
        print(r.summary(graph))
    print(f"\n{'='*72}")


def print_table(results):
    print(f"\n{'='*72}")
    print("  PERFORMANCE COMPARISON")
    print(f"{'='*72}")
    print(f"  {'ALGO':<15} {'TYPE':<12} {'VISITED':>8} {'COST':>10} "
          f"{'EDGES':>7}  {'OPTIMAL':<14} FOUND")
    print(f"  {DIV}")
    for r in results:
        cs = f"{r.total_cost:.3f}" if r.found else "—"
        es = str(r.path_length)    if r.found else "—"
        fs = "✓ YES"               if r.found else "✗ NO"
        print(f"  {r.algorithm:<15} {ALGO_TYPE.get(r.algorithm,'-'):<12} "
              f"{r.nodes_visited:>8} {cs:>10} {es:>7}  "
              f"{OPTIMAL.get(r.algorithm,'-'):<14} {fs}")
    print(f"{'='*72}")
    found = [r for r in results if r.found]
    if found:
        be = min(found, key=lambda r: r.nodes_visited)
        we = max(found, key=lambda r: r.nodes_visited)
        bc = min(found, key=lambda r: r.total_cost)
        print(f"\n  🏆 Most efficient : {be.algorithm} ({be.nodes_visited} nodes visited)")
        print(f"  📉 Least efficient: {we.algorithm} ({we.nodes_visited} nodes visited)")
        print(f"  💰 Lowest cost    : {bc.algorithm} (cost = {bc.total_cost:.4f})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="female",
                        choices=list(PROFILES.keys()))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    weights = PROFILES[args.profile]

    print(f"\n{'='*72}")
    print("  MIRPUR PATHFINDING — OSMnx + 12 ALGORITHMS")
    print(f"{'='*72}")
    print(f"  Profile: {args.profile}")
    print()
    print(describe_weights(weights))
    print()

    graph = CityGraph()
    source, goal = graph.source, graph.goal

    print(f"\n  Running 12 algorithms...")
    results = run_all(graph, source, goal, weights)

    print_results(results, graph)
    print_table(results)

    if not args.no_plot:
        generate_all(results, graph, source, goal, args.profile)
