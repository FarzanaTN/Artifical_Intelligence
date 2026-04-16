"""
main.py — Mirpur City Pathfinding Assignment
=============================================
Usage:
    python main.py                        # default run
    python main.py --src 0 --dst 15       # choose source/destination
    python main.py --profile female       # use a traveler profile
    python main.py --save                 # save all plots as PNG files
    python main.py --no-plot              # only print results, no visuals

Run All Algorithms and Compare Performance.
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mirpur_map import MirpurGraph, NODES
from heuristic import DEFAULT_WEIGHTS, PROFILES, describe_weights, compute_edge_cost
from algorithms import bfs, dfs, ucs, greedy, astar


DIVIDER = "─" * 70


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


def print_comparison_table(results):
    algo_type = {
        "BFS": "Uninformed", "DFS": "Uninformed", "UCS": "Uninformed",
        "Greedy": "Informed", "A*": "Informed"
    }
    optimal = {
        "BFS": "hop-optimal", "DFS": "No",
        "UCS": "Yes (cost)", "Greedy": "No", "A*": "Yes (cost)"
    }

    print(f"\n{'='*70}")
    print(f"  {'ALGORITHM':<12} {'TYPE':<12} {'VISITED':>8} {'COST':>10} {'EDGES':>6}  {'OPTIMAL':<14} {'FOUND'}")
    print(f"  {DIVIDER}")
    for r in results:
        cost_str  = f"{r.total_cost:.2f}" if r.found else "—"
        edge_str  = str(r.path_length) if r.found else "—"
        found_str = "YES ✓" if r.found else "NO ✗"
        print(f"  {r.algorithm:<12} {algo_type[r.algorithm]:<12} {r.nodes_visited:>8} "
              f"{cost_str:>10} {edge_str:>6}  {optimal[r.algorithm]:<14} {found_str}")
    print(f"{'='*70}")

    # Best/worst highlights
    found_results = [r for r in results if r.found]
    if found_results:
        best_visited  = min(found_results, key=lambda r: r.nodes_visited)
        worst_visited = max(found_results, key=lambda r: r.nodes_visited)
        best_cost     = min(found_results, key=lambda r: r.total_cost)

        print(f"\n  🏆 Most efficient (fewest nodes visited) : {best_visited.algorithm}"
              f" ({best_visited.nodes_visited} nodes)")
        print(f"  📉 Least efficient (most nodes visited)  : {worst_visited.algorithm}"
              f" ({worst_visited.nodes_visited} nodes)")
        print(f"  💰 Lowest cost path                      : {best_cost.algorithm}"
              f" (cost = {best_cost.total_cost:.2f})")
        print()


def run(source, goal, weights, show_plots=True, save_plots=False):
    graph = MirpurGraph()

    print(f"\n{'='*70}")
    print(f"  MIRPUR CITY PATHFINDING — AI SEARCH ASSIGNMENT")
    print(f"{'='*70}")
    print(f"  Source : [{source}] {graph.node_name(source)}")
    print(f"  Goal   : [{goal}]  {graph.node_name(goal)}")
    print()
    print(describe_weights(weights))

    print(f"\n{'='*70}")
    print("  RUNNING ALL ALGORITHMS")
    print(f"{'='*70}")

    algorithms = [
        ("BFS",    bfs),
        ("DFS",    dfs),
        ("UCS",    ucs),
        ("Greedy", greedy),
        ("A*",     astar),
    ]

    results = []
    for name, fn in algorithms:
        print(f"\n  ▶ Running {name}...")
        result = fn(graph, source, goal, weights)
        print_result(result, graph)
        results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("  PERFORMANCE COMPARISON")
    print_comparison_table(results)

    if show_plots:
        from visualize import draw_city_map, draw_path, compare_results

        # 1. Base map
        print("  [Plot 1/7] City map...")
        save = "output_city_map.png" if save_plots else None
        draw_city_map(save_path=save)

        # 2. Individual algorithm maps
        for r in results:
            print(f"  [Plot] {r.algorithm} path map...")
            save = f"output_{r.algorithm.lower()}_path.png" if save_plots else None
            draw_path(r, weights, source, goal, save_path=save)

        # 3. Comparison chart
        print("  [Plot] Comparison chart...")
        save = "output_comparison.png" if save_plots else None
        compare_results(results, source, goal, save_path=save)

    return results


def interactive_mode():
    """Let user pick source/destination from printed node list."""
    graph = MirpurGraph()
    print("\n  NODES:")
    for nid, info in NODES.items():
        print(f"    [{nid:2d}] {info['name']}")
    src = int(input("\n  Enter source node ID: ").strip())
    dst = int(input("  Enter destination node ID: ").strip())
    return src, dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirpur Pathfinding — AI Assignment")
    parser.add_argument("--src",      type=int, default=0,
                        help="Source node ID (default: 0 = Mirpur-1 Bus Stand)")
    parser.add_argument("--dst",      type=int, default=15,
                        help="Destination node ID (default: 15 = Mazar Road Junction)")
    parser.add_argument("--profile",  type=str, default="default",
                        choices=list(PROFILES.keys()),
                        help="Traveler profile (default/female/elderly/rush/budget)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactively select source/destination")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip matplotlib plots, only print results")
    parser.add_argument("--save",     action="store_true",
                        help="Save all plots as PNG files")
    args = parser.parse_args()

    weights = PROFILES[args.profile]
    src, dst = args.src, args.dst

    if args.interactive:
        src, dst = interactive_mode()

    run(
        source=src,
        goal=dst,
        weights=weights,
        show_plots=not args.no_plot,
        save_plots=args.save,
    )
