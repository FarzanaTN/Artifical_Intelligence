from graph import load_graph
from algorithms import bfs, dfs, ucs, astar
from visualizer import save_path_image
from metrics import run_with_metrics
from cost_heuristic import heuristic_fn

G, start, goal = load_graph()

algorithms = {
    "BFS": lambda: bfs(G, start, goal),
    "DFS": lambda: dfs(G, start, goal),
    "UCS": lambda: ucs(G, start, goal),
    "A*": lambda: astar(G, start, goal, heuristic_fn)
}

results = {}

for name, func in algorithms.items():
    print("\nRunning", name)

    data = run_with_metrics(func)
    result = data["result"]

    # ✔ FINAL PATH (THIS IS WHAT YOU ASKED)
    path = None
    if result:
        path = result.get("path")

    # ✔ SAVE IMAGE AFTER EACH ALGO
    save_path_image(G, path, name)

    # ✔ STORE EVERYTHING
    results[name] = {
        "path": path,
        "time": data["time"],
        "space": data["memory"]   # THIS = SPACE
    }

# =======================
# FINAL OUTPUT (YOU WANTED THIS)
# =======================

print("\n===== FINAL COMPARISON =====\n")

for name, r in results.items():
    print(name)
    print("Path  :", r["path"])
    print("Time  :", r["time"])
    print("Space :", r["space"])
    print("----------------------")