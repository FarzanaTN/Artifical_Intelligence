import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time
import math
import random

# ============================================================================
# SETUP
# ============================================================================
OUTPUT_FOLDER = "assignment_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

random.seed(42)

# ============================================================================
# 1. RISK MODEL
# ============================================================================

def edge_risk_multiplier(traffic, safety):
    """
    Real-world edge risk used in ACTUAL cost (g(n)).
    """
    w1, w2 = 0.6, 0.4
    return 1 + (w1 * traffic + w2 * safety)


def heuristic_risk_factor():
    """
    Estimated average risk used in heuristic h(n).
    Must be a LOWER/AVERAGE estimate (not exact edge values).
    """
    return 1.3


# ============================================================================
# 2. LOAD GRAPH
# ============================================================================

def load_mirpur_map():
    print("Fetching Mirpur map data...")

    center = (23.8041, 90.3625)
    G = ox.graph_from_point(center, dist=1000, network_type="drive")

    try:
        G = ox.truncate.largest_component(G, strongly=True)
    except:
        largest = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    for u, v, k, data in G.edges(data=True, keys=True):

        length = data.get("length", 1)

        # deterministic synthetic risk attributes
        traffic = random.uniform(1.0, 2.0)
        safety = random.uniform(1.0, 1.5)

        data["traffic"] = traffic
        data["safety"] = safety

        # ============================================================
        # g(n): REAL COST (distance × real risk)
        # ============================================================
        data["cost"] = length * edge_risk_multiplier(traffic, safety)

    print(f"Graph loaded: {len(G.nodes)} nodes.")
    return G


# ============================================================================
# 3. HEURISTIC (NOW INCLUDES RISK)
# ============================================================================

def heuristic(graph, node, goal):
    """
    h(n) = straight-line distance × estimated risk factor
    (includes risk as instructed by your professor)
    """
    n1 = graph.nodes[node]
    n2 = graph.nodes[goal]

    dist = math.sqrt(
        (n1["x"] - n2["x"])**2 +
        (n1["y"] - n2["y"])**2
    ) * 111000  # meters approx

    return dist * heuristic_risk_factor()


# ============================================================================
# 4. PATH RECONSTRUCTION
# ============================================================================

def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return path[::-1]


# ============================================================================
# 5. ALGORITHMS
# ============================================================================

def bfs(graph, start, goal):
    q, vis, cf = [start], {start}, {start: None}
    count = 0

    while q:
        node = q.pop(0)
        count += 1

        if node == goal:
            return reconstruct_path(cf, goal), count

        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                q.append(n)

    return None, count


def dfs(graph, start, goal):
    stack, vis, cf = [start], {start}, {start: None}
    count = 0

    while stack:
        node = stack.pop()
        count += 1

        if node == goal:
            return reconstruct_path(cf, goal), count

        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                stack.append(n)

    return None, count


def ucs(graph, start, goal):
    pq = [(0, start)]
    g = {start: 0}
    cf = {start: None}
    visited = {}
    count = 0

    while pq:
        cost, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited[node] = cost
        count += 1

        if node == goal:
            return reconstruct_path(cf, goal), count

        for n in graph.neighbors(node):
            new_cost = cost + graph[node][n][0]["cost"]

            if n not in g or new_cost < g[n]:
                g[n] = new_cost
                cf[n] = node
                heapq.heappush(pq, (new_cost, n))

    return None, count


def greedy_bfs(graph, start, goal):
    pq = [(heuristic(graph, start, goal), start)]
    vis = {start}
    cf = {start: None}
    count = 0

    while pq:
        _, node = heapq.heappop(pq)
        count += 1

        if node == goal:
            return reconstruct_path(cf, goal), count

        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                heapq.heappush(pq, (heuristic(graph, n, goal), n))

    return None, count


def weighted_a_star(graph, start, goal, weight=1.5):
    pq = [(0, start)]
    g = {start: 0}
    cf = {start: None}
    count = 0

    while pq:
        _, node = heapq.heappop(pq)
        count += 1

        if node == goal:
            return reconstruct_path(cf, goal), count

        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]

            if n not in g or new_g < g[n]:
                g[n] = new_g
                f = new_g + weight * heuristic(graph, n, goal)
                cf[n] = node
                heapq.heappush(pq, (f, n))

    return None, count


# ============================================================================
# 6. PLOTTING (NO POPUPS)
# ============================================================================

def plot_route(G, path, name):
    fig, ax = ox.plot_graph_route(
        G,
        path,
        route_color="red",
        node_size=0,
        show=False,
        close=True
    )

    fig.savefig(f"{OUTPUT_FOLDER}/{name}_route.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_complexities(results):
    names = [r["name"] for r in results]
    times = [r["time"] for r in results]
    nodes = [r["nodes"] for r in results]

    plt.figure()
    plt.bar(names, times)
    plt.yscale("log")
    plt.title("Time Complexity (ms)")
    plt.savefig(f"{OUTPUT_FOLDER}/time_complexity.png")
    plt.close()

    plt.figure()
    plt.bar(names, nodes)
    plt.yscale("log")
    plt.title("Space Complexity (Nodes Explored)")
    plt.savefig(f"{OUTPUT_FOLDER}/space_complexity.png")
    plt.close()


# ============================================================================
# 7. MAIN RUNNER
# ============================================================================

def run_all():
    G = load_mirpur_map()

    start = ox.distance.nearest_nodes(G, 90.3545, 23.7997)
    goal = ox.distance.nearest_nodes(G, 90.3683, 23.8069)

    algos = [
        ("BFS", lambda: bfs(G, start, goal)),
        ("DFS", lambda: dfs(G, start, goal)),
        ("UCS", lambda: ucs(G, start, goal)),
        ("Greedy", lambda: greedy_bfs(G, start, goal)),
        ("WeightedA*", lambda: weighted_a_star(G, start, goal))
    ]

    results = []

    for name, func in algos:
        print(f"Running {name}...")
        t0 = time.time()

        path, count = func()

        results.append({
            "name": name,
            "time": (time.time() - t0) * 1000,
            "nodes": count
        })

        if path:
            plot_route(G, path, name)

    plot_complexities(results)
    print("DONE → All outputs saved in:", OUTPUT_FOLDER)


# ============================================================================
# ENTRY
# ============================================================================

if __name__ == "__main__":
    run_all()