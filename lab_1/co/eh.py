import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time
import math
import random
from collections import deque

# ============================================================================
# SETUP
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "assignment_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ox.settings.use_cache = True
random.seed(42)

# ============================================================================
# RISK MODEL
# ============================================================================
def edge_risk_multiplier(traffic, safety, gender_risk, age_risk):
    return 1 + (0.4*traffic + 0.3*safety + 0.2*gender_risk + 0.1*age_risk)

# ============================================================================
# LOAD GRAPH
# ============================================================================
def load_mirpur_map():
    print("Loading map...")
    center = (23.8041, 90.3625)

    G = ox.graph_from_point(center, dist=1000, network_type="drive")

    # version-safe largest component
    try:
        G = ox.truncate.largest_component(G, strongly=True)
    except:
        largest = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    risks = []

    for u, v, k, data in G.edges(data=True, keys=True):
        length = data.get("length", 1)

        traffic = random.uniform(1.0, 2.0)
        safety = random.uniform(1.0, 1.5)
        gender = random.uniform(1.0, 1.3)
        age = random.uniform(1.0, 1.2)

        risk = edge_risk_multiplier(traffic, safety, gender, age)
        data["cost"] = length * risk
        risks.append(risk)

    G.graph["avg_risk"] = sum(risks) / len(risks)

    print("Graph loaded:", len(G.nodes))
    return G

# ============================================================================
# HEURISTIC
# ============================================================================
def heuristic(G, n, g):
    n1, n2 = G.nodes[n], G.nodes[g]
    dist = math.sqrt((n1["x"]-n2["x"])**2 + (n1["y"]-n2["y"])**2) * 111000
    return dist * G.graph["avg_risk"]

# ============================================================================
# PATH UTILS
# ============================================================================
def reconstruct_path(cf, node):
    path = []
    while node is not None:
        path.append(node)
        node = cf.get(node)
    return path[::-1]

def merge_paths(cf1, cf2, meet):
    path1 = reconstruct_path(cf1, meet)

    path2 = []
    node = cf2.get(meet)
    while node is not None:
        path2.append(node)
        node = cf2.get(node)

    return path1 + path2

# ============================================================================
# BASIC SEARCH
# ============================================================================
def bfs(G, s, g):
    q = deque([s])
    vis = {s}
    cf = {s: None}
    count = 0

    while q:
        n = q.popleft()
        count += 1

        if n == g:
            return reconstruct_path(cf, g), count

        for nb in G.neighbors(n):
            if nb not in vis:
                vis.add(nb)
                cf[nb] = n
                q.append(nb)

    return None, count

def dfs(G, s, g):
    stack = [s]
    vis = {s}
    cf = {s: None}
    count = 0

    while stack:
        n = stack.pop()
        count += 1

        if n == g:
            return reconstruct_path(cf, g), count

        for nb in G.neighbors(n):
            if nb not in vis:
                vis.add(nb)
                cf[nb] = n
                stack.append(nb)

    return None, count

def ucs(G, s, g):
    pq = [(0, s)]
    g_cost = {s: 0}
    cf = {s: None}
    count = 0

    while pq:
        cost, n = heapq.heappop(pq)
        count += 1

        if n == g:
            return reconstruct_path(cf, g), count

        for nb in G.neighbors(n):
            new = cost + G[n][nb][0]["cost"]
            if nb not in g_cost or new < g_cost[nb]:
                g_cost[nb] = new
                cf[nb] = n
                heapq.heappush(pq, (new, nb))

    return None, count

def greedy(G, s, g):
    pq = [(heuristic(G, s, g), s)]
    vis = {s}
    cf = {s: None}
    count = 0

    while pq:
        _, n = heapq.heappop(pq)
        count += 1

        if n == g:
            return reconstruct_path(cf, g), count

        for nb in G.neighbors(n):
            if nb not in vis:
                vis.add(nb)
                cf[nb] = n
                heapq.heappush(pq, (heuristic(G, nb, g), nb))

    return None, count

def weighted_a_star(G, s, g, w=1.5):
    pq = [(0, s)]
    g_cost = {s: 0}
    cf = {s: None}
    count = 0

    while pq:
        _, n = heapq.heappop(pq)
        count += 1

        if n == g:
            return reconstruct_path(cf, g), count

        for nb in G.neighbors(n):
            new_g = g_cost[n] + G[n][nb][0]["cost"]

            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                f = new_g + w * heuristic(G, nb, g)
                cf[nb] = n
                heapq.heappush(pq, (f, nb))

    return None, count

# ============================================================================
# DLS + IDDFS
# ============================================================================
def dls(G, node, goal, limit, vis, cf):
    if node == goal:
        return True

    if limit <= 0:
        return False

    for nb in G.neighbors(node):
        if nb not in vis:
            vis.add(nb)
            cf[nb] = node
            if dls(G, nb, goal, limit-1, vis, cf):
                return True

    return False

def iddfs(G, s, g, max_depth=50):
    for d in range(max_depth):
        vis = {s}
        cf = {s: None}
        if dls(G, s, g, d, vis, cf):
            return reconstruct_path(cf, g), d
    return None, 0

# ============================================================================
# BIDIRECTIONAL BFS
# ============================================================================
def bidirectional_bfs(G, s, g):
    q1, q2 = deque([s]), deque([g])
    vis1, vis2 = {s}, {g}
    cf1, cf2 = {s: None}, {g: None}

    while q1 and q2:
        n1 = q1.popleft()
        for nb in G.neighbors(n1):
            if nb not in vis1:
                vis1.add(nb)
                cf1[nb] = n1
                q1.append(nb)
                if nb in vis2:
                    return merge_paths(cf1, cf2, nb), 0

        n2 = q2.popleft()
        for nb in G.neighbors(n2):
            if nb not in vis2:
                vis2.add(nb)
                cf2[nb] = n2
                q2.append(nb)
                if nb in vis1:
                    return merge_paths(cf1, cf2, nb), 0

    return None, 0

# ============================================================================
# VISUALIZATION (SAFE)
# ============================================================================
def plot_route(G, path, name):
    valid = [path[0]]

    for u, v in zip(path[:-1], path[1:]):
        if G.get_edge_data(u, v) is not None:
            valid.append(v)

    if len(valid) < 2:
        print(name, "invalid path")
        return

    fig, ax = ox.plot_graph(
        G,
        bgcolor="black",
        node_size=5,
        edge_color="gray",
        show=False,
        close=False
    )

    ox.plot_graph_route(
        G,
        valid,
        route_color="cyan",
        route_linewidth=4,
        node_size=0,
        ax=ax,
        show=False,
        close=True
    )

    fig.savefig(f"{OUTPUT_FOLDER}/{name}.png", dpi=300)
    plt.close(fig)

# ============================================================================
# RUN
# ============================================================================
def run_all():
    G = load_mirpur_map()

    # IMPORTANT FIX
    G = G.to_undirected()

    start = ox.distance.nearest_nodes(G, 90.3545, 23.7997)
    goal = ox.distance.nearest_nodes(G, 90.3683, 23.8069)

    algos = [
        ("BFS", lambda: bfs(G, start, goal)),
        ("DFS", lambda: dfs(G, start, goal)),
        ("UCS", lambda: ucs(G, start, goal)),
        ("Greedy", lambda: greedy(G, start, goal)),
        ("WeightedA*", lambda: weighted_a_star(G, start, goal)),
        ("IDDFS", lambda: iddfs(G, start, goal)),
        ("BiBFS", lambda: bidirectional_bfs(G, start, goal)),
    ]

    for name, func in algos:
        print("Running:", name)
        t0 = time.time()

        path, _ = func()

        print(f"{name} done in {(time.time()-t0)*1000:.2f} ms")

        if path:
            plot_route(G, path, name)

    print("Saved in:", OUTPUT_FOLDER)

# ============================================================================
if __name__ == "__main__":
    run_all()