import osmnx as ox
import networkx as nx
import random
from collections import deque
import math
import matplotlib.pyplot as plt

# -----------------------------
# LOAD GRAPH
# -----------------------------
print("Loading graph...")
G = ox.load_graphml("mirpur.graphml")
print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges\n")

# -----------------------------
# ADD EDGE METRICS
# -----------------------------
random.seed(42)  # reproducible metrics
for u, v, d in G.edges(data=True):
    d['cost'] = float(d.get('length', 1))  # Use length as base cost
    d['safety'] = random.uniform(0, 1)  # Random safety metric (higher = safer)
    d['traffic'] = random.uniform(0, 1)  # Random traffic metric
    d['risk'] = 1 - d['safety'] + d['traffic']  # Risk = inverse of safety + traffic

# -----------------------------
# RISK FUNCTION
# -----------------------------
def risk_function(safety, traffic):
    return 1 - safety + traffic

# -----------------------------
# HEURISTIC FUNCTION
# -----------------------------
def heuristic(G, node, goal):
    x1, y1 = G.nodes[node]['x'], G.nodes[node]['y']
    x2, y2 = G.nodes[goal]['x'], G.nodes[goal]['y']
    euclidean_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return euclidean_distance

# -----------------------------
# ACTUAL COST FUNCTION
# -----------------------------
def actual_cost(G, u, v):
    edge_data = G[u][v][0]  # Get edge attributes
    distance = edge_data.get('length', 1)
    risk = edge_data.get('risk', 1)
    return distance + risk

# -----------------------------
# SEARCH ALGORITHMS
# -----------------------------

# BFS (Uninformed)
def bfs(G, source, goal):
    visited = set()
    queue = deque([(source, [source])])
    while queue:
        node, path = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path, len(visited)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None, len(visited)

# DFS (Uninformed)
def dfs(G, source, goal):
    visited = set()
    stack = [(source, [source])]
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path, len(visited)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None, len(visited)

# A* (Informed)
def a_star(G, source, goal):
    visited = set()
    queue = [(0, source, [source])]  # (cost, node, path)
    while queue:
        cost, node, path = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path, len(visited)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                edge_cost = actual_cost(G, node, neighbor)
                h = heuristic(G, neighbor, goal)
                queue.append((cost + edge_cost + h, neighbor, path + [neighbor]))
                queue.sort()  # Sort by cost
    return None, len(visited)

# Greedy (Informed)
def greedy(G, source, goal):
    visited = set()
    queue = [(heuristic(G, source, goal), source, [source])]  # (heuristic, node, path)
    while queue:
        _, node, path = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path, len(visited)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                queue.append((heuristic(G, neighbor, goal), neighbor, path + [neighbor]))
                queue.sort()  # Sort by heuristic
    return None, len(visited)

# -----------------------------
# VISUALIZATION
# -----------------------------
def save_path_image(G, path, filename):
    fig, ax = ox.plot_graph_route(G, path, route_linewidth=2, node_size=10, show=False, close=False)
    filepath = f"output/{filename}"
    plt.savefig(filepath)
    plt.close()
    print(f"Path saved to {filepath}")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Select source and goal nodes
    source = list(G.nodes())[0]  # Example source node
    goal = list(G.nodes())[-1]  # Example goal node

    print(f"Source: {source}, Goal: {goal}\n")

    # Run BFS
    print("Running BFS...")
    bfs_path, bfs_visited = bfs(G, source, goal)
    save_path_image(G, bfs_path, "bfs_path.png")

    # Run DFS
    print("Running DFS...")
    dfs_path, dfs_visited = dfs(G, source, goal)
    save_path_image(G, dfs_path, "dfs_path.png")

    # Run A*
    print("Running A*...")
    a_star_path, a_star_visited = a_star(G, source, goal)
    save_path_image(G, a_star_path, "a_star_path.png")

    # Run Greedy
    print("Running Greedy...")
    greedy_path, greedy_visited = greedy(G, source, goal)
    save_path_image(G, greedy_path, "greedy_path.png")

    # Compare results
    print("\nComparison:")
    print(f"BFS: Nodes Visited = {bfs_visited}")
    print(f"DFS: Nodes Visited = {dfs_visited}")
    print(f"A*: Nodes Visited = {a_star_visited}")
    print(f"Greedy: Nodes Visited = {greedy_visited}")