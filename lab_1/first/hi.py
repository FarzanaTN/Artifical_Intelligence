import osmnx as ox
import networkx as nx
import random
import math
from collections import deque
import pandas as pd

# =============================
# LOAD MODERATE MIRPUR MAP
# =============================
ox.settings.use_cache = True

print("Loading Mirpur map...")
point = (23.8041, 90.3668)  # Mirpur
G = ox.graph_from_point(point, dist=1000, network_type='drive')
print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges\n")

# =============================
# ADD EDGE METRICS
# =============================
random.seed(42)

for u, v, d in G.edges(data=True):
    d['distance'] = float(d.get('length', 1))
    d['safety'] = random.uniform(0, 1)
    d['traffic'] = random.uniform(0, 1)
    d['gender'] = random.uniform(0, 1)
    d['age'] = random.uniform(0, 1)
    d['time'] = d['distance'] * (1 + d['traffic'])

# =============================
# NORMALIZATION
# =============================
def normalize(attr):
    vals = [d[attr] for u, v, d in G.edges(data=True)]
    max_v = max(vals)
    min_v = min(vals)
    for u, v, d in G.edges(data=True):
        d[f'n_{attr}'] = (d[attr] - min_v) / (max_v - min_v + 1e-6)

for attr in ['distance', 'time', 'traffic']:
    normalize(attr)

# =============================
# UNIFIED COST FUNCTION (g(n))
# =============================
def compute_weight(d, w):
    return (
        w['distance'] * d['n_distance'] +
        w['time'] * d['n_time'] +
        w['traffic'] * d['n_traffic'] +
        w['safety'] * (1 - d['safety']) +
        w['gender'] * (1 - d['gender']) +
        w['age'] * (1 - d['age'])
    )

# =============================
# HEURISTIC FUNCTION (h(n))
# =============================
def heuristic(u, v):
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# =============================
# BFS (UNINFORMED)
# =============================
def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])
    count = 0

    while queue:
        node, path = queue.popleft()
        count += 1

        if node == goal:
            return path, count

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, count

# =============================
# PATH COST
# =============================
def path_cost(path):
    return sum(G[u][v][0]['weight'] for u, v in zip(path[:-1], path[1:]))

# =============================
# SOURCE & DESTINATION
# =============================
nodes = list(G.nodes)
src = nodes[0]
dest = nodes[len(nodes)//2]

print("Source:", src)
print("Destination:", dest)

# =============================
# SCENARIOS (SAME FUNCTION, DIFFERENT WEIGHTS)
# =============================
scenarios = {
    "Balanced": {'distance':0.3,'time':0.2,'traffic':0.2,'safety':0.2,'gender':0.05,'age':0.05},
    "Safe": {'distance':0.1,'time':0.1,'traffic':0.1,'safety':0.6,'gender':0.05,'age':0.05},
    "Fast": {'distance':0.4,'time':0.3,'traffic':0.2,'safety':0.05,'gender':0.025,'age':0.025}
}

results = []

# =============================
# RUN EXPERIMENTS
# =============================
for name, weights in scenarios.items():
    print(f"\n=== Scenario: {name} ===")

    # Apply weights
    for u, v, d in G.edges(data=True):
        d['weight'] = compute_weight(d, weights)

    # BFS
    bfs_path, bfs_nodes = bfs(G, src, dest)

    # Dijkstra
    dijkstra_path = nx.dijkstra_path(G, src, dest, weight='weight')

    # A*
    astar_path = nx.astar_path(G, src, dest, heuristic=heuristic, weight='weight')

    # Costs
    d_cost = path_cost(dijkstra_path)
    a_cost = path_cost(astar_path)

    print("BFS visited:", bfs_nodes)
    print("Dijkstra cost:", d_cost)
    print("A* cost:", a_cost)

    results.append([name, "BFS", bfs_nodes, len(bfs_path), "-"])
    results.append([name, "Dijkstra", len(dijkstra_path), len(dijkstra_path), round(d_cost,3)])
    results.append([name, "A*", len(astar_path), len(astar_path), round(a_cost,3)])

    # =============================
    # SAVE GRAPH IMAGE
    # =============================
    fig, ax = ox.plot_graph_route(G, astar_path, show=False, close=False)
    fig.savefig(f"{name}_path.png")

# =============================
# SAVE RESULTS
# =============================
df = pd.DataFrame(results, columns=["Scenario","Algorithm","NodesVisited","PathLength","Cost"])
df.to_csv("results.csv", index=False)

print("\nFinal Results:")
print(df)