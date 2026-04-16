import osmnx as ox
import networkx as nx
import random
from collections import deque
import math

# =============================
# LOAD GRAPH (FAST)
# =============================
print("Loading graph...")
G = ox.load_graphml("mirpur.graphml")
print("Graph loaded!")

# =============================
# ADD EDGE FEATURES
# =============================
for u, v, d in G.edges(data=True):
    d['cost'] = float(d.get('length', 1))
    d['safety'] = random.uniform(0, 1)
    d['gender'] = random.uniform(0, 1)
    d['age'] = random.uniform(0, 1)

# =============================
# WEIGHT FUNCTION
# =============================
def compute_weight(d, w1=0.5, w2=0.3, w3=0.2):
    return (w1 * d['cost'] +
            w2 * (1 - d['safety']) +
            w3 * (1 - d['gender']))

# Apply weights
for u, v, d in G.edges(data=True):
    d['weight'] = compute_weight(d)

# =============================
# SOURCE & DESTINATION
# =============================
nodes = list(G.nodes)

src = nodes[0]
dest = nodes[len(nodes)//2]

print("Source:", src)
print("Destination:", dest)

# =============================
# BFS (Uninformed)
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

bfs_path, bfs_nodes = bfs(G, src, dest)
print("BFS visited nodes:", bfs_nodes)

# =============================
# DIJKSTRA
# =============================
dijkstra_path = nx.dijkstra_path(G, src, dest, weight='weight')

# Count nodes visited (approx)
dijkstra_nodes = len(dijkstra_path)

print("Dijkstra path length:", len(dijkstra_path))

# =============================
# A* SEARCH
# =============================
def heuristic(u, v):
    # Use Euclidean distance (BETTER than 0)
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

astar_path = nx.astar_path(G, src, dest, heuristic=heuristic, weight='weight')

astar_nodes = len(astar_path)

print("A* path length:", len(astar_path))

# =============================
# PATH COST
# =============================
def path_cost(graph, path):
    return sum(graph[u][v][0]['weight'] for u, v in zip(path[:-1], path[1:]))

print("Dijkstra cost:", path_cost(G, dijkstra_path))
print("A* cost:", path_cost(G, astar_path))

# =============================
# RESULTS COMPARISON
# =============================
print("\n=== COMPARISON ===")
print(f"BFS -> Nodes Visited: {bfs_nodes}")
print(f"Dijkstra -> Path Length: {len(dijkstra_path)}, Nodes ~ {dijkstra_nodes}")
print(f"A* -> Path Length: {len(astar_path)}, Nodes ~ {astar_nodes}")

# =============================
# VISUALIZATION
# =============================
ox.plot_graph_route(G, astar_path)