import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import math

# ----------------------------------------------------------------------------
# 1. LOAD MEDIUM GRAPH (Mirpur area)
# ----------------------------------------------------------------------------
print("Loading OSM graph (Mirpur)...")

place = "Mirpur, Dhaka, Bangladesh"

# Medium graph: drive network, simplified
G = ox.graph_from_place(place, network_type='drive', simplify=True)

# Make graph undirected for simplicity
G = ox.convert.to_undirected(G)
print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

# ----------------------------------------------------------------------------
# 2. DEFINE SOURCE & DESTINATION (real coordinates)
# ----------------------------------------------------------------------------
# Mirpur-1 & Pallabi approximate coordinates

mirpur1_coord = (23.8100, 90.3567)
pallabi_coord = (23.8330, 90.3627)

SOURCE = ox.distance.nearest_nodes(G, mirpur1_coord[1], mirpur1_coord[0])
DEST   = ox.distance.nearest_nodes(G, pallabi_coord[1], pallabi_coord[0])

print("Source node:", SOURCE)
print("Destination node:", DEST)

# ----------------------------------------------------------------------------
# 3. ASSIGN RISK ATTRIBUTES
# ----------------------------------------------------------------------------
def assign_risk(G):
    for u, v, data in G.edges(data=True):

        hw = data.get("highway", "residential")

        # Normalize highway type
        if isinstance(hw, list):
            hw = hw[0]

        base = {
            "primary": 0.6,
            "secondary": 0.5,
            "tertiary": 0.4,
            "residential": 0.3,
            "service": 0.7
        }.get(hw, 0.5)

        noise = random.uniform(-0.1, 0.1)
        risk = np.clip(base + noise, 0.1, 0.9)

        data["risk"] = risk

assign_risk(G)

# ----------------------------------------------------------------------------
# 4. COST FUNCTION
# ----------------------------------------------------------------------------
W_RISK = 0.4

def cost(u, v):
    data = G[u][v][0]  # OSMnx multi-edge
    dist = data.get("length", 50)
    risk = data.get("risk", 0.5)
    return dist * (1 + W_RISK * risk)

# ----------------------------------------------------------------------------
# 5. HEURISTIC FUNCTION
# ----------------------------------------------------------------------------
def heuristic(n1, n2):
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5 * 111000  # approx meters

# ----------------------------------------------------------------------------
# 6. A* SEARCH
# ----------------------------------------------------------------------------
def astar(start, goal):
    pq = [(0, start)]
    came = {start: None}
    g_cost = {start: 0}

    while pq:
        _, node = heapq.heappop(pq)

        if node == goal:
            path = []
            while node:
                path.append(node)
                node = came[node]
            return path[::-1]

        for nb in G.neighbors(node):
            new_g = g_cost[node] + cost(node, nb)

            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                f = new_g + heuristic(nb, goal)
                heapq.heappush(pq, (f, nb))
                came[nb] = node

    return None

# ----------------------------------------------------------------------------
# 7. RUN A*
# ----------------------------------------------------------------------------
print("Running A*...")

path = astar(SOURCE, DEST)

if path:
    print("Path found:", len(path), "nodes")
else:
    print("No path found")

# ----------------------------------------------------------------------------
# 8. VISUALIZATION
# ----------------------------------------------------------------------------
fig, ax = ox.plot_graph(G, node_size=0, edge_color="#999999",
                       bgcolor="black", show=False, close=False)

if path:
    ox.plot_graph_route(G, path, route_linewidth=4,
                        route_color="cyan", ax=ax)

plt.title("A* Path: Mirpur-1 → Pallabi", color="white")
plt.show()