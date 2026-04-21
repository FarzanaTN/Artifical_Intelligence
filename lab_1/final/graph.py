

import math
import random

import networkx as nx
import osmnx as ox

random.seed(42)


def edge_risk_multiplier(traffic: float, safety: float,
                         gender: float, age: float) -> float:
   
    return 1 + (0.40 * traffic + 0.25 * safety + 0.20 * gender + 0.15 * age)



def heuristic(graph, node, goal) -> float:
   
    n1 = graph.nodes[node]
    n2 = graph.nodes[goal]
    dist = math.sqrt((n1["x"] - n2["x"]) ** 2 +
                     (n1["y"] - n2["y"]) ** 2) * 111_000   # degrees → metres

    # Gather cost-per-metre from outgoing edges
    if graph.is_directed():
        out_edges = list(graph.out_edges(node, data=True))
    else:
        out_edges = [(node, v, d)
                     for v, d in graph[node].items()
                     for d in (d if isinstance(d, dict) else [d]).values()]

    if out_edges:
        cpm = []
        for _, _, edata in out_edges:
            if isinstance(edata, dict):
                length = edata.get("length", 1)
                cost   = edata.get("cost", length)
                cpm.append(cost / max(length, 1))
        local_risk = sum(cpm) / len(cpm) if cpm else 1.0
    else:
        local_risk = 1.0

    return dist + local_risk



def load_mirpur_map():
    """
    Download the Mirpur driving network (1 km radius) via OSMnx,
    keep the largest strongly-connected component, then annotate every
    edge with four synthetic risk attributes and a composite cost.

    Edge attributes added:
        traffic  float [0,1]  — congestion index
        safety   float [0,1]  — incident/crime index
        gender   float [0,1]  — gender-safety index
        age      float [0,1]  — age-safety index
        cost     float        — length × risk_multiplier  (used as g(n))

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    print("Fetching Mirpur map data ...")
    center = (23.8041, 90.3625)
    G = ox.graph_from_point(center, dist=1000, network_type="drive")

    try:
        G = ox.truncate.largest_component(G, strongly=True)
    except Exception:
        largest = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    for u, v, k, data in G.edges(data=True, keys=True):
        length          = data.get("length", 1)
        traffic         = random.uniform(0.0, 1.0)
        safety          = random.uniform(0.0, 1.0)
        gender          = random.uniform(0.0, 1.0)
        age             = random.uniform(0.0, 1.0)
        data["traffic"] = traffic
        data["safety"]  = safety
        data["gender"]  = gender
        data["age"]     = age
        data["cost"]    = length * edge_risk_multiplier(traffic, safety,
                                                        gender, age)

    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    return G


def reconstruct_path(came_from: dict, current) -> list:
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return path[::-1]


def path_cost(graph, path: list) -> float:
    return sum(
        graph[path[i]][path[i + 1]][0]["cost"]
        for i in range(len(path) - 1)
    )
