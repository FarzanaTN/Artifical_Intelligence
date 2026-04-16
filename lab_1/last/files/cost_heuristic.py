import math
import osmnx as ox

def cost_fn(G, u, v, data):
    return data.get("length", 1)


def heuristic_fn(G, node, goal):
    # straight-line distance (admissible heuristic)
    x1, y1 = G.nodes[node]['x'], G.nodes[node]['y']
    x2, y2 = G.nodes[goal]['x'], G.nodes[goal]['y']

    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)