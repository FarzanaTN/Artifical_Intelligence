"""
Mirpur City Map - Graph Definition
Nodes = Landmarks/Intersections
Edges = Roads with cost + metric attributes
"""

import math

# ─────────────────────────────────────────────
# NODES  (id, name, x, y)
# x, y are approximate map coordinates (used for heuristic)
# ─────────────────────────────────────────────
NODES = {
    0:  {"name": "Mirpur-1 Bus Stand",       "x": 1.0, "y": 5.0, "type": "transit"},
    1:  {"name": "Mirpur-2 Roundabout",      "x": 2.0, "y": 5.0, "type": "intersection"},
    2:  {"name": "Mirpur-10 Circle",         "x": 3.5, "y": 5.0, "type": "major"},
    3:  {"name": "Pallabi",                  "x": 5.0, "y": 5.5, "type": "area"},
    4:  {"name": "Section-6 Market",         "x": 1.0, "y": 3.5, "type": "market"},
    5:  {"name": "Mirpur-6 Intersection",    "x": 2.5, "y": 3.5, "type": "intersection"},
    6:  {"name": "Mirpur-11 Housing",        "x": 3.5, "y": 3.0, "type": "residential"},
    7:  {"name": "Kafrul",                   "x": 5.0, "y": 3.5, "type": "area"},
    8:  {"name": "Senpara Parbata",          "x": 1.0, "y": 2.0, "type": "area"},
    9:  {"name": "Shewrapara Metro",         "x": 2.5, "y": 2.0, "type": "transit"},
    10: {"name": "Mirpur DOHS",              "x": 3.5, "y": 2.0, "type": "area"},
    11: {"name": "Agargaon",                 "x": 5.0, "y": 2.0, "type": "major"},
    12: {"name": "Kazipara",                 "x": 1.5, "y": 0.8, "type": "transit"},
    13: {"name": "Mirpur-14 Gate",           "x": 3.0, "y": 0.8, "type": "area"},
    14: {"name": "BSCIC Industrial Area",    "x": 4.2, "y": 0.8, "type": "industrial"},
    15: {"name": "Mazar Road Junction",      "x": 5.5, "y": 1.0, "type": "intersection"},
}

# ─────────────────────────────────────────────
# EDGES  (node_a, node_b, attributes)
# base_cost   : raw travel cost (time/distance unit)
# safety      : 1-10  (10 = very safe)
# traffic     : 1-10  (10 = very congested)
# gender_safe : 1-10  (10 = safest for female traveler)
# age_ease    : 1-10  (10 = easiest for elderly)
# road_name   : label
# ─────────────────────────────────────────────
EDGES_DEF = [
    # (a, b, base_cost, safety, traffic, gender_safe, age_ease, road_name)
    (0,  1,  4, 7, 6,  7, 6, "Mirpur Main Road"),
    (1,  2,  5, 6, 8,  6, 5, "Mirpur Road North"),
    (2,  3,  4, 7, 6,  7, 6, "Pallabi Link Road"),
    (0,  4,  3, 6, 3,  6, 7, "Section-1 Road"),
    (1,  5,  4, 7, 6,  7, 6, "Section-2 Road"),
    (2,  6,  3, 8, 5,  8, 7, "Matikata Road"),
    (3,  7,  4, 6, 4,  6, 5, "Kafrul Link"),
    (4,  5,  3, 6, 4,  6, 6, "Cross Road A"),
    (5,  6,  4, 7, 7,  7, 6, "Cross Road B"),
    (6,  7,  3, 8, 5,  8, 6, "North Cross Road"),
    (4,  8,  4, 5, 3,  5, 6, "Senpara Road"),
    (5,  9,  5, 7, 6,  7, 6, "Shewrapara Road"),
    (6,  10, 4, 8, 5,  8, 7, "DOHS Link Road"),
    (7,  11, 5, 7, 6,  7, 6, "Agargaon Road"),
    (8,  9,  3, 6, 5,  6, 6, "Begum Rokeya Avenue"),
    (9,  10, 4, 7, 7,  7, 5, "Inner Connector Road"),
    (10, 11, 5, 8, 6,  8, 7, "DOHS-Agargaon Road"),
    (8,  12, 4, 6, 4,  6, 6, "Kazipara Road"),
    (9,  13, 5, 7, 5,  7, 6, "Central Link Road"),
    (10, 14, 4, 7, 4,  7, 5, "BSCIC Road"),
    (11, 15, 5, 6, 5,  6, 6, "Mazar Road"),
    (12, 13, 4, 6, 3,  6, 6, "Section Link"),
    (13, 14, 5, 7, 4,  7, 6, "Industrial Avenue"),
    (14, 15, 4, 7, 5,  7, 6, "East Exit Road"),
    (1,  9,  6, 7, 7,  7, 5, "Bypass Road A"),
    (3,  11, 7, 6, 6,  6, 5, "Bypass Road B"),
    (2,  10, 5, 8, 5,  8, 6, "Central Bypass"),
    (5,  13, 6, 7, 6,  7, 6, "Diagonal Link"),
    (7,  10, 5, 7, 5,  7, 6, "East-West Connector"),
    (3,  6,  4, 7, 5,  7, 6, "Northern Shortcut"),
]


class MirpurGraph:
    """Weighted undirected graph of Mirpur city."""

    def __init__(self):
        self.nodes = NODES
        self.adjacency = {n: [] for n in NODES}
        self._build(EDGES_DEF)

    def _build(self, edges):
        for (a, b, cost, safety, traffic, gender, age, name) in edges:
            attr = {
                "base_cost":   cost,
                "safety":      safety,
                "traffic":     traffic,
                "gender_safe": gender,
                "age_ease":    age,
                "road_name":   name,
            }
            self.adjacency[a].append((b, attr))
            self.adjacency[b].append((a, attr))

    def neighbors(self, node):
        return self.adjacency[node]

    def node_name(self, nid):
        return self.nodes[nid]["name"]

    def all_nodes(self):
        return list(self.nodes.keys())

    def euclidean(self, a, b):
        """Straight-line distance used as heuristic."""
        ax, ay = self.nodes[a]["x"], self.nodes[a]["y"]
        bx, by = self.nodes[b]["x"], self.nodes[b]["y"]
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def display(self):
        print("=" * 60)
        print("  MIRPUR CITY MAP")
        print("=" * 60)
        print(f"  Nodes : {len(self.nodes)}")
        print(f"  Edges : {len(EDGES_DEF)}")
        print("-" * 60)
        for nid, info in self.nodes.items():
            neighbors = [self.nodes[nb]["name"] for nb, _ in self.adjacency[nid]]
            print(f"  [{nid:2d}] {info['name']:<30} → {', '.join(neighbors)}")
        print("=" * 60)
