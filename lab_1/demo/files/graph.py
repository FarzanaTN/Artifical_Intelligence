"""
graph.py
========
Downloads the REAL Mirpur road network from OpenStreetMap using OSMnx.
Wraps it into Node / Edge dataclasses with all required fields.
Assigns synthetic safety/traffic/gender/age metrics based on OSM road tags.

Run once — graph is cached to disk as mirpur_graph.graphml so
subsequent runs don't need internet.
"""

import os
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import osmnx as ox
ox.settings.use_cache = True
ox.settings.log_console = True
import networkx as nx

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
CACHE_FILE  = os.path.join(os.path.dirname(__file__), "mirpur_graph.graphml")
BBOX = (23.820, 23.800, 90.370, 90.350)# SOURCE: Mirpur-2 / Sher-e-Bangla National Cricket Stadium
# DEST  : Pallabi Bus Stand
SRC_LAT, SRC_LON = 23.8025, 90.3573
DST_LAT, DST_LON = 23.8191, 90.3653

random.seed(42)   # reproducible synthetic metrics


# ─────────────────────────────────────────────────────────────────────
# NODE DATACLASS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Node:
    id:          int        # OSM node id (osmid)
    latitude:    float
    longitude:   float
    name:        str        # from OSM 'name' tag, else auto-generated
    street_count: int       # number of connecting roads (from OSMnx)
    highway_tag: str        # OSM highway tag if node is a junction
    x:           float      # local km east  (projected)
    y:           float      # local km north (projected)

    def __hash__(self):   return hash(self.id)
    def __eq__(self, o):  return isinstance(o, Node) and self.id == o.id
    def __repr__(self):   return f"Node({self.id}, '{self.name}')"
    def short(self):      return f"[{self.id}] {self.name}"


# ─────────────────────────────────────────────────────────────────────
# EDGE DATACLASS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Edge:
    src:           int        # OSM node id
    dst:           int        # OSM node id
    osmid:         object     # OSM way id(s)
    name:          str        # road name from OSM
    highway:       str        # OSM highway tag (primary/secondary/residential/…)
    length_m:      float      # true road length in metres (from OSMnx)
    oneway:        bool

    # ── Derived / synthetic metrics (1-10 scale) ──────────────────────
    base_cost:     float      # travel-time proxy = length_m / speed_kmh * 60
    safety:        int        # inferred from highway tag + lighting proxy
    lighting:      int        # estimated from road type
    gender_safe:   int        # estimated safety for female traveler
    age_ease:      int        # ease for elderly/disabled
    traffic:       int        # estimated congestion
    road_quality:  int        # estimated surface quality

    def __repr__(self):
        return (f"Edge({self.src}→{self.dst} | {self.name} | "
                f"{self.highway} | {self.length_m:.0f}m | cost={self.base_cost:.2f})")


# ─────────────────────────────────────────────────────────────────────
# METRIC ASSIGNMENT (based on OSM highway tag)
# ─────────────────────────────────────────────────────────────────────
# Typical speeds for Dhaka roads (km/h)
SPEED = {
    "motorway": 60, "trunk": 50, "primary": 40, "secondary": 30,
    "tertiary": 25, "unclassified": 20, "residential": 20,
    "living_street": 15, "service": 15, "footway": 5,
    "pedestrian": 5, "path": 5, "cycleway": 12, "default": 20,
}

# (safety, lighting, gender_safe, age_ease, traffic, road_quality)
# Base values per road type — randomised ±1 for variety
BASE_METRICS = {
    "motorway":        (7, 8, 6, 5, 9, 2),
    "trunk":           (7, 8, 6, 5, 8, 8),
    "primary":         (9, 8, 8, 7, 8, 8),
    "secondary":       (7, 7, 7, 6, 7, 7),
    "tertiary":        (6, 6, 6, 6, 6, 7),
    "unclassified":    (5, 5, 5, 5, 5, 5),
    "residential":     (6, 6, 6, 7, 4, 6),
    "living_street":   (7, 6, 7, 8, 3, 6),
    "service":         (5, 5, 5, 5, 3, 5),
    "footway":         (8, 6, 8, 7, 2, 5),
    "pedestrian":      (9, 7, 9, 8, 2, 3),
    "path":            (5, 4, 5, 4, 1, 4),
    "cycleway":        (7, 6, 7, 6, 2, 1),
    "default":         (5, 5, 5, 5, 5, 0),
}

#Adds ±1 noise so roads don’t all look identical. This makes AI experiments less deterministic.
# def _jitter(v: int, lo=1, hi=10) -> int:
#     return max(lo, min(hi, v + random.randint(-1, 1)))

def _jitter(v: int, lo=1, hi=10) -> int:
    # Increase variance to +/- 3 to force algorithms to choose 
    # a "safer" long path vs a "dangerous" short path.
    return max(lo, min(hi, v + random.randint(-3, 3)))


def _metrics_for(highway: str):
    key = highway if highway in BASE_METRICS else "default"
    s, li, g, a, tr, rq = BASE_METRICS[key]
    return (_jitter(s), _jitter(li), _jitter(g), _jitter(a), _jitter(tr), _jitter(rq))


def _speed(highway: str) -> float:
    return SPEED.get(highway, SPEED["default"])


def _base_cost(length_m: float, highway: str) -> float:
    """Travel time in minutes."""
    speed_ms = _speed(highway) * 1000 / 60   # m/min
    return round(length_m / speed_ms, 4)

# def _safety_penalty(s, li, g, a, tr, rq):
#     return (
#         (10 - s) +
#         (10 - li) +
#         (10 - g) +
#         (10 - a) +
#         (10 - rq) +
#         tr
#     ) / 6

# ─────────────────────────────────────────────────────────────────────
# COORDINATE CONVERSION
# ─────────────────────────────────────────────────────────────────────
LAT0 = BBOX[1]   # south
LON0 = BBOX[3]   # west
KM_LAT = 111.0
KM_LON = 111.0 * math.cos(math.radians(23.81))

#Converts GPS → local flat plane (km). Faster Euclidean heuristic for A*

def _xy(lat, lon):
    return round((lon - LON0) * KM_LON, 5), round((lat - LAT0) * KM_LAT, 5)


# ─────────────────────────────────────────────────────────────────────
# CITY GRAPH
# ─────────────────────────────────────────────────────────────────────
class CityGraph:
    """
    Wraps OSMnx MultiDiGraph into Node / Edge dataclasses.
    Simplified to undirected for pathfinding (takes min cost per pair).
    """

    def __init__(self):
        self.nodes:  dict[int, Node]       = {}
        self.edges:  List[Edge]            = []
        self.adj:    dict[int, list]       = {}   # id → [(nbr_id, Edge)]
        self._osm_G: nx.MultiDiGraph       = None
        self.source: int                   = None
        self.goal:   int                   = None
        self._build()

    # ── Download / load ───────────────────────────────────────────────
    def _build(self):
        print("  Loading OSM road network...")
        if os.path.exists(CACHE_FILE):
            print(f"  (Using cached graph: {CACHE_FILE})")
            G = ox.load_graphml(CACHE_FILE)
        # else:
        #     print("  Downloading from OpenStreetMap (first run only)...")
        #     G = ox.graph_from_place(
        #         "Mirpur, Dhaka, Bangladesh",
        #         network_type="drive",
        #         simplify=True

        #     )
        #     ox.save_graphml(G, CACHE_FILE)
        #     print(f"  Saved to {CACHE_FILE}")
        
        center_point = (23.8100, 90.3640) 
        G = ox.graph_from_point(
            center_point, 
            dist=1500, # 2.5km radius
            network_type="drive", 
            simplify=True
        )
        ox.save_graphml(G, CACHE_FILE)

        self._osm_G = G

        # Project for distance calc. Converts GPS → metric projection system.
        G_proj = ox.project_graph(G)

        # ── Build Node objects ─────────────────────────────────────────
        for osmid, data in G.nodes(data=True):
            lat = data.get("y", 0)
            lon = data.get("x", 0)
            x, y = _xy(lat, lon)
            self.nodes[osmid] = Node(
                id=osmid,
                latitude=lat,
                longitude=lon,
                name=data.get("name", f"Node-{str(osmid)[-4:]}"),
                street_count=data.get("street_count", 0),
                highway_tag=data.get("highway", ""),
                x=x, y=y,
            )
            self.adj[osmid] = []

        # ── Build Edge objects (undirected: keep min cost per pair) ────
        seen_pairs: dict[tuple, Edge] = {}

        for u, v, data in G.edges(data=True):
            if u not in self.nodes or v not in self.nodes:
                continue
            hw = data.get("highway", "default")
            if isinstance(hw, list):
                hw = hw[0]

            length  = data.get("length", 50.0)
            bc      = _base_cost(length, hw)
            # s, li, g, a, tr, rq = _metrics_for(hw)

            # time_cost = _base_cost(length, hw)
            # safety_pen = _safety_penalty(s, li, g, a, tr, rq)

            # bc = 0.2 * time_cost + 0.8 * safety_pen
            
            name    = data.get("name", "")
            if isinstance(name, list):
                name = name[0] if name else ""
            name    = name or f"{hw.capitalize()} road"

            s, li, g, a, tr, rq = _metrics_for(hw)

            e = Edge(
                src=u, dst=v,
                osmid=data.get("osmid", 0),
                name=name,
                highway=hw,
                length_m=length,
                oneway=data.get("oneway", False),
                base_cost=bc,
                safety=s, lighting=li, gender_safe=g,
                age_ease=a, traffic=tr, road_quality=rq,
            )

            pair = (min(u, v), max(u, v))
            if pair not in seen_pairs or e.base_cost < seen_pairs[pair].base_cost:
                seen_pairs[pair] = e

        for pair, e in seen_pairs.items():
            u, v = pair
            self.edges.append(e)
            self.adj[u].append((v, e))
            self.adj[v].append((u, e))

        # ── Nearest nodes to source / destination ─────────────────────
        self.source = ox.nearest_nodes(G, SRC_LON, SRC_LAT)
        self.goal   = ox.nearest_nodes(G, DST_LON, DST_LAT)

        print(f"  Graph loaded: {len(self.nodes)} nodes, {len(self.edges)} edges")
        print(f"  Source : {self.source}  ({self.nodes[self.source].latitude:.4f},"
              f" {self.nodes[self.source].longitude:.4f})")
        print(f"  Goal   : {self.goal}  ({self.nodes[self.goal].latitude:.4f},"
              f" {self.nodes[self.goal].longitude:.4f})")

    # ── Accessors ─────────────────────────────────────────────────────
    def neighbors(self, node_id: int) -> list:
        return self.adj.get(node_id, [])

    def node(self, nid: int) -> Node:
        return self.nodes[nid]

    def name(self, nid: int) -> str:
        return self.nodes[nid].name

    def euclidean(self, a: int, b: int) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        dx = na.x - nb.x
        dy = na.y - nb.y
        return math.sqrt(dx * dx + dy * dy)

    def osm_graph(self) -> nx.MultiDiGraph:
        return self._osm_G
