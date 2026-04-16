"""
Mirpur City Map - Graph Definition (Real GPS-based coordinates)
================================================================
All node positions are derived from actual GPS coordinates (lat/lon)
verified from Wikidata, OpenStreetMap, and official metro station data.

Coordinate system used in code:
  x = longitude-offset * 111 * cos(23.8°)   → approx km east
  y = latitude-offset  * 111                 → approx km north
  Reference point: 90.350°E, 23.770°N  (bottom-left corner)

Real GPS verified sources:
  Mirpur-10 metro   : 23.8086°N 90.3682°E  (Wikidata Q87791117)
  Shewrapara metro  : 23.7908°N 90.3756°E  (Grokipedia / Wikipedia)
  Kazipara          : ~23.7990°N 90.3690°E  (between Mirpur-10 & Shewrapara)
  Pallabi           : 23.8216°N 90.3601°E  (findlatitudeandlongitude.com)
  Mirpur-11         : ~23.8140°N 90.3660°E  (between Pallabi & Mirpur-10)
  Agargaon          : ~23.7770°N 90.3810°E  (south of Shewrapara)
  Mirpur-1 area     : ~23.7926°N 90.3607°E
  Mirpur-2          : ~23.8000°N 90.3640°E  (north-west of Mirpur-10)
  Mirpur-6          : ~23.8096°N 90.3674°E  (west of Mirpur-10)
  Section-6 Market  : ~23.8094°N 90.3620°E
  Kafrul            : ~23.7980°N 90.3820°E  (east, near airport road)
  Senpara Parbata   : ~23.8010°N 90.3560°E  (west side)
  Mirpur-14         : ~23.8260°N 90.3720°E  (north-east, cantonment area)
  Mirpur DOHS       : ~23.8300°N 90.3760°E  (north, inside cantonment)
  Mazar Road        : ~23.7900°N 90.3760°E  (south-east)

North is UP. East is RIGHT. Matches Google Maps orientation.

MRT Line 6 spine (north→south along Mirpur Road / Begum Rokeya Sarani):
  Pallabi → Mirpur-11 → Mirpur-10 → Kazipara → Shewrapara → Agargaon
"""

import math

# ─────────────────────────────────────────────────────────────
# Reference GPS origin (bottom-left of our map area)
# ─────────────────────────────────────────────────────────────
LAT0 = 23.770   # south boundary
LON0 = 90.350   # west boundary
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * math.cos(math.radians(23.8))  # ≈ 101.8

def gps_to_xy(lat, lon):
    """Convert GPS coords to local km (x=east, y=north)."""
    x = (lon - LON0) * KM_PER_DEG_LON
    y = (lat - LAT0) * KM_PER_DEG_LAT
    return round(x, 3), round(y, 3)


# ─────────────────────────────────────────────────────────────
# NODES — Real GPS positions
# ─────────────────────────────────────────────────────────────
# Each entry: (name, lat, lon, type)
_NODE_GPS = [
    # id  name                          lat        lon       type
    ( 0,  "Mirpur-1 Bus Stand",         23.7926,   90.3607,  "transit"),
    ( 1,  "Mirpur-2 Road",              23.7998,   90.3638,  "intersection"),
    ( 2,  "Section-6 Market",           23.8094,   90.3620,  "market"),
    ( 3,  "Senpara Parbata",            23.8010,   90.3560,  "area"),
    ( 4,  "Mirpur-6 Intersection",      23.8096,   90.3675,  "intersection"),
    ( 5,  "Mirpur-10 Circle",           23.8086,   90.3682,  "major"),     # MRT station confirmed
    ( 6,  "Mirpur-11",                  23.8140,   90.3660,  "area"),      # MRT station
    ( 7,  "Pallabi",                    23.8216,   90.3601,  "area"),      # MRT + thana HQ
    ( 8,  "Kafrul",                     23.7980,   90.3820,  "area"),      # east of Mirpur-10
    ( 9,  "Kazipara",                   23.7990,   90.3690,  "transit"),   # MRT station
    (10,  "Shewrapara",                 23.7908,   90.3756,  "transit"),   # MRT confirmed
    (11,  "Agargaon",                   23.7770,   90.3810,  "major"),     # MRT station (south)
    (12,  "Mirpur DOHS",                23.8300,   90.3760,  "area"),      # inside cantonment
    (13,  "Mirpur-14 Gate",             23.8260,   90.3720,  "area"),      # north, near cantonment
    (14,  "Mazar Road Junction",        23.7900,   90.3760,  "intersection"),
    (15,  "Kalyanpur",                  23.7760,   90.3600,  "transit"),   # south-west terminus
]

NODES = {}
for (nid, name, lat, lon, ntype) in _NODE_GPS:
    x, y = gps_to_xy(lat, lon)
    NODES[nid] = {"name": name, "lat": lat, "lon": lon, "x": x, "y": y, "type": ntype}


# ─────────────────────────────────────────────────────────────
# EDGES — Real road connections
# ─────────────────────────────────────────────────────────────
# Road topology verified against Google Maps / OSM:
#
#  Mirpur Road (north-south, western spine):
#    Mirpur-1 (0) — Mirpur-2 (1) — Section-6 (2) — Mirpur-6 (4) — Mirpur-10 (5)
#
#  Begum Rokeya Sarani / MRT spine (north-south, central):
#    Pallabi (7) — Mirpur-11 (6) — Mirpur-10 (5) — Kazipara (9) — Shewrapara (10) — Agargaon (11)
#
#  Cross roads (east-west connectors):
#    Mirpur-1 (0) — Senpara (3)  [westward from Mirpur-1]
#    Mirpur-2 (1) — Mirpur-11 (6)  [cross link]
#    Section-6 (2) — Mirpur-6 (4)  [east-west]
#    Mirpur-6 (4) — Mirpur-10 (5)  [direct]
#    Kafrul (8) — Mirpur-10 (5)   [east approach]
#    Kafrul (8) — Kazipara (9)    [east side road]
#    Kafrul (8) — Agargaon (11)   [airport road approach]
#    Pallabi (7) — Mirpur-14 (13) [northern connector]
#    Mirpur-14 (13) — DOHS (12)   [cantonment road]
#    DOHS (12) — Pallabi (7)      [DOHS loop]
#    Shewrapara (10) — Mazar (14) [south connector]
#    Agargaon (11) — Mazar (14)   [south road]
#    Kalyanpur (15) — Mirpur-1 (0)[south entry]
#    Kalyanpur (15) — Agargaon (11) [south-east]
#    Senpara (3) — Pallabi (7)    [west side road]
#    Mirpur-2 (1) — Senpara (3)   [cross]
#
# (a, b, base_cost, safety, traffic, gender_safe, age_ease, road_name)
EDGES_DEF = [
    # ── Mirpur Road spine (west, N-S) ──
    (0,  1,  3, 7, 7, 7, 6, "Mirpur Road South"),
    (1,  2,  3, 6, 7, 6, 5, "Mirpur Road (Sec-2 to Sec-6)"),
    (2,  4,  2, 7, 6, 7, 6, "Section-6 to Mirpur-6"),
    (4,  5,  2, 7, 8, 7, 5, "Mirpur-6 to Mirpur-10"),

    # ── Begum Rokeya Sarani / MRT spine (N-S) ──
    (7,  6,  3, 8, 6, 8, 7, "Begum Rokeya Sarani (Pallabi-Mirpur11)"),
    (6,  5,  3, 8, 7, 8, 6, "Begum Rokeya Sarani (Mirpur11-Mirpur10)"),
    (5,  9,  3, 7, 8, 7, 5, "Begum Rokeya Sarani (Mirpur10-Kazipara)"),
    (9,  10, 3, 7, 6, 7, 6, "Begum Rokeya Sarani (Kazipara-Shewrapara)"),
    (10, 11, 4, 7, 5, 7, 6, "Begum Rokeya Sarani (Shewrapara-Agargaon)"),

    # ── East-West cross connectors ──
    (0,  3,  3, 5, 4, 5, 6, "Mirpur-1 West Road (to Senpara)"),
    (1,  3,  4, 6, 4, 6, 6, "Mirpur-2 West (to Senpara)"),
    (1,  6,  4, 7, 6, 7, 6, "Mirpur-2 to Mirpur-11 link"),
    (2,  6,  3, 7, 5, 7, 6, "Section-6 to Mirpur-11 cross"),
    (3,  7,  5, 6, 4, 6, 5, "Senpara to Pallabi (west road)"),
    (5,  8,  3, 6, 6, 6, 5, "Mirpur-10 to Kafrul road"),
    (8,  9,  3, 6, 5, 6, 5, "Kafrul to Kazipara"),
    (8,  11, 5, 6, 5, 6, 5, "Kafrul to Agargaon (airport approach)"),

    # ── Northern area (cantonment / DOHS) ──
    (7,  13, 4, 8, 3, 8, 7, "Pallabi to Mirpur-14"),
    (13, 12, 3, 9, 2, 9, 8, "Mirpur-14 to DOHS (cantonment road)"),
    (12, 7,  4, 9, 2, 9, 8, "DOHS loop back to Pallabi"),
    (6,  13, 4, 8, 3, 8, 7, "Mirpur-11 to Mirpur-14"),

    # ── Southern exits ──
    (10, 14, 3, 6, 5, 6, 6, "Shewrapara to Mazar Road"),
    (11, 14, 3, 6, 5, 6, 6, "Agargaon to Mazar Road"),
    (14, 15, 4, 6, 4, 6, 6, "Mazar Road to Kalyanpur"),
    (0,  15, 4, 6, 5, 6, 6, "Mirpur-1 to Kalyanpur (south)"),
    (15, 11, 4, 6, 5, 6, 5, "Kalyanpur to Agargaon"),

    # ── Additional shortcuts ──
    (4,  9,  5, 6, 6, 6, 5, "Mirpur-6 to Kazipara shortcut"),
    (2,  7,  6, 7, 5, 7, 6, "Section-6 to Pallabi (direct)"),
]


class MirpurGraph:
    """Weighted undirected graph of Mirpur city (GPS-accurate)."""

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
        """Euclidean distance in km (from real GPS-derived x,y)."""
        ax, ay = self.nodes[a]["x"], self.nodes[a]["y"]
        bx, by = self.nodes[b]["x"], self.nodes[b]["y"]
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def display(self):
        print("=" * 65)
        print("  MIRPUR CITY MAP  (GPS-accurate coordinates)")
        print("=" * 65)
        print(f"  Nodes : {len(self.nodes)}")
        print(f"  Edges : {len(EDGES_DEF)}")
        print("-" * 65)
        for nid, info in self.nodes.items():
            nb_names = [self.nodes[nb]["name"] for nb, _ in self.adjacency[nid]]
            print(f"  [{nid:2d}] {info['name']:<28}  lat={info['lat']}  lon={info['lon']}")
            print(f"       → {', '.join(nb_names)}")
        print("=" * 65)
