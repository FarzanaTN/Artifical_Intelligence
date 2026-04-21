import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import heapq
import time
import math
import random
import numpy as np

# ============================================================================
# SETUP
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "assignment_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("Output folder:", OUTPUT_FOLDER)

random.seed(42)

# ============================================================================
# COLOUR PALETTE
# ============================================================================
PALETTE = {
    "bg":           "#0d1117",
    "panel":        "#161b22",
    "border":       "#30363d",
    "accent":       "#58a6ff",
    "danger":       "#f85149",
    "success":      "#3fb950",
    "warning":      "#d29922",
    "purple":       "#bc8cff",
    "orange":       "#ffa657",
    "text":         "#e6edf3",
    "subtext":      "#8b949e",
    # Street graph tiers
    "node_regular": "#2a3440",
    "node_src":     "#3fb950",   # start  — bright green
    "node_dst":     "#f85149",   # goal   — bright red
    "node_meet":    "#ffa657",   # meeting point — orange (bidirectional)
    # Per-algo route colours
    "route_bfs":    "#58a6ff",
    "route_dfs":    "#3fb950",
    "route_ucs":    "#ffa657",
    "route_greedy": "#f85149",
    "route_astar":  "#00d2ff",
    "route_wastar": "#bc8cff",
    "route_dls":    "#79c0ff",
    "route_iddfs":  "#56d364",
    "route_bds":    "#ffb547",
    "route_bdastar":"#ff7b72",
    "route_idastar":"#d2a8ff",
}

ALGO_COLORS = {
    "BFS":        PALETTE["route_bfs"],
    "DFS":        PALETTE["route_dfs"],
    "UCS":        PALETTE["route_ucs"],
    "Greedy":     PALETTE["route_greedy"],
    "A*":         PALETTE["route_astar"],
    "WeightedA*": PALETTE["route_wastar"],
    "DLS":        PALETTE["route_dls"],
    "IDDFS":      PALETTE["route_iddfs"],
    "BiDi":       PALETTE["route_bds"],
    "BiDiA*":     PALETTE["route_bdastar"],
    "IDA*":       PALETTE["route_idastar"],
}

# ============================================================================
# 1. RISK MODEL  (traffic + safety + gender + age)
# ============================================================================

def edge_risk_multiplier(traffic, safety, gender, age):
    """
    Composite risk: R = 1 + 0.40·traffic + 0.25·safety + 0.20·gender + 0.15·age
    All inputs normalised to [0, 1].
    """
    return 1 + (0.40 * traffic + 0.25 * safety + 0.20 * gender + 0.15 * age)


# ============================================================================
# 2. DYNAMIC HEURISTIC
# ============================================================================

def heuristic(graph, node, goal):
    """
    h(n) = euclidean_distance(n, goal) * local_avg_cost_per_metre(n)

    The local risk factor is derived from actual outgoing edges of `node`
    instead of a constant multiplier, making h(n) spatially aware.
    """
    n1 = graph.nodes[node]
    n2 = graph.nodes[goal]
    dist = math.sqrt((n1["x"] - n2["x"])**2 + (n1["y"] - n2["y"])**2) * 111_000

    out_edges = list(graph.out_edges(node, data=True)) if graph.is_directed() \
                else [(node, v, d) for v, d in graph[node].items()
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

    return dist * local_risk


# ============================================================================
# 3. LOAD GRAPH
# ============================================================================

def load_mirpur_map():
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
        data["cost"]    = length * edge_risk_multiplier(traffic, safety, gender, age)

    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    return G


# ============================================================================
# 4. HELPERS
# ============================================================================

def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return path[::-1]


def path_cost(graph, path):
    return sum(graph[path[i]][path[i+1]][0]["cost"] for i in range(len(path)-1))


# ============================================================================
# 5. ALGORITHMS
#    All return (path, nodes_explored, meeting_node_or_None)
# ============================================================================

# ── BFS ─────────────────────────────────────────────────────────────────────
# f(n): NONE — pure FIFO queue ordered by hop-count.
# g(n) = 0, h(n) = 0 effectively.
# ----------------------------------------------------------------------------
def bfs(graph, start, goal):
    q, vis, cf = [start], {start}, {start: None}
    count = 0
    while q:
        node = q.pop(0); count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n); cf[n] = node; q.append(n)
    return None, count, None


# ── DFS ─────────────────────────────────────────────────────────────────────
# f(n): NONE — pure LIFO stack, dives deep ignoring cost.
# ----------------------------------------------------------------------------
def dfs(graph, start, goal):
    stack, vis, cf = [start], {start}, {start: None}
    count = 0
    while stack:
        node = stack.pop(); count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n); cf[n] = node; stack.append(n)
    return None, count, None


# ── DLS ─────────────────────────────────────────────────────────────────────
# f(n): NONE — DFS capped at hop-depth `limit`.
# ----------------------------------------------------------------------------
def dls(graph, start, goal, limit):
    stack = [(start, 0)]; vis = {start: 0}; cf = {start: None}; count = 0
    while stack:
        node, depth = stack.pop(); count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        if depth < limit:
            for n in graph.neighbors(node):
                if n not in vis:
                    vis[n] = depth+1; cf[n] = node
                    stack.append((n, depth+1))
    return None, count, None


# ── IDDFS ────────────────────────────────────────────────────────────────────
# f(n): NONE — runs DLS with increasing depth limit.
# BFS-complete + DFS-memory.
# ----------------------------------------------------------------------------
def iddfs(graph, start, goal, max_depth=60):
    total = 0
    for limit in range(1, max_depth+1):
        path, count, _ = dls(graph, start, goal, limit)
        total += count
        if path is not None:
            return path, total, None
    return None, total, None


# ── UCS ──────────────────────────────────────────────────────────────────────
# f(n) = g(n)  [h(n) = 0].
# Priority queue ordered by accumulated risk-weighted cost.
# Optimal (Dijkstra-equivalent).
# ----------------------------------------------------------------------------
def ucs(graph, start, goal):
    pq = [(0, start)]; g = {start: 0}; cf = {start: None}
    visited = {}; count = 0
    while pq:
        cost, node = heapq.heappop(pq)
        if node in visited: continue
        visited[node] = cost; count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            nc = cost + graph[node][n][0]["cost"]
            if n not in g or nc < g[n]:
                g[n] = nc; cf[n] = node
                heapq.heappush(pq, (nc, n))
    return None, count, None


# ── Greedy BFS ───────────────────────────────────────────────────────────────
# f(n) = h(n)  [g(n) = 0].
# Fast but not optimal — ignores accumulated cost.
# ----------------------------------------------------------------------------
def greedy_bfs(graph, start, goal):
    pq = [(heuristic(graph, start, goal), start)]
    vis = {start}; cf = {start: None}; count = 0
    while pq:
        _, node = heapq.heappop(pq); count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n); cf[n] = node
                heapq.heappush(pq, (heuristic(graph, n, goal), n))
    return None, count, None


# ── A* ───────────────────────────────────────────────────────────────────────
# f(n) = g(n) + h(n)   ← the canonical full A* formula.
#
#   g(n) : exact risk-weighted cost from start to n  (accumulated edge costs)
#   h(n) : dynamic heuristic — geodesic distance × local risk factor
#   f(n) : priority used to order the open set
#
# At each step we pop the node with lowest f(n).
# For each neighbour v:
#   new_g = g(current) + cost(current → v)          ← g(v) candidate
#   f(v)  = new_g + h(v, goal)                       ← f(n) = g(n) + h(n)
# We update only if new_g improves the known g(v).
#
# Optimal when h(n) is admissible (never overestimates true cost to goal).
# ----------------------------------------------------------------------------
def a_star(graph, start, goal):
    pq = [(heuristic(graph, start, goal), start)]
    g  = {start: 0}
    cf = {start: None}
    count = 0
    while pq:
        f_val, node = heapq.heappop(pq); count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]   # g(n)
            if n not in g or new_g < g[n]:
                g[n]  = new_g
                f     = new_g + heuristic(graph, n, goal)  # f = g + h
                cf[n] = node
                heapq.heappush(pq, (f, n))
    return None, count, None


# ── Weighted A* ──────────────────────────────────────────────────────────────
# f(n) = g(n) + W·h(n).  W > 1 inflates h → greedier, fewer nodes expanded,
# path cost within W × optimal.  W = 1 reduces to standard A*.
# ----------------------------------------------------------------------------
def weighted_a_star(graph, start, goal, weight=1.5):
    pq = [(0, start)]; g = {start: 0}; cf = {start: None}; count = 0
    while pq:
        _, node = heapq.heappop(pq); count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]
            if n not in g or new_g < g[n]:
                g[n]  = new_g
                f     = new_g + weight * heuristic(graph, n, goal)
                cf[n] = node
                heapq.heappush(pq, (f, n))
    return None, count, None


# ── Bidirectional Dijkstra (BiDi) ────────────────────────────────────────────
# Forward frontier : f(n) = g_fwd(n)
# Backward frontier: f(n) = g_bwd(n)
# They expand simultaneously; algorithm stops when their sum ≥ best known.
# Returns meeting node so it can be highlighted on the map.
# ----------------------------------------------------------------------------
def bidirectional_dijkstra(graph, start, goal):
    pq_f = [(0, start)]; g_f = {start: 0}; cf_f = {start: None}; vis_f = {}
    rev_G = graph.reverse(copy=False)
    pq_b = [(0, goal)];  g_b = {goal: 0};  cf_b = {goal: None};  vis_b = {}
    best = math.inf; meet = None; count = 0

    while pq_f or pq_b:
        if pq_f:
            c, u = heapq.heappop(pq_f)
            if u not in vis_f:
                vis_f[u] = c; count += 1
                if u in vis_b and c + vis_b[u] < best:
                    best = c + vis_b[u]; meet = u
                for v in graph.neighbors(u):
                    nc = c + graph[u][v][0]["cost"]
                    if v not in g_f or nc < g_f[v]:
                        g_f[v] = nc; cf_f[v] = u; heapq.heappush(pq_f, (nc, v))
        if pq_b:
            c, u = heapq.heappop(pq_b)
            if u not in vis_b:
                vis_b[u] = c; count += 1
                if u in vis_f and vis_f[u] + c < best:
                    best = vis_f[u] + c; meet = u
                for v in rev_G.neighbors(u):
                    nc = c + rev_G[u][v][0]["cost"]
                    if v not in g_b or nc < g_b[v]:
                        g_b[v] = nc; cf_b[v] = u; heapq.heappush(pq_b, (nc, v))
        if pq_f and pq_b and pq_f[0][0] + pq_b[0][0] >= best:
            break

    if meet is None:
        return None, count, None
    fwd = reconstruct_path(cf_f, meet)
    bwd = reconstruct_path(cf_b, meet)
    return fwd + bwd[-2::-1], count, meet


# ── Bidirectional A* (BiDi A*) ───────────────────────────────────────────────
# Forward : f(n) = g_fwd(n) + h(n → goal)
# Backward: f(n) = g_bwd(n) + h(n → start)
# Returns meeting node.
# ----------------------------------------------------------------------------
def bidirectional_astar(graph, start, goal):
    pq_f = [(heuristic(graph, start, goal), start)]
    g_f  = {start: 0}; cf_f = {start: None}; vis_f = {}
    rev_G = graph.reverse(copy=False)
    pq_b = [(heuristic(graph, goal, start), goal)]
    g_b  = {goal: 0};  cf_b = {goal: None};  vis_b = {}
    best = math.inf; meet = None; count = 0

    while pq_f or pq_b:
        if pq_f:
            _, u = heapq.heappop(pq_f)
            if u not in vis_f:
                vis_f[u] = g_f[u]; count += 1
                if u in vis_b:
                    cand = g_f[u] + g_b[u]
                    if cand < best: best = cand; meet = u
                for v in graph.neighbors(u):
                    ng = g_f[u] + graph[u][v][0]["cost"]
                    if v not in g_f or ng < g_f[v]:
                        g_f[v] = ng; cf_f[v] = u
                        heapq.heappush(pq_f, (ng + heuristic(graph, v, goal), v))
        if pq_b:
            _, u = heapq.heappop(pq_b)
            if u not in vis_b:
                vis_b[u] = g_b[u]; count += 1
                if u in vis_f:
                    cand = g_f.get(u, math.inf) + g_b[u]
                    if cand < best: best = cand; meet = u
                for v in rev_G.neighbors(u):
                    ng = g_b[u] + rev_G[u][v][0]["cost"]
                    if v not in g_b or ng < g_b[v]:
                        g_b[v] = ng; cf_b[v] = u
                        heapq.heappush(pq_b, (ng + heuristic(graph, v, start), v))
        if pq_f and pq_b and pq_f[0][0] + pq_b[0][0] >= best + 1e-9:
            break

    if meet is None:
        return None, count, None
    fwd = reconstruct_path(cf_f, meet)
    bwd = reconstruct_path(cf_b, meet)
    return fwd + bwd[-2::-1], count, meet


# ── IDA* ─────────────────────────────────────────────────────────────────────
# f(n) = g(n) + h(n) evaluated at EVERY node.
# Threshold raised to min-f-exceeding value each iteration.  O(path) memory.
# ----------------------------------------------------------------------------
def ida_star(graph, start, goal):
    threshold = heuristic(graph, start, goal)
    path = [start]; count = [0]

    def search(g_cost, thresh):
        node = path[-1]
        f = g_cost + heuristic(graph, node, goal); count[0] += 1
        if f > thresh: return f
        if node == goal: return "FOUND"
        minimum = math.inf
        for n in graph.neighbors(node):
            if n in path: continue
            path.append(n)
            res = search(g_cost + graph[node][n][0]["cost"], thresh)
            if res == "FOUND": return "FOUND"
            if res < minimum: minimum = res
            path.pop()
        return minimum

    for _ in range(500):
        res = search(0, threshold)
        if res == "FOUND": return list(path), count[0], None
        if res == math.inf: return None, count[0], None
        threshold = res
    return None, count[0], None


# ============================================================================
# 6. VISUALISATION
# ============================================================================

# All streets rendered in one flat colour — clear on dark background
STREET_COLOR  = "#4a6741"   # muted green-grey, visible on #0d1117
STREET_CASING = "#0d1117"   # matches bg — separates parallel roads
STREET_WIDTH  = 1.4
NODE_COLOR    = "#5a7a6e"


def _draw_base_graph(G, ax, dim_factor=1.0):
    """
    Flat uniform street renderer — every road the same colour.
    Two passes: dark casing then coloured fill so parallel roads
    don't bleed together.
    """
    from matplotlib.collections import LineCollection

    segs = [[[G.nodes[u]["x"], G.nodes[u]["y"]],
             [G.nodes[v]["x"], G.nodes[v]["y"]]]
            for u, v in G.edges()]

    # Pass 1 — casing (dark outline to separate adjacent roads)
    ax.add_collection(LineCollection(
        segs, linewidths=STREET_WIDTH * 2.8,
        colors=STREET_CASING, alpha=dim_factor, zorder=1))

    # Pass 2 — coloured fill
    ax.add_collection(LineCollection(
        segs, linewidths=STREET_WIDTH,
        colors=STREET_COLOR, alpha=0.92 * dim_factor, zorder=2))

    # Intersection dots
    xs = [G.nodes[n]["x"] for n in G.nodes]
    ys = [G.nodes[n]["y"] for n in G.nodes]
    ax.scatter(xs, ys, s=5 * dim_factor, c=NODE_COLOR,
               alpha=0.80 * dim_factor, zorder=3, linewidths=0)

    # Fix axis limits (LineCollection does not auto-scale)
    margin_x = (max(xs) - min(xs)) * 0.03
    margin_y = (max(ys) - min(ys)) * 0.03
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)


def _draw_route_overlay(G, ax, path, color):
    """
    Draw a route as a glowing line:
    - Wide soft halo (same colour, very transparent)
    - Medium bright line
    - Thin white core for crispness
    """
    from matplotlib.collections import LineCollection

    segs = [[[G.nodes[path[i]]["x"], G.nodes[path[i]]["y"]],
             [G.nodes[path[i+1]]["x"], G.nodes[path[i+1]]["y"]]]
            for i in range(len(path) - 1)]

    # Halo
    ax.add_collection(LineCollection(segs, linewidths=9,  colors=color, alpha=0.18, zorder=10))
    # Main coloured line
    ax.add_collection(LineCollection(segs, linewidths=3.8, colors=color, alpha=0.95, zorder=11))
    # Bright core
    ax.add_collection(LineCollection(segs, linewidths=1.0, colors="white", alpha=0.55, zorder=12))

    # Dots at every route node
    xs = [G.nodes[n]["x"] for n in path]
    ys = [G.nodes[n]["y"] for n in path]
    ax.scatter(xs, ys, s=16, c=color, alpha=0.55, zorder=13, linewidths=0)


def _draw_special_nodes(G, ax, start, goal, meet=None):
    """Overlay start (green ▲), goal (red ★) and meeting point (orange ◆)."""
    # Glow ring behind each marker
    for node, color, size in [
        (start, PALETTE["node_src"], 420),
        (goal,  PALETTE["node_dst"], 420),
    ]:
        ax.scatter([G.nodes[node]["x"]], [G.nodes[node]["y"]],
                   s=size * 1.8, c=color, alpha=0.18, zorder=18, linewidths=0)

    ax.scatter([G.nodes[start]["x"]], [G.nodes[start]["y"]],
               s=280, c=PALETTE["node_src"], marker="^",
               edgecolors="white", linewidths=1.8, zorder=20)
    ax.scatter([G.nodes[goal]["x"]], [G.nodes[goal]["y"]],
               s=340, c=PALETTE["node_dst"], marker="*",
               edgecolors="white", linewidths=1.8, zorder=20)

    if meet is not None and meet not in (start, goal):
        ax.scatter([G.nodes[meet]["x"]], [G.nodes[meet]["y"]],
                   s=340, c=PALETTE["node_meet"], alpha=0.20,
                   zorder=18, linewidths=0)
        ax.scatter([G.nodes[meet]["x"]], [G.nodes[meet]["y"]],
                   s=200, c=PALETTE["node_meet"], marker="D",
                   edgecolors="white", linewidths=1.8, zorder=20)


def _style_ax(ax, title):
    """Apply consistent dark-theme axis styling."""
    ax.set_aspect("equal")
    ax.set_title(title, color=PALETTE["text"], fontsize=14,
                 fontweight="bold", fontfamily="monospace", pad=14)
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["subtext"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.set_xlabel("Longitude", color=PALETTE["subtext"], fontsize=8)
    ax.set_ylabel("Latitude",  color=PALETTE["subtext"], fontsize=8)


def plot_route(G, path, name, start, goal, meet=None, color=None):
    """
    Full route map with:
    - Layered street network (casing + fill + typed widths)
    - Glowing route line with white core
    - Glow-ringed start / goal / meeting markers
    """
    if color is None:
        color = ALGO_COLORS.get(name, PALETTE["accent"])

    fig, ax = plt.subplots(figsize=(14, 12), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    _draw_base_graph(G, ax, dim_factor=1.0)
    _draw_route_overlay(G, ax, path, color)
    _draw_special_nodes(G, ax, start, goal, meet)

    legend_elems = [
        mpatches.Patch(facecolor=color,               label="Route"),
        mpatches.Patch(facecolor=PALETTE["node_src"], label="Start  ▲"),
        mpatches.Patch(facecolor=PALETTE["node_dst"], label="Goal   ★"),
    ]
    if meet is not None and meet not in (start, goal):
        legend_elems.append(
            mpatches.Patch(facecolor=PALETTE["node_meet"], label="Meeting ◆"))

    ax.legend(handles=legend_elems, loc="lower left",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)

    _style_ax(ax, f"Algorithm: {name}   |   {len(path)} hops   |   cost = {path_cost(G, path):.0f}")
    fig.tight_layout(pad=1.5)
    out = os.path.join(OUTPUT_FOLDER, f"{name}_route.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_risk_heatmap(G, start, goal):
    """
    Risk heat-map: every edge coloured green→orange→red by composite cost.
    Road widths still reflect road type for realism.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    edge_list = list(G.edges(data=True))
    costs = np.array([d.get("cost", 1) for _, _, d in edge_list])
    norm  = Normalize(vmin=costs.min(), vmax=costs.max())
    cmap  = LinearSegmentedColormap.from_list(
        "risk", ["#3fb950", "#d29922", "#f85149"])

    fig, ax = plt.subplots(figsize=(14, 12), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Build segment list and per-segment risk colours
    segs = [[[G.nodes[u]["x"], G.nodes[u]["y"]],
             [G.nodes[v]["x"], G.nodes[v]["y"]]]
            for (u, v, _) in edge_list]
    colors_mapped = [cmap(norm(c)) for c in costs]

    # Pass 1 — dark casing
    ax.add_collection(LineCollection(
        segs, linewidths=STREET_WIDTH * 2.8,
        colors=STREET_CASING, alpha=0.95, zorder=1))

    # Pass 2 — risk-coloured fill
    ax.add_collection(LineCollection(
        segs, linewidths=STREET_WIDTH * 1.2,
        colors=colors_mapped, alpha=0.92, zorder=2))

    # Intersection nodes
    xs_all = [G.nodes[n]["x"] for n in G.nodes]
    ys_all = [G.nodes[n]["y"] for n in G.nodes]
    margin_x = (max(xs_all) - min(xs_all)) * 0.03
    margin_y = (max(ys_all) - min(ys_all)) * 0.03
    ax.set_xlim(min(xs_all) - margin_x, max(xs_all) + margin_x)
    ax.set_ylim(min(ys_all) - margin_y, max(ys_all) + margin_y)

    xs = [G.nodes[n]["x"] for n in G.nodes]
    ys = [G.nodes[n]["y"] for n in G.nodes]
    ax.scatter(xs, ys, s=6, c="#4a6a86", alpha=0.6, zorder=3, linewidths=0)

    _draw_special_nodes(G, ax, start, goal)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
    cbar.ax.set_ylabel("Composite Risk Cost", color=PALETTE["subtext"], fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["subtext"])

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=PALETTE["node_src"], label="Start  ▲"),
            mpatches.Patch(facecolor=PALETTE["node_dst"], label="Goal   ★"),
        ],
        loc="lower left", facecolor=PALETTE["panel"],
        edgecolor=PALETTE["border"], labelcolor=PALETTE["text"], fontsize=9)

    _style_ax(ax, "Risk Heat-Map  |  green = safe   orange = moderate   red = high")
    fig.tight_layout(pad=1.5)
    out = os.path.join(OUTPUT_FOLDER, "risk_heatmap.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_all_routes_overlay(G, algo_paths, start, goal):
    """All algorithm routes overlaid on the full street map."""
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(16, 14), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Base map at reduced brightness so routes pop
    _draw_base_graph(G, ax, dim_factor=0.55)

    legend_elems = []
    for name, path in algo_paths:
        if path is None:
            continue
        color = ALGO_COLORS.get(name, PALETTE["accent"])
        segs = [[[G.nodes[path[i]]["x"], G.nodes[path[i]]["y"]],
                 [G.nodes[path[i+1]]["x"], G.nodes[path[i+1]]["y"]]]
                for i in range(len(path) - 1)]
        ax.add_collection(LineCollection(segs, linewidths=4.5,
                                         colors=color, alpha=0.20, zorder=10))
        ax.add_collection(LineCollection(segs, linewidths=2.2,
                                         colors=color, alpha=0.88, zorder=11))
        legend_elems.append(mpatches.Patch(facecolor=color, label=name))

    _draw_special_nodes(G, ax, start, goal)

    legend_elems += [
        mpatches.Patch(facecolor=PALETTE["node_src"], label="Start  ▲"),
        mpatches.Patch(facecolor=PALETTE["node_dst"], label="Goal   ★"),
    ]
    ax.legend(handles=legend_elems, loc="lower left",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=8, ncol=2)

    _style_ax(ax, "All Algorithms — Route Overlay")
    fig.tight_layout(pad=1.5)
    out = os.path.join(OUTPUT_FOLDER, "all_routes_overlay.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_complexities(results):
    names  = [r["name"]  for r in results]
    times  = [r["time"]  for r in results]
    nodes  = [r["nodes"] for r in results]
    colors = [ALGO_COLORS.get(n, PALETTE["accent"]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(22, 7), facecolor=PALETTE["bg"])
    for ax, values, title, unit in zip(
        axes,
        [times, nodes],
        ["Time Complexity", "Space Complexity"],
        ["ms (log scale)", "Nodes Explored (log scale)"],
    ):
        ax.set_facecolor(PALETTE["panel"])
        bars = ax.bar(names, values, color=colors,
                      edgecolor=PALETTE["border"], linewidth=0.8, zorder=3)
        ax.set_yscale("log")
        ax.set_title(title, color=PALETTE["text"], fontsize=14,
                     fontweight="bold", fontfamily="monospace", pad=10)
        ax.set_xlabel("Algorithm", color=PALETTE["subtext"], fontsize=11)
        ax.set_ylabel(unit, color=PALETTE["subtext"], fontsize=10)
        ax.tick_params(axis="x", colors=PALETTE["subtext"], rotation=35)
        ax.tick_params(axis="y", colors=PALETTE["subtext"])
        ax.yaxis.grid(True, color=PALETTE["border"], linestyle="--",
                      alpha=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.15,
                    f"{val:.1f}" if val < 1000 else f"{int(val):,}",
                    ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=8, fontfamily="monospace")

    fig.tight_layout(pad=2.5)
    out = os.path.join(OUTPUT_FOLDER, "complexity_comparison.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_cost_comparison(results):
    """
    Bar chart comparing total path cost (risk-weighted distance) per algorithm.
    A dashed line marks the optimal (lowest) cost for easy reference.
    Bars for algorithms that found no path are hatched and labelled N/A.
    """
    names  = [r["name"] for r in results]
    costs  = [r["cost"] for r in results]
    colors = [ALGO_COLORS.get(n, PALETTE["accent"]) for n in names]

    finite = [c for c in costs if c != float("inf")]
    optimal     = min(finite) if finite else 0
    display_max = max(finite) * 1.12 if finite else 1
    display_costs = [c if c != float("inf") else display_max for c in costs]

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    bars = ax.bar(names, display_costs, color=colors,
                  edgecolor=PALETTE["border"], linewidth=0.8, zorder=3)

    for bar, raw in zip(bars, costs):
        if raw == float("inf"):
            bar.set_hatch("////")
            bar.set_alpha(0.45)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5, "N/A",
                    ha="center", va="center",
                    color=PALETTE["text"], fontsize=9,
                    fontfamily="monospace", fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + display_max * 0.01,
                    f"{raw:,.0f}",
                    ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=8.5,
                    fontfamily="monospace")

    # Optimal reference line
    ax.axhline(optimal, color=PALETTE["success"], linewidth=1.4,
               linestyle="--", zorder=4, alpha=0.85)
    ax.text(len(names) - 0.5, optimal * 1.012,
            f"Optimal: {optimal:,.0f}",
            color=PALETTE["success"], fontsize=8.5,
            fontfamily="monospace", ha="right", va="bottom")

    ax.set_ylim(0, display_max * 1.18)
    ax.set_title("Path Cost Comparison  (risk-weighted distance)",
                 color=PALETTE["text"], fontsize=14, fontweight="bold",
                 fontfamily="monospace", pad=12)
    ax.set_xlabel("Algorithm", color=PALETTE["subtext"], fontsize=11)
    ax.set_ylabel("Total Path Cost", color=PALETTE["subtext"], fontsize=10)
    ax.tick_params(axis="x", colors=PALETTE["subtext"], rotation=30)
    ax.tick_params(axis="y", colors=PALETTE["subtext"])
    ax.yaxis.grid(True, color=PALETTE["border"], linestyle="--",
                  alpha=0.45, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    fig.tight_layout(pad=2)
    out = os.path.join(OUTPUT_FOLDER, "cost_comparison.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_risk_metrics_distribution(G):
    metrics = {"Traffic": [], "Safety": [], "Gender": [], "Age": []}
    for _, _, data in G.edges(data=True):
        metrics["Traffic"].append(data.get("traffic", 0))
        metrics["Safety"].append(data.get("safety",  0))
        metrics["Gender"].append(data.get("gender",  0))
        metrics["Age"].append(data.get("age",    0))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    bp = ax.boxplot(
        list(metrics.values()), labels=list(metrics.keys()),
        patch_artist=True,
        medianprops=dict(color=PALETTE["warning"], linewidth=2),
        whiskerprops=dict(color=PALETTE["subtext"]),
        capprops=dict(color=PALETTE["subtext"]),
        flierprops=dict(markerfacecolor=PALETTE["danger"], marker="o", markersize=4))
    for patch, c in zip(bp["boxes"],
                        [PALETTE["accent"], PALETTE["success"],
                         PALETTE["purple"], PALETTE["orange"]]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_title("Risk Metric Distributions (per edge)",
                 color=PALETTE["text"], fontsize=14, fontweight="bold",
                 fontfamily="monospace", pad=10)
    ax.tick_params(colors=PALETTE["subtext"])
    ax.yaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.set_ylabel("Normalised Score [0, 1]", color=PALETTE["subtext"])
    fig.tight_layout(pad=2)
    out = os.path.join(OUTPUT_FOLDER, "risk_metrics_distribution.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# 7. MAIN
# ============================================================================

FN_NOTES = {
    "BFS":        "No f(n) — FIFO hop-count",
    "DFS":        "No f(n) — LIFO depth",
    "DLS":        "No f(n) — depth <= limit",
    "IDDFS":      "No f(n) — iterative depth limit",
    "UCS":        "f = g(n)",
    "Greedy":     "f = h(n)",
    "A*":         "f = g(n) + h(n)  [optimal]",
    "WeightedA*": "f = g(n) + 1.5*h(n)  [near-optimal]",
    "BiDi":       "f = g_fwd or g_bwd  [meet-in-middle]",
    "BiDiA*":     "f = g(n)+h(n) both directions",
    "IDA*":       "f = g(n)+h(n) threshold deepening",
}


def run_all():
    G     = load_mirpur_map()
    start = ox.distance.nearest_nodes(G, 90.3545, 23.7997)
    goal  = ox.distance.nearest_nodes(G, 90.3683, 23.8069)

    print(f"\nStart: Mirpur-1  |  Goal: Mirpur-10\n")

    algos = [
        ("BFS",        lambda: bfs(G, start, goal)),
        ("DFS",        lambda: dfs(G, start, goal)),
        ("DLS",        lambda: dls(G, start, goal, limit=40)),
        ("IDDFS",      lambda: iddfs(G, start, goal, max_depth=40)),
        ("UCS",        lambda: ucs(G, start, goal)),
        ("Greedy",     lambda: greedy_bfs(G, start, goal)),
        ("A*",         lambda: a_star(G, start, goal)),
        ("WeightedA*", lambda: weighted_a_star(G, start, goal, weight=1.5)),
        ("BiDi",       lambda: bidirectional_dijkstra(G, start, goal)),
        ("BiDiA*",     lambda: bidirectional_astar(G, start, goal)),
        ("IDA*",       lambda: ida_star(G, start, goal)),
    ]

    results    = []
    algo_paths = []

    for name, func in algos:
        print(f"Running {name} ...")
        t0          = time.time()
        path, count, meet = func()
        elapsed     = (time.time() - t0) * 1000

        cost = path_cost(G, path) if path else float("inf")
        results.append({"name": name, "time": elapsed, "nodes": count, "cost": cost})
        algo_paths.append((name, path))

        status = f"{len(path)} hops, cost={cost:.1f}" if path else "NO PATH"
        # print(f"  {name:<12} | {elapsed:7.1f} ms | {count:5} explored | {status}")
        print(f"  {name:<12} | {elapsed:7.1f} ms | {status}")


        if path:
            plot_route(G, path, name, start, goal, meet)

    print("\nGenerating summary plots ...")
    plot_complexities(results)
    plot_cost_comparison(results)
    plot_risk_heatmap(G, start, goal)
    plot_all_routes_overlay(G, algo_paths, start, goal)
    plot_risk_metrics_distribution(G)

    print("\n" + "=" * 82)
    # print(f"{'Algorithm':<14} {'Time(ms)':>9} {'Explored':>10} {'PathCost':>12}  f(n) formula")
    print(f"{'Algorithm':<14} {'Time(ms)':>9}  {'PathCost':>12}  f(n) formula")

    print("=" * 82)
    for r in results:
        n = r["name"]
        print(f"{n:<14} {r['time']:>9.1f} {r['nodes']:>10} {r['cost']:>12.1f}  {FN_NOTES.get(n,'')}")
    print("=" * 82)
    print(f"\nAll outputs -> {OUTPUT_FOLDER}")


if __name__ == "__main__":
    run_all()