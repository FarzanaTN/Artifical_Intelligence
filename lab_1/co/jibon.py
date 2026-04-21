import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import heapq
import time
import math
import random
import numpy as np



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "assignment_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("Output folder:", OUTPUT_FOLDER)

random.seed(42)

PALETTE = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "accent":    "#58a6ff",
    "danger":    "#f85149",
    "success":   "#3fb950",
    "warning":   "#d29922",
    "purple":    "#bc8cff",
    "orange":    "#ffa657",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
    "route_bfs":     "#58a6ff",
    "route_dfs":     "#3fb950",
    "route_ucs":     "#ffa657",
    "route_greedy":  "#f85149",
    "route_wastar":  "#bc8cff",
    "route_dls":     "#79c0ff",
    "route_iddfs":   "#56d364",
    "route_bds":     "#ffb547",
    "route_bdastar": "#ff7b72",
    "route_idastar": "#d2a8ff",
}

ALGO_COLORS = {
    "BFS":       PALETTE["route_bfs"],
    "DFS":       PALETTE["route_dfs"],
    "UCS":       PALETTE["route_ucs"],
    "Greedy":    PALETTE["route_greedy"],
    "WeightedA*":PALETTE["route_wastar"],
    "DLS":       PALETTE["route_dls"],
    "IDDFS":     PALETTE["route_iddfs"],
    "BiDi":      PALETTE["route_bds"],
    "BiDiA*":    PALETTE["route_bdastar"],
    "IDA*":      PALETTE["route_idastar"],
}


def edge_risk_multiplier(traffic, safety, gender, age):
    """
    Composite risk score driving g(n):
        R = 1 + w1*traffic + w2*safety + w3*gender + w4*age
    All sub-scores are normalised to [0, 1] before weighting.
    """
    w1, w2, w3, w4 = 0.40, 0.25, 0.20, 0.15
    return 1 + (w1 * traffic + w2 * safety + w3 * gender + w4 * age)


def heuristic(graph, node, goal):
    """
    h(n) = geodesic_distance(n, goal) × dynamic_risk_factor(n)

    Instead of a constant multiplier (1.3), we read the *average* risk of
    edges leaving 'node' and use that to scale the admissible distance
    estimate.  This keeps h(n) ≤ true cost more faithfully while still
    favouring lower-risk directions.

    Formula:
        h(n) = dist_euclidean(n, goal)_in_metres
               × (1 + local_avg_risk_penalty(n))

    The local risk penalty is computed as the mean of the normalised
    composite risk of all outgoing edges — so in high-risk neighbourhoods
    the heuristic pushes the algorithm to look harder.
    """
    n1 = graph.nodes[node]
    n2 = graph.nodes[goal]

    # Euclidean distance (approx metres via degree→metre conversion)
    dist = math.sqrt(
        (n1["x"] - n2["x"]) ** 2 +
        (n1["y"] - n2["y"]) ** 2
    ) * 111_000

    # Gather the raw cost-per-metre of outgoing edges
    out_edges = list(graph.out_edges(node, data=True)) if graph.is_directed() \
                else [(node, v, d) for v, d in graph[node].items()
                      for d in (d if isinstance(d, dict) else [d]).values()]

    if out_edges:
        costs_per_m = []
        for _, _, edata in out_edges:
            if isinstance(edata, dict):
                length = edata.get("length", 1)
                cost   = edata.get("cost", length)
                costs_per_m.append(cost / max(length, 1))
        local_risk = (sum(costs_per_m) / len(costs_per_m)) if costs_per_m else 1.0
    else:
        local_risk = 1.0  # no edges: neutral

    return dist * local_risk



# G = {
#     node1: {
#         neighbor1: {edge_data},
#         neighbor2: {edge_data}
#     },
#     node2: {
#         neighbor3: {edge_data}
#     }
# }

# G[101][205] = {
#     "length": 120.5,
#     "cost": 180.2,
#     "traffic": 0.6,
#     "safety": 0.3
# }



def load_mirpur_map():
    print("Fetching Mirpur map data …")
    center = (23.8041, 90.3625)
    G = ox.graph_from_point(center, dist=1000, network_type="drive")

    try:
        G = ox.truncate.largest_component(G, strongly=True)
    except Exception:
        largest = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    for u, v, k, data in G.edges(data=True, keys=True):
        length = data.get("length", 1)

        # --- synthetic risk attributes (deterministic via seeded RNG) ---
        traffic = random.uniform(0.0, 1.0)   # congestion index  [0,1]
        safety  = random.uniform(0.0, 1.0)   # incident index    [0,1]
        gender  = random.uniform(0.0, 1.0)   # gender-safety idx [0,1]
        age     = random.uniform(0.0, 1.0)   # age-safety index  [0,1]

        data["traffic"] = traffic
        data["safety"]  = safety
        data["gender"]  = gender
        data["age"]     = age

        # g(n): real accumulated cost = distance × composite risk
        data["cost"] = length * edge_risk_multiplier(traffic, safety, gender, age)

    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    return G



def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return path[::-1]



def path_cost(graph, path):
    total = 0
    for i in range(len(path) - 1):
        total += graph[path[i]][path[i + 1]][0]["cost"]
    return total


# ============================================================================
# 6. ALGORITHMS
# ============================================================================
# Each section explains HOW f(n) = g(n) + h(n) is (or isn't) used.
# ============================================================================

# ── BFS ─────────────────────────────────────────────────────────────────────
# f(n) usage: NONE.
# BFS ignores both cost and heuristic.  It uses a FIFO queue, so every node
# is reached in order of hop-count (number of edges), not distance or risk.
# g(n) = 0, h(n) = 0 effectively.
# ----------------------------------------------------------------------------
def bfs(graph, start, goal):
    q, vis, cf = [start], {start}, {start: None}
    count = 0
    while q:
        node = q.pop(0)
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                q.append(n)
    return None, count


# ── DFS ─────────────────────────────────────────────────────────────────────
# f(n) usage: NONE.
# DFS uses a LIFO stack; it dives deep along one path ignoring cost/risk.
# g(n) = 0, h(n) = 0 effectively.
# ----------------------------------------------------------------------------
def dfs(graph, start, goal):
    stack, vis, cf = [start], {start}, {start: None}
    count = 0
    while stack:
        node = stack.pop()
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                stack.append(n)
    return None, count


# ── DLS ─────────────────────────────────────────────────────────────────────
# f(n) usage: NONE (depth-limited DFS).
# Explores up to `limit` hops deep; no cost or heuristic.
# g(n) = hop count (implicit depth), h(n) = 0.
# ----------------------------------------------------------------------------

def dls(graph, start, goal, limit):
    # Stack stores tuples of (node, current_depth)
    stack = [(start, 0)]
    vis = {start: 0} # Track node: depth_found_at
    cf = {start: None}
    count = 0
    
    while stack:
        node, depth = stack.pop()
        count += 1
        
        if node == goal:
            return reconstruct_path(cf, goal), count
        
        # Only expand neighbors if we haven't hit the depth limit
        if depth < limit:
            for n in graph.neighbors(node):
                # Standard DLS allows revisiting nodes if found at a shallower depth,
                # but for a basic conversion, we check if it's unvisited.
                if n not in vis:
                    vis[n] = depth + 1
                    cf[n] = node
                    stack.append((n, depth + 1))
                    
    return None, count

# ── IDDFS ────────────────────────────────────────────────────────────────────
# f(n) usage: NONE (iterative deepening over hop depth).
# Runs DLS with increasing limits until a solution is found.
# Combines BFS's completeness with DFS's memory usage.
# g(n) = hop count (depth), h(n) = 0.
# ----------------------------------------------------------------------------
def iddfs(graph, start, goal, max_depth=60):
    """Iterative Deepening Depth-First Search."""
    total_count = 0
    for limit in range(1, max_depth + 1):
        path, count = dls(graph, start, goal, limit)
        total_count += count
        if path is not None:
            return path, total_count
    return None, total_count


# ── UCS ──────────────────────────────────────────────────────────────────────
# f(n) usage: f(n) = g(n)  [h(n) = 0].
# Priority queue ordered purely by accumulated cost g(n).
# Optimal when all edge costs are non-negative (Dijkstra-equivalent).
# The risk-weighted `data["cost"]` IS used — this is the most risk-aware
# uninformed algorithm.
# ----------------------------------------------------------------------------
def ucs(graph, start, goal):
    pq      = [(0, start)]
    g       = {start: 0}        #g(n)
    cf      = {start: None}
    visited = {}
    count   = 0
    while pq:
        cost, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited[node] = cost
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            new_cost = cost + graph[node][n][0]["cost"]
            if n not in g or new_cost < g[n]:
                g[n]  = new_cost
                cf[n] = node                    # parent
                heapq.heappush(pq, (new_cost, n))
    return None, count


# ── Greedy BFS ───────────────────────────────────────────────────────────────
# f(n) usage: f(n) = h(n)  [g(n) = 0].
# Priority queue ordered purely by heuristic h(n) — how close we *seem*
# to the goal.  Fast but not optimal (ignores accumulated risk cost).
# ----------------------------------------------------------------------------
def greedy_bfs(graph, start, goal):
    pq  = [(heuristic(graph, start, goal), start)]
    vis = {start}
    cf  = {start: None}
    count = 0
    while pq:
        _, node = heapq.heappop(pq)
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                heapq.heappush(pq, (heuristic(graph, n, goal), n))
    return None, count


# ── Weighted A* ──────────────────────────────────────────────────────────────
# f(n) usage: f(n) = g(n) + W × h(n)   (W = weight, default 1.5).
# Inflating h(n) by W makes the algorithm more greedy — fewer nodes explored
# but the path may be slightly sub-optimal (within a W factor of optimal).
# Full A* is the special case W = 1.
# ----------------------------------------------------------------------------
def weighted_a_star(graph, start, goal, weight=1.5):
    pq = [(0, start)]
    g  = {start: 0}
    cf = {start: None}
    count = 0
    while pq:
        _, node = heapq.heappop(pq)
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]
            if n not in g or new_g < g[n]:
                g[n]  = new_g
                f     = new_g + weight * heuristic(graph, n, goal)
                cf[n] = node
                heapq.heappush(pq, (f, n))
    return None, count


# ── Bidirectional Dijkstra (BiDi) ────────────────────────────────────────────
# f(n) usage: f(n) = g_fwd(n)  OR  f(n) = g_bwd(n)  [h(n) = 0].
# Two UCS frontiers expand simultaneously — one from start, one from goal.
# They meet in the middle.  Cost used = data["cost"] (risk-weighted).
# When frontiers meet at node m, optimal path cost = g_fwd[m] + g_bwd[m].
# ----------------------------------------------------------------------------
def bidirectional_dijkstra(graph, start, goal):
    # Forward
    pq_f  = [(0, start)];  g_f = {start: 0}     #best cost from start-> node
    cf_f = {start: None};  vis_f = {}
    # Backward (reversed graph)
    rev_G = graph.reverse(copy=False)
    pq_b  = [(0, goal)];   g_b = {goal: 0};   cf_b = {goal: None};   vis_b = {}

    best  = math.inf
    meet  = None
    count = 0

    while pq_f or pq_b:
        # --- forward step ---
        if pq_f:
            c, u = heapq.heappop(pq_f)
            if u not in vis_f:
                vis_f[u] = c;  count += 1
                if u in vis_b and c + vis_b[u] < best:
                    best = c + vis_b[u];  meet = u
                for v in graph.neighbors(u):
                    nc = c + graph[u][v][0]["cost"]
                    if v not in g_f or nc < g_f[v]:
                        g_f[v] = nc;  cf_f[v] = u
                        heapq.heappush(pq_f, (nc, v))

        # --- backward step ---
        if pq_b:
            c, u = heapq.heappop(pq_b)
            if u not in vis_b:
                vis_b[u] = c;  count += 1
                if u in vis_f and vis_f[u] + c < best:
                    best = vis_f[u] + c;  meet = u
                for v in rev_G.neighbors(u):
                    nc = c + rev_G[u][v][0]["cost"]
                    if v not in g_b or nc < g_b[v]:
                        g_b[v] = nc;  cf_b[v] = u
                        heapq.heappush(pq_b, (nc, v))

        # Termination: both frontiers expanded more than best
        if pq_f and pq_b:
            if pq_f[0][0] + pq_b[0][0] >= best:
                break

    if meet is None:
        return None, count

    # Reconstruct: fwd path to meet, then reverse bwd path from meet
    fwd = reconstruct_path(cf_f, meet)
    bwd = reconstruct_path(cf_b, meet)
    return fwd + bwd[-2::-1], count   # avoid duplicating 'meet'


# ── Bidirectional A* (BiDi A*) ───────────────────────────────────────────────
# f(n) usage:
#   Forward : f(n) = g_fwd(n) + h(n, goal)
#   Backward: f(n) = g_bwd(n) + h(n, start)
# Both frontiers use the same dynamic heuristic pointing at their respective
# targets.  Meeting in the middle with A* guidance is faster than plain BiDi.
# ----------------------------------------------------------------------------
def bidirectional_astar(graph, start, goal):
    pq_f  = [(heuristic(graph, start, goal), start)]
    g_f   = {start: 0};  cf_f = {start: None};  vis_f = {}

    rev_G = graph.reverse(copy=False)
    pq_b  = [(heuristic(graph, goal, start), goal)]
    g_b   = {goal: 0};   cf_b = {goal: None};   vis_b = {}

    best  = math.inf
    meet  = None
    count = 0

    while pq_f or pq_b:
        if pq_f:
            _, u = heapq.heappop(pq_f)
            if u not in vis_f:
                vis_f[u] = g_f[u];  count += 1
                if u in vis_b:
                    cand = g_f[u] + g_b[u]
                    if cand < best:
                        best = cand;  meet = u
                for v in graph.neighbors(u):
                    ng = g_f[u] + graph[u][v][0]["cost"]
                    if v not in g_f or ng < g_f[v]:
                        g_f[v] = ng;  cf_f[v] = u
                        heapq.heappush(pq_f, (ng + heuristic(graph, v, goal), v))

        if pq_b:
            _, u = heapq.heappop(pq_b)
            if u not in vis_b:
                vis_b[u] = g_b[u];  count += 1
                if u in vis_f:
                    cand = g_f.get(u, math.inf) + g_b[u]
                    if cand < best:
                        best = cand;  meet = u
                for v in rev_G.neighbors(u):
                    ng = g_b[u] + rev_G[u][v][0]["cost"]
                    if v not in g_b or ng < g_b[v]:
                        g_b[v] = ng;  cf_b[v] = u
                        heapq.heappush(pq_b, (ng + heuristic(graph, v, start), v))

        if pq_f and pq_b:
            if pq_f[0][0] + pq_b[0][0] >= best + 1e-9:
                break

    if meet is None:
        return None, count

    fwd = reconstruct_path(cf_f, meet)
    bwd = reconstruct_path(cf_b, meet)
    return fwd + bwd[-2::-1], count


# ── IDA* ─────────────────────────────────────────────────────────────────────
# f(n) usage: f(n) = g(n) + h(n)  — full A* evaluation at EVERY node.
# IDA* is A* with iterative-deepening over the f-cost threshold instead of
# hop depth.  Memory: O(path length) — only the current path is kept.
# Each iteration expands all nodes with f(n) ≤ threshold; threshold grows to
# the smallest f-value that exceeded the previous threshold.
# ----------------------------------------------------------------------------
def ida_star(graph, start, goal):
    threshold = heuristic(graph, start, goal)
    path      = [start]
    count     = [0]

    def search(g_cost, threshold):
        node = path[-1]
        f    = g_cost + heuristic(graph, node, goal)
        count[0] += 1

        if f > threshold:
            return f          # signal: new minimum threshold

        if node == goal:
            return "FOUND"

        minimum = math.inf
        for n in graph.neighbors(node):
            if n in path:     # avoid cycles on current path
                continue
            path.append(n)
            edge_cost = graph[node][n][0]["cost"]
            result    = search(g_cost + edge_cost, threshold)

            if result == "FOUND":
                return "FOUND"
            if result < minimum:
                minimum = result
            path.pop()

        return minimum

    for _ in range(500):       # safety cap on iterations
        result = search(0, threshold)
        if result == "FOUND":
            return list(path), count[0]
        if result == math.inf:
            return None, count[0]
        threshold = result

    return None, count[0]



def _dark_fig(figsize=(14, 10)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    return fig, ax


def plot_route(G, path, name, color=None):
    """Plot a single route on the Mirpur street graph with dark theme."""
    if color is None:
        color = ALGO_COLORS.get(name, PALETTE["accent"])

    # Node positions
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    edge_list = list(G.edges())

    fig, ax = _dark_fig(figsize=(14, 12))

    # --- Draw all streets (dim) ---
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=edge_list,
        edge_color=PALETTE["border"],
        width=0.6, alpha=0.45, arrows=False
    )

    # --- Draw route edges (bright) ---
    route_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=route_edges,
        edge_color=color,
        width=3.5, alpha=0.95, arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.05"
    )

    # --- Start / goal markers ---
    if path:
        sx, sy = G.nodes[path[0]]["x"],  G.nodes[path[0]]["y"]
        gx, gy = G.nodes[path[-1]]["x"], G.nodes[path[-1]]["y"]
        ax.scatter([sx], [sy], s=160, c=PALETTE["success"],  zorder=10, marker="^", edgecolors="white", linewidths=1.5)
        ax.scatter([gx], [gy], s=160, c=PALETTE["danger"],   zorder=10, marker="*", edgecolors="white", linewidths=1.5)

    # --- Labels ---
    ax.set_title(
        f"Algorithm: {name}   ·   {len(path)} nodes in path",
        color=PALETTE["text"], fontsize=15, fontweight="bold",
        pad=14, fontfamily="monospace"
    )
    ax.tick_params(colors=PALETTE["subtext"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    legend_elements = [
        mpatches.Patch(facecolor=color,              label="Route"),
        mpatches.Patch(facecolor=PALETTE["success"], label="Start ▲"),
        mpatches.Patch(facecolor=PALETTE["danger"],  label="Goal ★"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)

    fig.tight_layout(pad=1.5)
    out = os.path.join(OUTPUT_FOLDER, f"{name}_route.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_risk_heatmap(G):
    """Visualise the composite risk score on every edge as a colour gradient."""
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    edge_list = list(G.edges(data=True))

    # Normalise costs for colour mapping
    costs = [d.get("cost", 1) for _, _, d in edge_list]
    c_min, c_max = min(costs), max(costs)
    norm_costs = [(c - c_min) / (c_max - c_min + 1e-9) for c in costs]

    cmap = LinearSegmentedColormap.from_list(
        "risk", ["#3fb950", "#d29922", "#f85149"]
    )
    edge_colors = [cmap(nc) for nc in norm_costs]

    fig, ax = _dark_fig(figsize=(14, 12))
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in edge_list],
        edge_color=edge_colors,
        width=1.8, alpha=0.8, arrows=False
    )
    ax.set_title(
        "Risk Heat-Map  ·  green = safe   orange = moderate   red = high risk",
        color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14,
        fontfamily="monospace"
    )
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=c_min, vmax=c_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
    cbar.ax.set_ylabel("Composite Risk Cost", color=PALETTE["subtext"], fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["subtext"])

    fig.tight_layout(pad=1.5)
    out = os.path.join(OUTPUT_FOLDER, "risk_heatmap.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_all_routes_overlay(G, algo_paths):
    """All routes overlaid on one map for quick comparison."""
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    edge_list = list(G.edges())

    fig, ax = _dark_fig(figsize=(16, 13))
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=edge_list,
        edge_color=PALETTE["border"],
        width=0.5, alpha=0.35, arrows=False
    )

    legend_elements = []
    for name, path in algo_paths:
        if path is None:
            continue
        color = ALGO_COLORS.get(name, PALETTE["accent"])
        route_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=route_edges,
            edge_color=color,
            width=2.2, alpha=0.75, arrows=False
        )
        legend_elements.append(mpatches.Patch(facecolor=color, label=name))

    ax.set_title(
        "All Algorithms — Route Overlay",
        color=PALETTE["text"], fontsize=16, fontweight="bold",
        pad=14, fontfamily="monospace"
    )
    ax.legend(handles=legend_elements, loc="lower left",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9, ncol=2)
    fig.tight_layout(pad=1.5)
    out = os.path.join(OUTPUT_FOLDER, "all_routes_overlay.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_complexities(results):
    """Dark-themed bar charts for time and space complexity."""
    names  = [r["name"]  for r in results]
    times  = [r["time"]  for r in results]
    nodes  = [r["nodes"] for r in results]
    colors = [ALGO_COLORS.get(n, PALETTE["accent"]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=PALETTE["bg"])

    for ax, values, title, unit in zip(
        axes,
        [times, nodes],
        ["Time Complexity", "Space Complexity"],
        ["ms (log scale)", "Nodes Explored (log scale)"]
    ):
        ax.set_facecolor(PALETTE["panel"])
        bars = ax.bar(names, values, color=colors, edgecolor=PALETTE["border"],
                      linewidth=0.8, zorder=3)
        ax.set_yscale("log")
        ax.set_title(title, color=PALETTE["text"], fontsize=14, fontweight="bold",
                     fontfamily="monospace", pad=10)
        ax.set_xlabel("Algorithm", color=PALETTE["subtext"], fontsize=11)
        ax.set_ylabel(unit, color=PALETTE["subtext"], fontsize=10)
        ax.tick_params(axis="x", colors=PALETTE["subtext"], rotation=30)
        ax.tick_params(axis="y", colors=PALETTE["subtext"])
        ax.yaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                f"{val:.1f}" if val < 1000 else f"{int(val):,}",
                ha="center", va="bottom",
                color=PALETTE["text"], fontsize=8.5, fontfamily="monospace"
            )

    fig.tight_layout(pad=2.5)
    out = os.path.join(OUTPUT_FOLDER, "complexity_comparison.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_risk_metrics_distribution(G):
    """Box-plots of the four risk metrics across all edges."""
    metrics = {"Traffic": [], "Safety": [], "Gender": [], "Age": []}
    for _, _, data in G.edges(data=True):
        metrics["Traffic"].append(data.get("traffic", 0))
        metrics["Safety"].append(data.get("safety",  0))
        metrics["Gender"].append(data.get("gender",  0))
        metrics["Age"].append(data.get("age",    0))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    bp = ax.boxplot(
        list(metrics.values()),
        labels=list(metrics.keys()),
        patch_artist=True,
        medianprops=dict(color=PALETTE["warning"], linewidth=2),
        whiskerprops=dict(color=PALETTE["subtext"]),
        capprops=dict(color=PALETTE["subtext"]),
        flierprops=dict(markerfacecolor=PALETTE["danger"], marker="o", markersize=4)
    )
    box_colors = [PALETTE["accent"], PALETTE["success"], PALETTE["purple"], PALETTE["orange"]]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

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
# 8. MAIN RUNNER
# ============================================================================

def run_all():
    G     = load_mirpur_map()
    start = ox.distance.nearest_nodes(G, 90.3545, 23.7997)
    goal  = ox.distance.nearest_nodes(G, 90.3683, 23.8069)

    # print(f"\nStart node: {start}  |  Goal node: {goal}\n")
    print(f"\nStart node: Mirpur 1  |  Goal node: Mirpur 10 \n")

    algos = [
        ("BFS",      lambda: bfs(G, start, goal)),
        ("DFS",      lambda: dfs(G, start, goal)),
        ("DLS",      lambda: dls(G, start, goal, limit=40)),
        ("IDDFS",    lambda: iddfs(G, start, goal, max_depth=40)),
        ("UCS",      lambda: ucs(G, start, goal)),
        ("Greedy",   lambda: greedy_bfs(G, start, goal)),
        ("WeightedA*",lambda: weighted_a_star(G, start, goal, weight=1.5)),
        ("BiDi",     lambda: bidirectional_dijkstra(G, start, goal)),
        ("BiDiA*",   lambda: bidirectional_astar(G, start, goal)),
        ("IDA*",     lambda: ida_star(G, start, goal)),
    ]

    results    = []
    algo_paths = []

    for name, func in algos:
        print(f"Running {name} …")
        t0          = time.time()
        path, count = func()
        elapsed     = (time.time() - t0) * 1000

        cost = path_cost(G, path) if path else float("inf")
        results.append({"name": name, "time": elapsed, "nodes": count, "cost": cost})
        algo_paths.append((name, path))

        status = f"{len(path)} nodes in path, cost={cost:.1f}" if path else "NO PATH"
        print(f"  {name:<12} | {elapsed:7.1f} ms | {status}")

        # print(f"  {name:<12} | {elapsed:7.1f} ms | {count:5} explored | {status}")

        if path:
            plot_route(G, path, name)

    print("\nGenerating summary plots …")
    plot_complexities(results)
    plot_risk_heatmap(G)
    plot_all_routes_overlay(G, algo_paths)
    plot_risk_metrics_distribution(G)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Algorithm':<14} {'Time(ms)':>9}  {'PathCost':>12}  f(n) formula")

    # print(f"{'Algorithm':<14} {'Time(ms)':>9} {'Explored':>10} {'PathCost':>12}  f(n) formula")
    print("=" * 72)
    fn_notes = {
        "BFS":       "No f(n)  — FIFO hop-count",
        "DFS":       "No f(n)  — LIFO depth",
        "DLS":       "No f(n)  — depth ≤ limit",
        "IDDFS":     "No f(n)  — iterative depth limit",
        "UCS":       "f = g(n)",
        "Greedy":    "f = h(n)",
        "WeightedA*":"f = g(n) + 1.5·h(n)",
        "BiDi":      "f = g_fwd(n) or g_bwd(n)",
        "BiDiA*":    "f = g(n) + h(n) [bidirectional]",
        "IDA*":      "f = g(n) + h(n) [threshold]",
    }
    for r in results:
        n = r["name"]
        print(f"{n:<14} {r['time']:>9.1f} {r['nodes']:>10} {r['cost']:>12.1f}  {fn_notes.get(n,'')}")
    print("=" * 72)
    print(f"\nAll outputs saved → {OUTPUT_FOLDER}")


if __name__ == "__main__":
    run_all()