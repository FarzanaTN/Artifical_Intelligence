

import os


BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "assignment_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


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
    # Special map nodes
    "node_regular": "#2a3440",
    "node_src":     "#3fb950",   # start  — bright green
    "node_dst":     "#f85149",   # goal   — bright red
    "node_meet":    "#ffa657",   # bidirectional meeting point — orange
    # Per-algorithm route colours
    "route_bfs":     "#58a6ff",
    "route_dfs":     "#3fb950",
    "route_ucs":     "#ffa657",
    "route_greedy":  "#f85149",
    "route_astar":   "#00d2ff",
    "route_wastar":  "#bc8cff",
    "route_dls":     "#79c0ff",
    "route_iddfs":   "#56d364",
    "route_bds":     "#ffb547",
    "route_bdastar": "#ff7b72",
    "route_idastar": "#d2a8ff",
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

FN_NOTES = {
    "BFS":        "No f(n) — FIFO hop-count",
    "DFS":        "No f(n) — LIFO depth",
    "DLS":        "No f(n) — depth <= limit",
    "IDDFS":      "No f(n) — iterative depth limit",
    "UCS":        "f = g(n)",
    "Greedy":     "f = h(n)",
    "A*":         "f = g(n) + h(n)  [optimal]",
    "WeightedA*": "f = g(n) + 1.5·h(n)  [near-optimal]",
    "BiDi":       "f = g_fwd or g_bwd  [meet-in-middle]",
    "BiDiA*":     "f = g(n) + h(n)  [bidirectional]",
    "IDA*":       "f = g(n) + h(n)  [threshold deepening]",
}
