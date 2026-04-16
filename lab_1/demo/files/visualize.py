"""
visualize.py
============
Saves all plots to ./output/ using the real OSM basemap via OSMnx.

Images produced:
  city_map.png                — real OSM basemap + node/edge overlay
  path_<Algo>.png             — per-algorithm: visited nodes + final path
  comparison_nodes.png        — bar chart: nodes visited
  comparison_cost.png         — bar chart: path cost
  comparison_combined.png     — full dashboard
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import osmnx as ox
import networkx as nx

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

C = {
    "source":    "#1DB954",
    "dest":      "#E84040",
    "visited":   "#F5A623",
    "path":      "#7B2FBE",
    "text":      "#1A1A1A",
    "subtext":   "#555550",
    "grid":      "#DDDDCC",
    "uninformed":"#3A7DC9",
    "informed":  "#7B2FBE",
    "panel":     "#FAFAF7",
}

ALGO_TYPE = {
    "BFS":"Uninformed","DFS":"Uninformed","DLS":"Uninformed",
    "IDDFS":"Uninformed","UCS":"Uninformed","BiDS":"Uninformed",
    "Greedy":"Informed","A*":"Informed","Weighted A*":"Informed",
    "BiA*":"Informed","IDA*":"Informed","Beam":"Informed",
}
OPTIMAL = {
    "BFS":"Hop-optimal","DFS":"No","DLS":"No","IDDFS":"Hop-optimal",
    "UCS":"Yes (cost)","BiDS":"Hop-optimal","Greedy":"No",
    "A*":"Yes (cost)","Weighted A*":"Within ε","BiA*":"Approx.",
    "IDA*":"Yes (cost)","Beam":"No",
}


def _save(fig, fname):
    p = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {p}")
    return p


def _node_xy(graph, nid):
    n = graph.nodes[nid]
    return n.x, n.y


def _osm_basemap(graph, ax, figsize_hint=None):
    """
    Plot the real OSM road network as background using OSMnx.
    Uses ox.plot_graph() internals to draw edges on the given ax.
    """
    G = graph.osm_graph()

    # Build position dict: osmid → (x_local, y_local)
    pos = {nid: (n.x, n.y) for nid, n in graph.nodes.items()}

    # Draw edges by road type
    highway_colors = {
        "primary":      "#D0683A",
        "secondary":    "#FFC0CB",
        "tertiary":     "#808878",
        "residential":  "#408EC6",
        "unclassified": "#C8C8B8",
        "default":      "#D0D0C0",
    }
    highway_widths = {
        "primary": 1.8, "secondary": 1.4, "tertiary": 1.1,
        "residential": 0.7, "unclassified": 0.6, "default": 0.5,
    }

    drawn = set()
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos: continue
        pair = (min(u,v), max(u,v))
        if pair in drawn: continue
        drawn.add(pair)
        hw = data.get("highway","default")
        if isinstance(hw, list): hw = hw[0]
        col = highway_colors.get(hw, highway_colors["default"])
        lw  = highway_widths.get(hw, highway_widths["default"])
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=col, linewidth=lw, zorder=1, alpha=0.85,
                solid_capstyle="round")

    ax.set_aspect("equal")
    ax.set_facecolor("#F0EFE8")
    ax.grid(True, color=C["grid"], linewidth=0.3, linestyle="--", alpha=0.5, zorder=0)
    ax.tick_params(colors=C["subtext"], labelsize=7)
    ax.set_xlabel("← West       km East →", fontsize=8, color=C["subtext"])
    ax.set_ylabel("← South      km North →", fontsize=8, color=C["subtext"])
    ax.text(0.99, 0.01, "N↑", transform=ax.transAxes,
            fontsize=10, ha="right", va="bottom",
            color=C["subtext"], fontweight="bold")


# ─────────────────────────────────────────────────────────────────────
# 1. CITY MAP
# ─────────────────────────────────────────────────────────────────────
def save_city_map(graph, source, goal):
    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor(C["panel"])
    ax.set_title("Mirpur Road Network — OpenStreetMap (OSMnx)\n"
                 f"Source: Mirpur-2 Stadium  →  Dest: Pallabi Bus Stand",
                 fontsize=12, fontweight="bold", color=C["text"], pad=10)

    _osm_basemap(graph, ax)

    # Source and destination
    sx, sy = _node_xy(graph, source)
    gx, gy = _node_xy(graph, goal)
    ax.scatter(sx, sy, s=300, c=C["source"], marker="o",
               zorder=9, edgecolors="white", linewidths=2)
    ax.text(sx, sy+0.04, "SOURCE\n(Mirpur-2 Stadium)",
            fontsize=8, ha="center", va="bottom",
            color=C["source"], fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["source"], alpha=0.85))

    ax.scatter(gx, gy, s=300, c=C["dest"], marker="*",
               zorder=9, edgecolors="white", linewidths=1.5)
    ax.text(gx, gy+0.04, "DEST\n(Pallabi Bus Stand)",
            fontsize=8, ha="center", va="bottom",
            color=C["dest"], fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["dest"], alpha=0.85))

    # Legend
    handles = [
        mpatches.Patch(color=C["source"], label=f"Source (node {source})"),
        mpatches.Patch(color=C["dest"],   label=f"Goal (node {goal})"),
        mpatches.Patch(color="#D0683A",   label="Primary road"),
        mpatches.Patch(color="#2F3C7E",   label="Secondary road"),
        mpatches.Patch(color="#408EC6",   label="Residential road"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.92, edgecolor=C["grid"])

    ax.text(0.01, 0.01,
            "© OpenStreetMap contributors  |  OSMnx road network",
            transform=ax.transAxes, fontsize=7, color=C["subtext"],
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grid"], alpha=0.8))

    plt.tight_layout()
    return _save(fig, "city_map.png")


# ─────────────────────────────────────────────────────────────────────
# 2. PER-ALGORITHM PATH MAP
# ─────────────────────────────────────────────────────────────────────
def save_path_map(result, graph, source, goal, profile_name):
    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor(C["panel"])
    ax.set_title(
        f"{result.algorithm}  [{ALGO_TYPE.get(result.algorithm,'')}]  |  "
        f"Profile: {profile_name}\n"
        f"Mirpur-2 Stadium  →  Pallabi Bus Stand  |  "
        f"Visited: {result.nodes_visited}  Cost: {result.total_cost:.3f}",
        fontsize=11, fontweight="bold", color=C["text"], pad=8)

    _osm_basemap(graph, ax)

    pos = {nid: (n.x, n.y) for nid, n in graph.nodes.items()}

    # Visited nodes (numbered in expansion order)
    for i, nid in enumerate(result.visited_order):
        if nid not in pos: continue
        if nid in (source, goal): continue
        x, y = pos[nid]
        ax.scatter(x, y, s=120, c=C["visited"], marker="o", zorder=6,
                   edgecolors="white", linewidths=0.8, alpha=0.85)
        if i < 60:   # only label first 60 to avoid clutter on large graphs
            ax.text(x, y, str(i+1), fontsize=4.5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=7)

    # Final path
    if result.path:
        px = [pos[n][0] for n in result.path if n in pos]
        py = [pos[n][1] for n in result.path if n in pos]
        ax.plot(px, py, color=C["path"], linewidth=4, zorder=5,
                alpha=0.85, solid_capstyle="round", solid_joinstyle="round")
        for nid in result.path:
            if nid not in pos: continue
            if nid in (source, goal): continue
            ax.scatter(*pos[nid], s=100, c=C["path"], marker="o",
                       zorder=8, edgecolors="white", linewidths=1.0)

    # Source / goal markers
    if source in pos:
        ax.scatter(*pos[source], s=300, c=C["source"], marker="o",
                   zorder=9, edgecolors="white", linewidths=2)
        ax.text(pos[source][0], pos[source][1]+0.04, "SRC",
                fontsize=8, ha="center", va="bottom",
                color=C["source"], fontweight="bold", zorder=10)
    if goal in pos:
        ax.scatter(*pos[goal], s=300, c=C["dest"], marker="*",
                   zorder=9, edgecolors="white", linewidths=1.5)
        ax.text(pos[goal][0], pos[goal][1]+0.04, "DST",
                fontsize=8, ha="center", va="bottom",
                color=C["dest"], fontweight="bold", zorder=10)

    # Stats box
    status = "✓ FOUND" if result.found else "✗ NOT FOUND"
    extra = "\n".join(f"{k}: {v}" for k, v in result.extra.items())
    stats = (f"{status}\n"
             f"Nodes visited : {result.nodes_visited}\n"
             f"Path edges    : {result.path_length}\n"
             f"Total cost    : {result.total_cost:.4f}"
             + (f"\n{extra}" if extra else ""))
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color=C["text"],
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C["grid"], alpha=0.92))

    handles = [
        mpatches.Patch(color=C["source"],  label="Source"),
        mpatches.Patch(color=C["dest"],    label="Destination"),
        mpatches.Patch(color=C["visited"], label=f"Visited ({result.nodes_visited} nodes, numbered)"),
        mpatches.Patch(color=C["path"],    label="Final path"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.92, edgecolor=C["grid"])

    plt.tight_layout()
    safe = result.algorithm.replace("*","star").replace(" ","_")
    return _save(fig, f"path_{safe}.png")


# ─────────────────────────────────────────────────────────────────────
# 3. COMPARISON CHARTS
# ─────────────────────────────────────────────────────────────────────
def _bcolors(results):
    return [C["uninformed"] if ALGO_TYPE.get(r.algorithm)=="Uninformed"
            else C["informed"] for r in results]


def save_comparison_nodes(results, graph, source, goal):
    algos   = [r.algorithm for r in results]
    visited = [r.nodes_visited for r in results]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["panel"])
    bars = ax.barh(algos, visited, color=_bcolors(results),
                   edgecolor="white", linewidth=0.8, height=0.58)
    ax.set_xlabel("Nodes Expanded", fontsize=11, color=C["subtext"])
    ax.set_title(
        "Nodes Visited per Algorithm\n"
        "Blue = Uninformed  |  Purple = Informed  |  Fewer = more efficient",
        fontsize=12, fontweight="bold", color=C["text"])
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.5)
    ax.tick_params(colors=C["subtext"], labelsize=9)
    for bar, val in zip(bars, visited):
        ax.text(bar.get_width()+max(visited)*0.005,
                bar.get_y()+bar.get_height()/2,
                str(val), va="center", fontsize=9,
                fontweight="bold", color=C["text"])
    plt.tight_layout()
    return _save(fig, "comparison_nodes.png")


def save_comparison_cost(results, graph, source, goal):
    found  = [r for r in results if r.found]
    algos  = [r.algorithm for r in found]
    costs  = [r.total_cost for r in found]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["panel"])
    ax.bar(algos, costs, color=_bcolors(found),
           edgecolor="white", linewidth=0.8, width=0.58)
    ax.set_ylabel("Total Path Cost", fontsize=11, color=C["subtext"])
    ax.set_title(
        "Path Cost per Algorithm\n"
        "Blue = Uninformed  |  Purple = Informed  |  Lower = cheaper",
        fontsize=12, fontweight="bold", color=C["text"])
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", color=C["grid"], linewidth=0.5)
    ax.tick_params(axis="x", rotation=35, labelsize=8, colors=C["subtext"])
    ax.tick_params(axis="y", colors=C["subtext"])
    for i, c in enumerate(costs):
        ax.text(i, c+max(costs)*0.005, f"{c:.2f}", ha="center",
                fontsize=8, fontweight="bold", color=C["text"])
    plt.tight_layout()
    return _save(fig, "comparison_cost.png")


def save_comparison_combined(results, graph, source, goal, profile_name):
    algos   = [r.algorithm for r in results]
    found   = [r.found for r in results]
    visited = [r.nodes_visited for r in results]
    costs   = [r.total_cost if r.found else 0 for r in results]
    lengths = [r.path_length if r.found else 0 for r in results]
    colors  = _bcolors(results)

    fig = plt.figure(figsize=(20, 14), facecolor=C["panel"])
    fig.suptitle(
        f"Full Algorithm Comparison  |  Mirpur-2 Stadium → Pallabi Bus Stand"
        f"  |  Profile: {profile_name}\n"
        f"OSMnx real road network  |  {len(graph.nodes)} nodes  {len(graph.edges)} edges",
        fontsize=13, fontweight="bold", color=C["text"], y=0.99)

    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.32,
                  height_ratios=[1, 1, 1.5])

    # Nodes visited
    ax1 = fig.add_subplot(gs[0, :])
    bars1 = ax1.barh(algos, visited, color=colors,
                     edgecolor="white", linewidth=0.7, height=0.6)
    ax1.set_xlabel("Nodes Expanded", fontsize=10, color=C["subtext"])
    ax1.set_title("Nodes Visited  (fewer = more efficient)",
                  fontsize=11, fontweight="bold", color=C["text"])
    ax1.set_facecolor(C["panel"]); ax1.spines[:].set_visible(False)
    ax1.grid(axis="x", color=C["grid"], linewidth=0.4)
    ax1.tick_params(colors=C["subtext"], labelsize=8)
    for bar, val in zip(bars1, visited):
        ax1.text(bar.get_width()+max(visited)*0.004,
                 bar.get_y()+bar.get_height()/2,
                 str(val), va="center", fontsize=8,
                 fontweight="bold", color=C["text"])

    # Path cost
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(algos, costs, color=colors, edgecolor="white",
            linewidth=0.7, width=0.6)
    ax2.set_title("Path Cost  (lower = cheaper)",
                  fontsize=10, fontweight="bold", color=C["text"])
    ax2.set_facecolor(C["panel"]); ax2.spines[:].set_visible(False)
    ax2.grid(axis="y", color=C["grid"], linewidth=0.4)
    ax2.tick_params(axis="x", rotation=40, labelsize=7, colors=C["subtext"])
    ax2.tick_params(axis="y", colors=C["subtext"])
    for i,(c,f) in enumerate(zip(costs,found)):
        ax2.text(i, c+max(costs)*0.01 if max(costs)>0 else 0.1,
                 f"{c:.2f}" if f else "N/F",
                 ha="center", fontsize=7, fontweight="bold", color=C["text"])

    # Path length
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(algos, lengths, color=colors, edgecolor="white",
            linewidth=0.7, width=0.6)
    ax3.set_title("Path Length  (edges)",
                  fontsize=10, fontweight="bold", color=C["text"])
    ax3.set_facecolor(C["panel"]); ax3.spines[:].set_visible(False)
    ax3.grid(axis="y", color=C["grid"], linewidth=0.4)
    ax3.tick_params(axis="x", rotation=40, labelsize=7, colors=C["subtext"])
    ax3.tick_params(axis="y", colors=C["subtext"])
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for i,(l,f) in enumerate(zip(lengths,found)):
        if f: ax3.text(i, l+0.05, str(l), ha="center", fontsize=7,
                       fontweight="bold", color=C["text"])

    # Summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    cols = ["Algorithm","Type","Nodes\nVisited","Path\nCost",
            "Path\nEdges","Optimal?","Found?","Extra Info"]
    rows = []
    for r in results:
        ex = ", ".join(f"{k}={v}" for k,v in r.extra.items()) if r.extra else "—"
        rows.append([
            r.algorithm,
            ALGO_TYPE.get(r.algorithm,"-"),
            str(r.nodes_visited),
            f"{r.total_cost:.3f}" if r.found else "—",
            str(r.path_length)    if r.found else "—",
            OPTIMAL.get(r.algorithm,"-"),
            "✓" if r.found else "✗",
            ex,
        ])
    tbl = ax4.table(cellText=rows, colLabels=cols,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.55)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#2D3748")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i,r in enumerate(results, 1):
        if r.algorithm in ("UCS","A*","IDA*"): bg="#EBF8F0"
        elif ALGO_TYPE.get(r.algorithm)=="Informed": bg="#F3EEF9"
        else: bg="#F8F8F8"
        for j in range(len(cols)):
            tbl[i,j].set_facecolor(bg)

    ax4.set_title(
        "Performance Summary  |  Green=Optimal  Purple=Other Informed  White=Uninformed",
        fontsize=9, fontweight="bold", color=C["text"], pad=10)

    handles=[
        mpatches.Patch(color=C["uninformed"],label="Uninformed search"),
        mpatches.Patch(color=C["informed"],  label="Informed search"),
    ]
    fig.legend(handles=handles, loc="upper right",
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.99,0.975))

    plt.tight_layout(rect=[0,0,1,0.97])
    return _save(fig, "comparison_combined.png")


# ─────────────────────────────────────────────────────────────────────
# MASTER
# ─────────────────────────────────────────────────────────────────────
def generate_all(results, graph, source, goal, profile_name):
    print(f"\n  Saving all plots to: {OUTPUT_DIR}/")
    SKIP_PLOTS = {"DLS", "IDDFS", "Beam", "IDA*"}    
    saved = []
    saved.append(save_city_map(graph, source, goal))
    for r in results:
        if r.algorithm in SKIP_PLOTS:
            continue
        saved.append(save_path_map(r, graph, source, goal, profile_name))
    saved.append(save_comparison_nodes(results, graph, source, goal))
    saved.append(save_comparison_cost(results, graph, source, goal))
    saved.append(save_comparison_combined(results, graph, source, goal, profile_name))
    print(f"\n  Total images saved: {len(saved)}")
    return saved
