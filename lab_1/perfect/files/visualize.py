"""
Visualization Module
=====================
draw_map()        - draw the Mirpur city graph
draw_path()       - overlay a search result on the map
compare_results() - bar charts comparing all algorithms
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from mirpur_map import NODES, EDGES_DEF


# ─────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────
CLR = {
    "bg":       "#F8F7F4",
    "edge":     "#CCCCC0",
    "node":     "#4A90C4",
    "source":   "#22C55E",
    "dest":     "#EF4444",
    "visited":  "#F59E0B",
    "path":     "#7C3AED",
    "text":     "#1A1A1A",
    "subtext":  "#6B7280",
    "grid":     "#E5E7EB",
}

NODE_TYPE_SHAPES = {
    "transit":      ("D", 200),   # diamond
    "major":        ("*", 280),   # star
    "intersection": ("o", 160),
    "market":       ("s", 170),   # square
    "area":         ("o", 140),
    "residential":  ("^", 150),
    "industrial":   ("h", 160),   # hexagon
}


def _pos():
    """Return {node_id: (x, y)} positions."""
    return {nid: (info["x"], info["y"]) for nid, info in NODES.items()}


def _draw_base(ax, pos, title="Mirpur City Map"):
    ax.set_facecolor(CLR["bg"])
    ax.set_title(title, fontsize=13, fontweight="bold", color=CLR["text"], pad=10)

    # Draw edges
    for (a, b, cost, safety, traffic, gender, age, name) in EDGES_DEF:
        x = [pos[a][0], pos[b][0]]
        y = [pos[a][1], pos[b][1]]
        ax.plot(x, y, color=CLR["edge"], linewidth=1.2, zorder=1)

        # Edge cost label at midpoint
        mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        ax.text(mx, my, str(cost), fontsize=6.5, ha="center", va="center",
                color=CLR["subtext"], zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", fc=CLR["bg"], ec="none", alpha=0.7))

    # Draw nodes
    for nid, info in NODES.items():
        x, y = pos[nid]
        shape, size = NODE_TYPE_SHAPES.get(info["type"], ("o", 150))
        ax.scatter(x, y, s=size, c=CLR["node"], marker=shape, zorder=3,
                   edgecolors="white", linewidths=1.0)
        ax.text(x, y - 0.22, info["name"].replace("\n", " "),
                fontsize=6.5, ha="center", va="top",
                color=CLR["text"], zorder=4)
        ax.text(x + 0.05, y + 0.15, str(nid),
                fontsize=6, ha="left", va="bottom",
                color=CLR["subtext"], fontweight="bold", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")


def draw_path(result, weights, source, goal, save_path=None):
    """Draw map with visited nodes and final path highlighted."""
    pos = _pos()
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(CLR["bg"])

    _draw_base(ax, pos, title=f"{result.algorithm}  |  "
               f"Source: {NODES[source]['name']}  →  Dest: {NODES[goal]['name']}")

    # Highlight visited nodes (in order)
    for i, nid in enumerate(result.visited_order):
        if nid not in (source, goal):
            x, y = pos[nid]
            ax.scatter(x, y, s=220, c=CLR["visited"], marker="o", zorder=5,
                       edgecolors="white", linewidths=1.0, alpha=0.75)
            ax.text(x, y, str(i + 1), fontsize=6, ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)

    # Highlight path edges
    if result.path:
        for i in range(len(result.path) - 1):
            a, b = result.path[i], result.path[i + 1]
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=CLR["path"], linewidth=4, zorder=4, alpha=0.8,
                    solid_capstyle="round")

        # Path nodes
        for nid in result.path:
            if nid not in (source, goal):
                x, y = pos[nid]
                ax.scatter(x, y, s=200, c=CLR["path"], marker="o", zorder=7,
                           edgecolors="white", linewidths=1.5)

    # Source & Destination
    ax.scatter(*pos[source], s=300, c=CLR["source"], marker="o", zorder=8,
               edgecolors="white", linewidths=2)
    ax.text(pos[source][0], pos[source][1] + 0.25, "SOURCE",
            fontsize=8, ha="center", color=CLR["source"], fontweight="bold", zorder=9)

    ax.scatter(*pos[goal], s=300, c=CLR["dest"], marker="*", zorder=8,
               edgecolors="white", linewidths=1.5)
    ax.text(pos[goal][0], pos[goal][1] + 0.25, "DEST",
            fontsize=8, ha="center", color=CLR["dest"], fontweight="bold", zorder=9)

    # Stats box
    status = "PATH FOUND ✓" if result.found else "NO PATH ✗"
    stats = (f"{status}\n"
             f"Nodes visited : {result.nodes_visited}\n"
             f"Path length   : {result.path_length} edges\n"
             f"Total cost    : {result.total_cost:.2f}")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            color=CLR["text"],
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=CLR["grid"], alpha=0.9))

    # Legend
    handles = [
        mpatches.Patch(color=CLR["source"],  label="Source"),
        mpatches.Patch(color=CLR["dest"],    label="Destination"),
        mpatches.Patch(color=CLR["visited"], label="Visited nodes"),
        mpatches.Patch(color=CLR["path"],    label="Final path"),
        mpatches.Patch(color=CLR["node"],    label="Unvisited nodes"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor=CLR["grid"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


def compare_results(results: list, source: int, goal: int, save_path=None):
    """
    Bar chart comparison of all algorithms:
      - Nodes visited
      - Total path cost
      - Path length (edges)
    """
    algos = [r.algorithm for r in results]
    found = [r.found for r in results]
    visited = [r.nodes_visited for r in results]
    costs = [r.total_cost if r.found else 0 for r in results]
    lengths = [r.path_length if r.found else 0 for r in results]

    bar_colors = []
    for f in found:
        bar_colors.append("#4A90C4" if f else "#D1D5DB")

    fig = plt.figure(figsize=(14, 9), facecolor=CLR["bg"])
    fig.suptitle(
        f"Algorithm Comparison  |  "
        f"{NODES[source]['name']}  →  {NODES[goal]['name']}",
        fontsize=14, fontweight="bold", color=CLR["text"], y=0.98
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Subplot 1: Nodes visited (main metric)
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.barh(algos, visited, color=bar_colors, edgecolor="white",
                    linewidth=0.8, height=0.5)
    ax1.set_xlabel("Nodes Visited (Expanded)", fontsize=10, color=CLR["subtext"])
    ax1.set_title("Nodes Visited ← Less is better (efficiency)", fontsize=10,
                  fontweight="bold", color=CLR["text"])
    ax1.set_facecolor(CLR["bg"])
    ax1.grid(axis="x", color=CLR["grid"], linewidth=0.5)
    ax1.spines[:].set_visible(False)
    ax1.tick_params(colors=CLR["subtext"])
    for bar, val in zip(bars, visited):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=10, fontweight="bold",
                 color=CLR["text"])

    # Annotate uninformed vs informed
    max_v = max(visited) if visited else 1
    ax1.axvline(x=max_v * 0.5, color="#E0E0E0", linestyle="--", linewidth=0.8)

    # ── Subplot 2: Path cost
    ax2 = fig.add_subplot(gs[0, 2])
    colors2 = [CLR["path"] if f else "#D1D5DB" for f in found]
    ax2.bar(algos, costs, color=colors2, edgecolor="white", linewidth=0.8, width=0.5)
    ax2.set_title("Path Cost\n← Lower is better", fontsize=10,
                  fontweight="bold", color=CLR["text"])
    ax2.set_facecolor(CLR["bg"])
    ax2.grid(axis="y", color=CLR["grid"], linewidth=0.5)
    ax2.spines[:].set_visible(False)
    ax2.tick_params(axis="x", rotation=30, labelsize=8, colors=CLR["subtext"])
    ax2.tick_params(axis="y", colors=CLR["subtext"])
    for i, (c, f) in enumerate(zip(costs, found)):
        if f:
            ax2.text(i, c + 0.1, f"{c:.1f}", ha="center", fontsize=8,
                     fontweight="bold", color=CLR["text"])
        else:
            ax2.text(i, 0.5, "N/A", ha="center", fontsize=8, color=CLR["subtext"])

    # ── Subplot 3: Path length (edges)
    ax3 = fig.add_subplot(gs[1, 2])
    colors3 = [CLR["visited"] if f else "#D1D5DB" for f in found]
    ax3.bar(algos, lengths, color=colors3, edgecolor="white", linewidth=0.8, width=0.5)
    ax3.set_title("Path Length\n(number of edges)", fontsize=10,
                  fontweight="bold", color=CLR["text"])
    ax3.set_facecolor(CLR["bg"])
    ax3.grid(axis="y", color=CLR["grid"], linewidth=0.5)
    ax3.spines[:].set_visible(False)
    ax3.tick_params(axis="x", rotation=30, labelsize=8, colors=CLR["subtext"])
    ax3.tick_params(axis="y", colors=CLR["subtext"])
    for i, (l, f) in enumerate(zip(lengths, found)):
        if f:
            ax3.text(i, l + 0.05, str(l), ha="center", fontsize=8,
                     fontweight="bold", color=CLR["text"])

    # ── Subplot 4: Summary table
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.axis("off")
    cols = ["Algorithm", "Type", "Nodes\nVisited", "Path\nCost", "Path\nEdges", "Optimal?", "Found?"]
    table_data = []
    algo_types = {
        "BFS":    "Uninformed", "DFS": "Uninformed", "UCS": "Uninformed",
        "Greedy": "Informed",   "A*":  "Informed"
    }
    optimal_map = {
        "BFS":    "Hop-optimal", "DFS": "No",
        "UCS":    "Yes (cost)",  "Greedy": "No",
        "A*":     "Yes (cost)"
    }
    for r in results:
        table_data.append([
            r.algorithm,
            algo_types.get(r.algorithm, "-"),
            str(r.nodes_visited),
            f"{r.total_cost:.2f}" if r.found else "—",
            str(r.path_length) if r.found else "—",
            optimal_map.get(r.algorithm, "-"),
            "✓" if r.found else "✗",
        ])

    tbl = ax4.table(cellText=table_data, colLabels=cols,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Style header
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#334155")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight rows
    for i, r in enumerate(results):
        row = i + 1
        bg = "#F0FDF4" if r.algorithm in ("UCS", "A*") else (
             "#FFF7ED" if not r.found else "#FAFAFA")
        for j in range(len(cols)):
            tbl[row, j].set_facecolor(bg)

    ax4.set_title("Performance Summary Table", fontsize=10,
                  fontweight="bold", color=CLR["text"], pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


def draw_city_map(save_path=None):
    """Just draw the base Mirpur city map with no path."""
    pos = _pos()
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(CLR["bg"])
    _draw_base(ax, pos, title="Mirpur City — Node-Edge Map")

    # Type legend
    type_info = {
        "transit":      ("D", "Transit / Metro"),
        "major":        ("*", "Major Junction"),
        "intersection": ("o", "Intersection"),
        "market":       ("s", "Market"),
        "area":         ("o", "Residential Area"),
        "industrial":   ("h", "Industrial Area"),
    }
    handles = [
        plt.scatter([], [], marker=m, s=100, c=CLR["node"],
                    label=lbl, edgecolors="white", linewidths=0.8)
        for m, lbl in type_info.values()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              title="Node Types", title_fontsize=8,
              framealpha=0.9, edgecolor=CLR["grid"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
