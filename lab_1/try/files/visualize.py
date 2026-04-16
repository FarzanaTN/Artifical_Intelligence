"""
Visualization Module — GPS-accurate Mirpur map
Run this via main.py. Plots are shown live using plt.show().
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mirpur_map import NODES, EDGES_DEF

CLR = {
    "bg":      "#F8F7F4",
    "edge":    "#C8C8BC",
    "node":    "#4A90C4",
    "source":  "#22C55E",
    "dest":    "#EF4444",
    "visited": "#F59E0B",
    "path":    "#7C3AED",
    "text":    "#1A1A1A",
    "subtext": "#6B7280",
    "grid":    "#E5E7EB",
    "mrt":     "#E05C2A",
}

NODE_TYPE_STYLE = {
    "transit":      ("D", 200, "#E05C2A"),
    "major":        ("*", 280, "#1D6FA4"),
    "intersection": ("o", 160, "#4A90C4"),
    "market":       ("s", 170, "#8B5CF6"),
    "area":         ("o", 140, "#4A90C4"),
}

MRT_NODES = {5, 6, 7, 9, 10, 11}


def _pos():
    return {nid: (info["x"], info["y"]) for nid, info in NODES.items()}


def _draw_base(ax, pos, title="Mirpur City Map"):
    ax.set_facecolor(CLR["bg"])
    ax.set_title(title, fontsize=12, fontweight="bold", color=CLR["text"], pad=10)

    # MRT Line 6 corridor highlight
    mrt_spine = [7, 6, 5, 9, 10, 11]
    for i in range(len(mrt_spine) - 1):
        a, b = mrt_spine[i], mrt_spine[i + 1]
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=CLR["mrt"], linewidth=5, zorder=1, alpha=0.2,
                solid_capstyle="round")

    # All road edges
    for (a, b, cost, *_rest, name) in EDGES_DEF:
        x = [pos[a][0], pos[b][0]]
        y = [pos[a][1], pos[b][1]]
        ax.plot(x, y, color=CLR["edge"], linewidth=1.0, zorder=2)
        mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        ax.text(mx, my, str(cost), fontsize=6, ha="center", va="center",
                color=CLR["subtext"], zorder=3,
                bbox=dict(boxstyle="round,pad=0.1", fc=CLR["bg"], ec="none", alpha=0.75))

    # Nodes
    for nid, info in NODES.items():
        x, y = pos[nid]
        shape, size, color = NODE_TYPE_STYLE.get(info["type"], ("o", 140, CLR["node"]))
        ec = CLR["mrt"] if nid in MRT_NODES else "white"
        lw = 2.0 if nid in MRT_NODES else 1.0
        ax.scatter(x, y, s=size, c=color, marker=shape, zorder=4,
                   edgecolors=ec, linewidths=lw)
        ax.text(x, y - 0.18, info["name"], fontsize=6.5, ha="center", va="top",
                color=CLR["text"], zorder=5)
        ax.text(x + 0.04, y + 0.12, str(nid), fontsize=5.5, ha="left", va="bottom",
                color=CLR["subtext"], fontweight="bold", zorder=5)

    ax.text(0.98, 0.02, "N↑", transform=ax.transAxes,
            fontsize=10, ha="right", va="bottom", color=CLR["subtext"], fontweight="bold")
    ax.plot([], [], color=CLR["mrt"], linewidth=4, alpha=0.4, label="MRT Line 6 corridor")
    ax.set_xlabel("← West      Distance (km)      East →", fontsize=8, color=CLR["subtext"])
    ax.set_ylabel("← South    Distance (km)    North →", fontsize=8, color=CLR["subtext"])
    ax.set_aspect("equal")
    ax.grid(True, color=CLR["grid"], linewidth=0.4, linestyle="--", alpha=0.6)
    ax.tick_params(colors=CLR["subtext"], labelsize=7)


def draw_city_map():
    """Show base Mirpur map (no path)."""
    pos = _pos()
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor(CLR["bg"])
    _draw_base(ax, pos, title="Mirpur City — GPS-accurate Node-Edge Map")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, edgecolor=CLR["grid"])
    ax.text(0.02, 0.02,
            "x = km east from 90.350°E,  y = km north from 23.770°N",
            transform=ax.transAxes, fontsize=7, color=CLR["subtext"], va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CLR["grid"], alpha=0.8))
    plt.tight_layout()
    plt.show()


def draw_path(result, source, goal):
    """Show map with visited nodes and final path for one algorithm."""
    pos = _pos()
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(CLR["bg"])
    _draw_base(ax, pos,
               title=f"{result.algorithm}  |  "
                     f"{NODES[source]['name']}  →  {NODES[goal]['name']}")

    # Visited nodes (numbered in expansion order)
    for i, nid in enumerate(result.visited_order):
        if nid not in (source, goal):
            x, y = pos[nid]
            ax.scatter(x, y, s=240, c=CLR["visited"], marker="o", zorder=6,
                       edgecolors="white", linewidths=1.0, alpha=0.8)
            ax.text(x, y, str(i + 1), fontsize=6.5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=7)

    # Final path edges
    if result.path:
        for i in range(len(result.path) - 1):
            a, b = result.path[i], result.path[i + 1]
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=CLR["path"], linewidth=4.5, zorder=5, alpha=0.85,
                    solid_capstyle="round")
        for nid in result.path:
            if nid not in (source, goal):
                ax.scatter(*pos[nid], s=200, c=CLR["path"], marker="o",
                           zorder=8, edgecolors="white", linewidths=1.5)

    # Source & destination markers
    ax.scatter(*pos[source], s=320, c=CLR["source"], marker="o",
               zorder=9, edgecolors="white", linewidths=2)
    ax.text(pos[source][0], pos[source][1] + 0.22, "SOURCE",
            fontsize=8, ha="center", color=CLR["source"], fontweight="bold", zorder=10)

    ax.scatter(*pos[goal], s=320, c=CLR["dest"], marker="*",
               zorder=9, edgecolors="white", linewidths=1.5)
    ax.text(pos[goal][0], pos[goal][1] + 0.22, "DEST",
            fontsize=8, ha="center", color=CLR["dest"], fontweight="bold", zorder=10)

    # Stats box
    status = "PATH FOUND ✓" if result.found else "NO PATH ✗"
    stats = (f"{status}\n"
             f"Nodes visited : {result.nodes_visited}\n"
             f"Path length   : {result.path_length} edges\n"
             f"Total cost    : {result.total_cost:.2f}")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left", color=CLR["text"],
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=CLR["grid"], alpha=0.9))

    handles = [
        mpatches.Patch(color=CLR["source"],  label="Source"),
        mpatches.Patch(color=CLR["dest"],    label="Destination"),
        mpatches.Patch(color=CLR["visited"], label="Visited nodes (numbered)"),
        mpatches.Patch(color=CLR["path"],    label="Final path"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor=CLR["grid"])
    plt.tight_layout()
    plt.show()


def compare_results(results, source, goal):
    """Show bar chart + table comparing all algorithm results."""
    algos   = [r.algorithm for r in results]
    found   = [r.found for r in results]
    visited = [r.nodes_visited for r in results]
    costs   = [r.total_cost if r.found else 0 for r in results]
    lengths = [r.path_length if r.found else 0 for r in results]

    fig = plt.figure(figsize=(14, 9), facecolor=CLR["bg"])
    fig.suptitle(
        f"Algorithm Comparison  |  "
        f"{NODES[source]['name']}  →  {NODES[goal]['name']}",
        fontsize=13, fontweight="bold", color=CLR["text"], y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Nodes visited
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.barh(algos, visited,
                    color=["#4A90C4" if f else "#D1D5DB" for f in found],
                    edgecolor="white", height=0.5)
    ax1.set_xlabel("Nodes Visited (Expanded)", fontsize=10, color=CLR["subtext"])
    ax1.set_title("Nodes Visited  ← Fewer = more efficient", fontsize=10,
                  fontweight="bold", color=CLR["text"])
    ax1.set_facecolor(CLR["bg"])
    ax1.grid(axis="x", color=CLR["grid"], linewidth=0.5)
    ax1.spines[:].set_visible(False)
    ax1.tick_params(colors=CLR["subtext"])
    for bar, val in zip(bars, visited):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=10, fontweight="bold", color=CLR["text"])

    # Path cost
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(algos, costs, color=[CLR["path"] if f else "#D1D5DB" for f in found],
            edgecolor="white", width=0.5)
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

    # Path length
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.bar(algos, lengths, color=[CLR["visited"] if f else "#D1D5DB" for f in found],
            edgecolor="white", width=0.5)
    ax3.set_title("Path Length\n(edges)", fontsize=10,
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

    # Summary table
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.axis("off")
    algo_types  = {"BFS": "Uninformed", "DFS": "Uninformed", "UCS": "Uninformed",
                   "Greedy": "Informed",  "A*":  "Informed"}
    optimal_map = {"BFS": "hop-optimal", "DFS": "No", "UCS": "Yes (cost)",
                   "Greedy": "No",        "A*":  "Yes (cost)"}
    cols = ["Algorithm", "Type", "Nodes Visited", "Path Cost", "Path Edges", "Optimal?", "Found?"]
    table_data = [
        [r.algorithm,
         algo_types.get(r.algorithm, "-"),
         str(r.nodes_visited),
         f"{r.total_cost:.2f}" if r.found else "—",
         str(r.path_length)   if r.found else "—",
         optimal_map.get(r.algorithm, "-"),
         "✓" if r.found else "✗"]
        for r in results
    ]
    tbl = ax4.table(cellText=table_data, colLabels=cols,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#334155")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, r in enumerate(results):
        bg = "#F0FDF4" if r.algorithm in ("UCS", "A*") else "#FAFAFA"
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(bg)
    ax4.set_title("Performance Summary", fontsize=10,
                  fontweight="bold", color=CLR["text"], pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
