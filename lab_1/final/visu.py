
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from config import OUTPUT_FOLDER, PALETTE, ALGO_COLORS
from graph import path_cost



STREET_COLOR  = "#4a6741"   # muted green-grey — clearly visible on dark bg
STREET_CASING = "#0d1117"   # matches bg — separates parallel roads
STREET_WIDTH  = 1.4
NODE_COLOR    = "#5a7a6e"



def _draw_base_graph(G, ax, dim_factor: float = 1.0) -> None:
   
    segs = [
        [[G.nodes[u]["x"], G.nodes[u]["y"]],
         [G.nodes[v]["x"], G.nodes[v]["y"]]]
        for u, v in G.edges()
    ]

    # Pass 1 — dark casing
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

    # LineCollection does not auto-scale axes
    margin_x = (max(xs) - min(xs)) * 0.03
    margin_y = (max(ys) - min(ys)) * 0.03
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)


def _draw_route_overlay(G, ax, path: list, color: str) -> None:
  
    segs = [
        [[G.nodes[path[i]]["x"],     G.nodes[path[i]]["y"]],
         [G.nodes[path[i+1]]["x"],   G.nodes[path[i+1]]["y"]]]
        for i in range(len(path) - 1)
    ]

    ax.add_collection(LineCollection(segs, linewidths=9,   colors=color,   alpha=0.18, zorder=10))
    ax.add_collection(LineCollection(segs, linewidths=3.8,  colors=color,   alpha=0.95, zorder=11))
    ax.add_collection(LineCollection(segs, linewidths=1.0,  colors="white", alpha=0.55, zorder=12))

    xs = [G.nodes[n]["x"] for n in path]
    ys = [G.nodes[n]["y"] for n in path]
    ax.scatter(xs, ys, s=16, c=color, alpha=0.55, zorder=13, linewidths=0)


def _draw_special_nodes(G, ax, start, goal, meet=None) -> None:
   
    for node, color in [(start, PALETTE["node_src"]),
                        (goal,  PALETTE["node_dst"])]:
        ax.scatter([G.nodes[node]["x"]], [G.nodes[node]["y"]],
                   s=756, c=color, alpha=0.18, zorder=18, linewidths=0)

    ax.scatter([G.nodes[start]["x"]], [G.nodes[start]["y"]],
               s=280, c=PALETTE["node_src"], marker="^",
               edgecolors="white", linewidths=1.8, zorder=20)
    ax.scatter([G.nodes[goal]["x"]], [G.nodes[goal]["y"]],
               s=340, c=PALETTE["node_dst"], marker="*",
               edgecolors="white", linewidths=1.8, zorder=20)

    if meet is not None and meet not in (start, goal):
        ax.scatter([G.nodes[meet]["x"]], [G.nodes[meet]["y"]],
                   s=612, c=PALETTE["node_meet"], alpha=0.20,
                   zorder=18, linewidths=0)
        ax.scatter([G.nodes[meet]["x"]], [G.nodes[meet]["y"]],
                   s=200, c=PALETTE["node_meet"], marker="D",
                   edgecolors="white", linewidths=1.8, zorder=20)


def _style_ax(ax, title: str) -> None:
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


def _bar_chart(ax, names, values, colors, title, ylabel) -> None:
    """Shared styled bar-chart helper (log scale)."""
    ax.set_facecolor(PALETTE["panel"])
    bars = ax.bar(names, values, color=colors,
                  edgecolor=PALETTE["border"], linewidth=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_title(title, color=PALETTE["text"], fontsize=14,
                 fontweight="bold", fontfamily="monospace", pad=10)
    ax.set_xlabel("Algorithm", color=PALETTE["subtext"], fontsize=11)
    ax.set_ylabel(ylabel, color=PALETTE["subtext"], fontsize=10)
    ax.tick_params(axis="x", colors=PALETTE["subtext"], rotation=35)
    ax.tick_params(axis="y", colors=PALETTE["subtext"])
    ax.yaxis.grid(True, color=PALETTE["border"], linestyle="--",
                  alpha=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                f"{val:.1f}" if val < 1000 else f"{int(val):,}",
                ha="center", va="bottom",
                color=PALETTE["text"], fontsize=8, fontfamily="monospace")


def _save(fig, filename: str) -> None:
    out = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_route(G, path: list, name: str, start, goal, meet=None,
               color: str = None) -> None:
   
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

    _style_ax(ax, f"{name}   |   {len(path)} hops   |   cost = {path_cost(G, path):.0f}")
    fig.tight_layout(pad=1.5)
    _save(fig, f"{name}_route.png")


def plot_risk_heatmap(G, start, goal) -> None:
   
    edge_list = list(G.edges(data=True))
    costs     = np.array([d.get("cost", 1) for _, _, d in edge_list])
    norm      = Normalize(vmin=costs.min(), vmax=costs.max())
    cmap      = LinearSegmentedColormap.from_list(
        "risk", ["#3fb950", "#d29922", "#f85149"])

    segs          = [[[G.nodes[u]["x"], G.nodes[u]["y"]],
                      [G.nodes[v]["x"], G.nodes[v]["y"]]]
                     for (u, v, _) in edge_list]
    colors_mapped = [cmap(norm(c)) for c in costs]

    fig, ax = plt.subplots(figsize=(14, 12), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Casing pass
    ax.add_collection(LineCollection(
        segs, linewidths=STREET_WIDTH * 2.8,
        colors=STREET_CASING, alpha=0.95, zorder=1))
    # Risk-coloured fill
    ax.add_collection(LineCollection(
        segs, linewidths=STREET_WIDTH * 1.2,
        colors=colors_mapped, alpha=0.92, zorder=2))

    xs = [G.nodes[n]["x"] for n in G.nodes]
    ys = [G.nodes[n]["y"] for n in G.nodes]
    ax.scatter(xs, ys, s=6, c="#4a6a86", alpha=0.6, zorder=3, linewidths=0)
    margin_x = (max(xs) - min(xs)) * 0.03
    margin_y = (max(ys) - min(ys)) * 0.03
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    _draw_special_nodes(G, ax, start, goal)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
    cbar.ax.set_ylabel("Composite Risk Cost",
                       color=PALETTE["subtext"], fontsize=9)
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
    _save(fig, "risk_heatmap.png")


def plot_all_routes_overlay(G, algo_paths: list, start, goal) -> None:
    """All algorithm routes overlaid on one dimmed street map."""
    fig, ax = plt.subplots(figsize=(16, 14), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    _draw_base_graph(G, ax, dim_factor=0.55)

    legend_elems = []
    for name, path in algo_paths:
        if path is None:
            continue
        color = ALGO_COLORS.get(name, PALETTE["accent"])
        segs  = [[[G.nodes[path[i]]["x"],   G.nodes[path[i]]["y"]],
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
    _save(fig, "all_routes_overlay.png")


def plot_complexities(results: list) -> None:
    """Side-by-side log-scale bar charts for time and space complexity."""
    names  = [r["name"]  for r in results]
    colors = [ALGO_COLORS.get(n, PALETTE["accent"]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(22, 7), facecolor=PALETTE["bg"])
    _bar_chart(axes[0], names, [r["time"]  for r in results],
               colors, "Time Complexity",  "ms (log scale)")
    _bar_chart(axes[1], names, [r["nodes"] for r in results],
               colors, "Space Complexity", "Nodes Explored (log scale)")

    fig.tight_layout(pad=2.5)
    _save(fig, "complexity_comparison.png")


def plot_cost_comparison(results: list) -> None:
   
    names  = [r["name"] for r in results]
    costs  = [r["cost"] for r in results]
    colors = [ALGO_COLORS.get(n, PALETTE["accent"]) for n in names]

    finite      = [c for c in costs if c != float("inf")]
    optimal     = min(finite) if finite else 0
    display_max = max(finite) * 1.12 if finite else 1
    disp_costs  = [c if c != float("inf") else display_max for c in costs]

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    bars = ax.bar(names, disp_costs, color=colors,
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
    _save(fig, "cost_comparison.png")


def plot_risk_metrics_distribution(G) -> None:
    """Box-plots of the four risk attributes across all edges."""
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
        flierprops=dict(markerfacecolor=PALETTE["danger"],
                        marker="o", markersize=4))

    for patch, c in zip(bp["boxes"],
                        [PALETTE["accent"], PALETTE["success"],
                         PALETTE["purple"], PALETTE["orange"]]):
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
    _save(fig, "risk_metrics_distribution.png")
