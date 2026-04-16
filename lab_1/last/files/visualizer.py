import os
import osmnx as ox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def save_path_image(G, path, filename):

    fig, ax = ox.plot_graph(
        G,
        show=False,
        close=False,
        node_size=0,
        edge_color="gray"
    )

    if path and len(path) > 1:
        ox.plot_graph_route(
            G,
            route=path,
            route_color="red",
            route_linewidth=3,
            ax=ax,
            show=False,
            close=False
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    print("Saved image:", save_path)