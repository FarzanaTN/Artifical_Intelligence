import osmnx as ox

def load_graph():
    print("Downloading Mirpur road network...")

    # real road network
    G = ox.graph_from_place(
        "Mirpur, Dhaka, Bangladesh",
        network_type="drive"
    )

    # manual coordinates (NO geocode)
    origin_lat, origin_lon = 23.7947, 90.3535   # Mirpur 1
    dest_lat, dest_lon = 23.8060, 90.3685      # Mirpur 10

    origin_node = ox.distance.nearest_nodes(G, origin_lon, origin_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lon, dest_lat)

    return G, origin_node, dest_node