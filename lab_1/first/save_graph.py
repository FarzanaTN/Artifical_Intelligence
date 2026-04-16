import osmnx as ox

# Enable logs + cache
ox.settings.use_cache = True
ox.settings.log_console = True

print("Loading Mirpur small map...")

# Mirpur coordinates (Dhaka)
point = (23.8041, 90.3668)

# Small radius (500m)
G = ox.graph_from_point(point, dist=500, network_type='drive')

print("Map loaded!")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

# 🔹 SAVE GRAPH
ox.save_graphml(G, "mirpur.graphml")
print("Graph saved as mirpur.graphml")