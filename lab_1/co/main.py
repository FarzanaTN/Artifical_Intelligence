# """
# =============================================================================
# AI Assignment: Informed vs Uninformed Search Algorithms
# Map: Mirpur, Dhaka City (OpenStreetMap)
# Source: Mirpur-1 Bus Stand
# Destination: Mirpur-10
# Algorithms: BFS, DFS, UCS (uninformed) | Greedy Best-First, A* (informed)
# =============================================================================
# """

# import os
# import osmnx as ox
# import networkx as nx
# import matplotlib.pyplot as plt
# import heapq
# import time
# import math

# # Ensure output folder exists
# OUTPUT_FOLDER = "output"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # ============================================================================
# # 1. LOAD MIRPUR MAP USING OSMNX
# # ============================================================================
# def load_graph():
#     print("Loading Mirpur map...")
#     graph = ox.graph_from_place("Mirpur, Dhaka, Bangladesh", network_type="drive")
    
#     # Ensure compatibility with older osmnx versions
#     try:
#         graph = ox.utils_graph.get_largest_component(graph, strongly=True)
#     except AttributeError:
#         largest_component = max(nx.strongly_connected_components(graph), key=len)
#         graph = graph.subgraph(largest_component).copy()
    
#     print(f"Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges\n")
#     return graph

# # ============================================================================
# # 2. RISK FUNCTION AND HEURISTIC
# # ============================================================================
# def heuristic(graph, node, goal):
#     """Heuristic function: h(n) = Euclidean distance."""
#     x1, y1 = graph.nodes[node]['x'], graph.nodes[node]['y']
#     x2, y2 = graph.nodes[goal]['x'], graph.nodes[goal]['y']
#     euclidean_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     return euclidean_distance

# def actual_cost(graph, u, v, weights):
#     """Calculate actual cost of traversing an edge."""
#     edge_data = graph[u][v][0]
#     distance = edge_data.get('length', 1)
#     return distance * weights["distance"]

# # ============================================================================
# # 3. SEARCH ALGORITHMS
# # ============================================================================

# # BFS (Uninformed)
# def bfs(graph, source, goal):
#     visited = set()
#     queue = [(source, [source])]
#     while queue:
#         node, path = queue.pop(0)
#         if node in visited:
#             continue
#         visited.add(node)
#         if node == goal:
#             return path, len(visited)
#         for neighbor in graph.neighbors(node):
#             if neighbor not in visited:
#                 queue.append((neighbor, path + [neighbor]))
#     return None, len(visited)

# # DFS (Uninformed)
# def dfs(graph, source, goal):
#     visited = set()
#     stack = [(source, [source])]
#     while stack:
#         node, path = stack.pop()
#         if node in visited:
#             continue
#         visited.add(node)
#         if node == goal:
#             return path, len(visited)
#         for neighbor in graph.neighbors(node):
#             if neighbor not in visited:
#                 stack.append((neighbor, path + [neighbor]))
#     return None, len(visited)

# # UCS (Uninformed)
# def ucs(graph, source, goal):
#     visited = set()
#     queue = [(0, source, [source])]  # (cost, node, path)
#     while queue:
#         cost, node, path = heapq.heappop(queue)
#         if node in visited:
#             continue
#         visited.add(node)
#         if node == goal:
#             return path, len(visited)
#         for neighbor in graph.neighbors(node):
#             edge_cost = graph[node][neighbor][0].get('length', 1)
#             heapq.heappush(queue, (cost + edge_cost, neighbor, path + [neighbor]))
#     return None, len(visited)

# # Greedy Best-First Search (Informed)
# def greedy(graph, source, goal):
#     visited = set()
#     queue = [(heuristic(graph, source, goal), source, [source])]  # (heuristic, node, path)
#     while queue:
#         _, node, path = heapq.heappop(queue)
#         if node in visited:
#             continue
#         visited.add(node)
#         if node == goal:
#             return path, len(visited)
#         for neighbor in graph.neighbors(node):
#             heapq.heappush(queue, (heuristic(graph, neighbor, goal), neighbor, path + [neighbor]))
#     return None, len(visited)

# # A* Search (Informed)
# def a_star(graph, source, goal, weights):
#     visited = set()
#     queue = [(0, 0, source, [source])]  # (f, g, node, path)
#     while queue:
#         _, g, node, path = heapq.heappop(queue)
#         if node in visited:
#             continue
#         visited.add(node)
#         if node == goal:
#             return path, len(visited)
#         for neighbor in graph.neighbors(node):
#             edge_cost = actual_cost(graph, node, neighbor, weights)
#             h = heuristic(graph, neighbor, goal)
#             heapq.heappush(queue, (g + edge_cost + h, g + edge_cost, neighbor, path + [neighbor]))
#     return None, len(visited)

# # ============================================================================
# # 4. VISUALIZATION
# # ============================================================================
# def save_path_image(graph, path, filename):
#     """Save an image of the graph with the final path highlighted."""
#     fig, ax = ox.plot_graph_route(
#         graph,
#         path,
#         route_linewidth=2,
#         node_size=10,
#         show=False,
#         close=False
#     )
#     filepath = os.path.join(OUTPUT_FOLDER, filename)
#     plt.savefig(filepath)
#     plt.close()
#     print(f"Path saved to {filepath}")

# # ============================================================================
# # 5. MAIN EXECUTION
# # ============================================================================
# if __name__ == "__main__":
#     # Load the graph
#     graph = load_graph()

#     # Define source and goal coordinates
#     mirpur1_coord = (23.7997, 90.3545)  # Mirpur-1 Bus Stand
#     mirpur10_coord = (23.8084, 90.3682)  # Mirpur-10

#     # Find nearest nodes to the coordinates
#     source = ox.distance.nearest_nodes(graph, mirpur1_coord[1], mirpur1_coord[0])
#     goal = ox.distance.nearest_nodes(graph, mirpur10_coord[1], mirpur10_coord[0])

#     # Define weights for cost calculation
#     weights = {
#         "distance": 1
#     }

#     print(f"Source: {source}, Goal: {goal}\n")

#     # Run BFS
#     print("Running BFS...")
#     start_time = time.time()
#     bfs_path, bfs_visited = bfs(graph, source, goal)
#     bfs_time = time.time() - start_time
#     save_path_image(graph, bfs_path, "bfs_path.png")

#     # Run DFS
#     print("Running DFS...")
#     start_time = time.time()
#     dfs_path, dfs_visited = dfs(graph, source, goal)
#     dfs_time = time.time() - start_time
#     save_path_image(graph, dfs_path, "dfs_path.png")

#     # Run UCS
#     print("Running UCS...")
#     start_time = time.time()
#     ucs_path, ucs_visited = ucs(graph, source, goal)
#     ucs_time = time.time() - start_time
#     save_path_image(graph, ucs_path, "ucs_path.png")

#     # Run Greedy
#     print("Running Greedy...")
#     start_time = time.time()
#     greedy_path, greedy_visited = greedy(graph, source, goal)
#     greedy_time = time.time() - start_time
#     save_path_image(graph, greedy_path, "greedy_path.png")

#     # Run A*
#     print("Running A*...")
#     start_time = time.time()
#     a_star_path, a_star_visited = a_star(graph, source, goal, weights)
#     a_star_time = time.time() - start_time
#     save_path_image(graph, a_star_path, "a_star_path.png")

#     # Compare results
#     print("\nComparison:")
#     print(f"BFS: Time = {bfs_time:.4f}s, Nodes Visited = {bfs_visited}")
#     print(f"DFS: Time = {dfs_time:.4f}s, Nodes Visited = {dfs_visited}")
#     print(f"UCS: Time = {ucs_time:.4f}s, Nodes Visited = {ucs_visited}")
#     print(f"Greedy: Time = {greedy_time:.4f}s, Nodes Visited = {greedy_visited}")
#     print(f"A*: Time = {a_star_time:.4f}s, Nodes Visited = {a_star_visited}")


import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time
import math
import random

# Setup
OUTPUT_FOLDER = "assignment_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================================
# 1. MAP LOADING & RISK SETUP
# ============================================================================
def load_mirpur_map():
    print("Fetching Mirpur map data...")
    center = (23.8041, 90.3625) 
    # Distance in meters; moderate size for performance
    graph = ox.graph_from_point(center, dist=1000, network_type="drive")
    
    # Fix for OSMnx 2.0+ attribute errors
    try:
        graph = ox.truncate.largest_component(graph, strongly=True)
    except AttributeError:
        largest_component = max(nx.strongly_connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()

    # Injecting Risk Factors
    for u, v, k, data in graph.edges(data=True, keys=True):
        length = data.get('length', 1)
        # Risk factors (1.0 = baseline, >1.0 = higher cost)
        traffic_factor = random.uniform(1.0, 2.5)
        safety_score = random.uniform(1.0, 1.2)
        # Actual Cost = Distance * Risk
        data['total_cost'] = length * traffic_factor * safety_score
        
    print(f"Graph loaded: {len(graph.nodes)} nodes.")
    return graph

def get_h(graph, node, goal):
    """Euclidean distance (Admissible)."""
    n1, n2 = graph.nodes[node], graph.nodes[goal]
    return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2) * 111000

def reconstruct_path(came_from, current):
    """Safe path reconstruction to avoid adding None to the node list."""
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return path[::-1]

# ============================================================================
# 2. SEARCH ALGORITHMS
# ============================================================================

# --- Uninformed ---
def bfs(graph, start, goal):
    queue, visited, came_from = [start], {start}, {start: None}
    count = 0
    while queue:
        curr = queue.pop(0); count += 1
        if curr == goal: return reconstruct_path(came_from, goal), count
        for n in graph.neighbors(curr):
            if n not in visited:
                visited.add(n); came_from[n] = curr; queue.append(n)
    return None, count

def dfs(graph, start, goal):
    stack, visited, came_from = [start], {start}, {start: None}
    count = 0
    while stack:
        curr = stack.pop(); count += 1
        if curr == goal: return reconstruct_path(came_from, goal), count
        for n in graph.neighbors(curr):
            if n not in visited:
                visited.add(n); came_from[n] = curr; stack.append(n)
    return None, count

def ucs(graph, start, goal):
    pq, visited, came_from = [(0, start)], {}, {start: None}
    count = 0
    while pq:
        cost, curr = heapq.heappop(pq)
        if curr in visited: continue
        visited[curr] = cost; count += 1
        if curr == goal: return reconstruct_path(came_from, goal), count
        for n in graph.neighbors(curr):
            new_cost = cost + graph[curr][n][0]['total_cost']
            if n not in visited or new_cost < visited[n]:
                came_from[n] = curr; heapq.heappush(pq, (new_cost, n))
    return None, count

def dls(graph, start, goal, limit):
    def rdls(node, g, d, v_path, info, cf):
        info['count'] += 1
        if node == g: return True
        if d <= 0: return False
        for n in graph.neighbors(node):
            if n not in v_path:
                cf[n] = node
                if rdls(n, g, d-1, v_path | {n}, info, cf): return True
        return False
    inf, cf = {'count': 0}, {start: None}
    found = rdls(start, goal, limit, {start}, inf, cf)
    return (reconstruct_path(cf, goal), inf['count']) if found else (None, inf['count'])

def iddfs(graph, start, goal, max_depth=25):
    total = 0
    for d in range(max_depth):
        p, c = dls(graph, start, goal, d)
        total += c
        if p: return p, total
    return None, total

def bds(graph, start, goal):
    f_q, b_q = [start], [goal]
    f_vis, b_vis = {start: None}, {goal: None}
    count = 0
    while f_q and b_q:
        c_f = f_q.pop(0); count += 1
        for n in graph.neighbors(c_f):
            if n not in f_vis:
                f_vis[n] = c_f; f_q.append(n)
                if n in b_vis: 
                    return reconstruct_path(f_vis, n) + reconstruct_path(b_vis, n)[::-1][1:], count
        c_b = b_q.pop(0); count += 1
        for n in graph.predecessors(c_b):
            if n not in b_vis:
                b_vis[n] = c_b; b_q.append(n)
                if n in f_vis: 
                    return reconstruct_path(f_vis, n) + reconstruct_path(b_vis, n)[::-1][1:], count
    return None, count

# --- Informed ---
def greedy_bfs(graph, start, goal):
    pq, visited, came_from = [(get_h(graph, start, goal), start)], {start}, {start: None}
    count = 0
    while pq:
        _, curr = heapq.heappop(pq); count += 1
        if curr == goal: return reconstruct_path(came_from, goal), count
        for n in graph.neighbors(curr):
            if n not in visited:
                visited.add(n); came_from[n] = curr; heapq.heappush(pq, (get_h(graph, n, goal), n))
    return None, count

def weighted_a_star(graph, start, goal, weight=1.5):
    pq, g_score, came_from = [(0, start)], {start: 0}, {start: None}
    count = 0
    while pq:
        _, curr = heapq.heappop(pq); count += 1
        if curr == goal: return reconstruct_path(came_from, goal), count
        for n in graph.neighbors(curr):
            new_g = g_score[curr] + graph[curr][n][0]['total_cost']
            if n not in g_score or new_g < g_score[n]:
                g_score[n] = new_g
                f = new_g + weight * get_h(graph, n, goal)
                came_from[n] = curr; heapq.heappush(pq, (f, n))
    return None, count

# ============================================================================
# 3. RUNNER & VISUALIZATION
# ============================================================================
def plot_complexities(results):
    names = [r['name'] for r in results]
    times = [r['time'] for r in results]
    nodes = [r['nodes'] for r in results]

    # --- 1. Save Time Complexity Graph ---
    plt.figure(figsize=(10, 6))
    plt.bar(names, times, color='skyblue', edgecolor='navy')
    plt.yscale('log')  # <--- LOG SCALE: Makes small values visible next to large ones
    plt.ylabel("Time (ms) - Log Scale")
    plt.title("Time Complexity Comparison")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/comparison_time.png")
    plt.close()

    # --- 2. Save Space Complexity Graph ---
    plt.figure(figsize=(10, 6))
    plt.bar(names, nodes, color='salmon', edgecolor='darkred')
    plt.yscale('log')  # <--- LOG SCALE: Essential for Node counts
    plt.ylabel("Nodes Explored - Log Scale")
    plt.title("Space Complexity Comparison")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/comparison_space.png")
    plt.close()


def run_all():
    G = load_mirpur_map()
    # Mirpur-1 Bus Stand to Mirpur-10 Circle
    start_node = ox.distance.nearest_nodes(G, 90.3545, 23.7997)
    goal_node = ox.distance.nearest_nodes(G, 90.3683, 23.8069)

    algos = [
        ("BFS", lambda: bfs(G, start_node, goal_node)),
        ("DFS", lambda: dfs(G, start_node, goal_node)),
        ("UCS", lambda: ucs(G, start_node, goal_node)),
        ("DLS (20)", lambda: dls(G, start_node, goal_node, 20)),
        ("IDDFS", lambda: iddfs(G, start_node, goal_node)),
        ("BDS", lambda: bds(G, start_node, goal_node)),
        ("Greedy BFS", lambda: greedy_bfs(G, start_node, goal_node)),
        ("Weighted A*", lambda: weighted_a_star(G, start_node, goal_node, 2.0))
    ]

    results_summary = []
    for name, func in algos:
        print(f"Running {name}...")
        t0 = time.time()
        path, count = func()
        duration = (time.time() - t0) * 1000
        
        results_summary.append({'name': name, 'time': duration, 'nodes': count})

        if path and len(path) > 1:
            try:
                fig, ax = ox.plot_graph_route(G, path, route_color="red", bgcolor="white", 
                                              node_size=0, route_linewidth=5, show=False, close=False)
                ax.set_title(f"{name} | Nodes: {count}")
                plt.savefig(f"{OUTPUT_FOLDER}/{name.lower().replace(' ', '_')}.png")
                plt.close()
            except Exception as e:
                print(f"Error plotting {name}: {e}")
        else:
            print(f"No path found for {name}.")

    plot_complexities(results_summary)
    print(f"\nAll images and comparison graphs saved in '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    run_all()