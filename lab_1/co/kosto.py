import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time
import math
import random

# ============================================================================
# SETUP
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "assignment_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("Folder created at:", OUTPUT_FOLDER)

random.seed(42)

# ============================================================================
# 1. RISK MODEL
# ============================================================================
# We assign weights to each risk factor so they sum to 1.0 (or whatever scale you prefer)
W_TRAFFIC, W_SAFETY, W_GENDER, W_AGE = 0.4, 0.3, 0.2, 0.1

def edge_risk_multiplier(traffic, safety, gender_risk, age_risk):
    return 1 + (W_TRAFFIC * traffic + W_SAFETY * safety + W_GENDER * gender_risk + W_AGE * age_risk)

# To ensure our heuristic h(n) never overestimates the real cost (admissibility),
# we calculate the absolute minimum possible risk multiplier in our graph.
MIN_POSSIBLE_TRAFFIC = 1.0
MIN_POSSIBLE_SAFETY = 1.0
MIN_POSSIBLE_GENDER = 1.0
MIN_POSSIBLE_AGE = 1.0

MIN_RISK_FACTOR = edge_risk_multiplier(
    MIN_POSSIBLE_TRAFFIC, MIN_POSSIBLE_SAFETY, MIN_POSSIBLE_GENDER, MIN_POSSIBLE_AGE
)

# ============================================================================
# 2. LOAD GRAPH
# ============================================================================
def load_mirpur_map():
    print("Fetching Mirpur map data...")
    center = (23.8041, 90.3625)
    G = ox.graph_from_point(center, dist=1000, network_type="drive")

    try:
        G = ox.truncate.largest_component(G, strongly=True)
    except:
        largest = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    for u, v, k, data in G.edges(data=True, keys=True):
        length = data.get("length", 1)

        # Expanded deterministic synthetic risk attributes
        traffic = random.uniform(MIN_POSSIBLE_TRAFFIC, 2.0)
        safety = random.uniform(MIN_POSSIBLE_SAFETY, 1.5)
        gender_risk = random.uniform(MIN_POSSIBLE_GENDER, 1.8) # e.g., lighting, isolation
        age_risk = random.uniform(MIN_POSSIBLE_AGE, 1.4)       # e.g., pedestrian crossing difficulty

        data["traffic"] = traffic
        data["safety"] = safety
        data["gender_risk"] = gender_risk
        data["age_risk"] = age_risk

        # g(n): REAL COST = distance × combined risk multiplier
        data["cost"] = length * edge_risk_multiplier(traffic, safety, gender_risk, age_risk)

    print(f"Graph loaded: {len(G.nodes)} nodes.")
    return G

# ============================================================================
# 3. HEURISTIC (FIXED FOR ADMISSIBILITY)
# ============================================================================
def heuristic(graph, node, goal):
    n1 = graph.nodes[node]
    n2 = graph.nodes[goal]

    # Euclidean distance in meters
    dist = math.sqrt((n1["x"] - n2["x"])**2 + (n1["y"] - n2["y"])**2) * 111000  

    # Multiply by the *minimum possible risk* to ensure h(n) <= true cost
    return dist * MIN_RISK_FACTOR

# ============================================================================
# 4. PATH RECONSTRUCTION
# ============================================================================
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return path[::-1]

# ============================================================================
# 5. ALGORITHMS
# ============================================================================

def bfs(graph, start, goal):
    q, vis, cf = [start], {start}, {start: None}
    count = 0
    while q:
        node = q.pop(0)
        count += 1
        if node == goal: return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                q.append(n)
    return None, count

def dfs(graph, start, goal):
    stack, vis, cf = [start], {start}, {start: None}
    count = 0
    while stack:
        node = stack.pop()
        count += 1
        if node == goal: return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                stack.append(n)
    return None, count

def ucs(graph, start, goal):
    pq = [(0, start)]
    g, cf, visited = {start: 0}, {start: None}, set()
    count = 0
    while pq:
        cost, node = heapq.heappop(pq)
        if node in visited: continue
        visited.add(node)
        count += 1
        if node == goal: return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            new_cost = cost + graph[node][n][0]["cost"]
            if n not in g or new_cost < g[n]:
                g[n], cf[n] = new_cost, node
                heapq.heappush(pq, (new_cost, n))
    return None, count

def greedy_bfs(graph, start, goal):
    pq = [(heuristic(graph, start, goal), start)]
    vis, cf = {start}, {start: None}
    count = 0
    while pq:
        _, node = heapq.heappop(pq)
        count += 1
        if node == goal: return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                heapq.heappush(pq, (heuristic(graph, n, goal), n))
    return None, count

def weighted_a_star(graph, start, goal, weight=1.5):
    pq = [(0, start)]
    g, cf, visited = {start: 0}, {start: None}, set()
    count = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node in visited: continue
        visited.add(node)
        count += 1
        if node == goal: return reconstruct_path(cf, goal), count
        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]
            if n not in g or new_g < g[n]:
                g[n], cf[n] = new_g, node
                f = new_g + weight * heuristic(graph, n, goal)
                heapq.heappush(pq, (f, n))
    return None, count

# --- NEW ALGORITHMS ---

def dls(graph, start, goal, limit):
    # Stack stores tuples of: (node, depth, path)
    stack = [(start, 0, [start])]
    count = 0
    while stack:
        node, depth, path = stack.pop()
        count += 1
        
        if node == goal:
            return path, count
            
        if depth < limit:
            for n in graph.neighbors(node):
                if n not in path: # Prevents simple cycles in current branch
                    stack.append((n, depth + 1, path + [n]))
    return None, count

def iddfs(graph, start, goal, max_limit=100):
    total_count = 0
    for limit in range(max_limit):
        path, count = dls(graph, start, goal, limit)
        total_count += count
        if path:
            return path, total_count
    return None, total_count

def bds(graph, start, goal):
    """Bidirectional Search (BFS based)"""
    if start == goal: return [start], 0
    
    q_f, q_b = [start], [goal]
    vis_f, vis_b = {start: None}, {goal: None}
    count = 0
    
    while q_f and q_b:
        # Forward step
        node_f = q_f.pop(0)
        count += 1
        for n in graph.neighbors(node_f):
            if n not in vis_f:
                vis_f[n] = node_f
                q_f.append(n)
                if n in vis_b: # Intersection found
                    path_f = reconstruct_path(vis_f, n)
                    path_b = reconstruct_path(vis_b, vis_b[n])[::-1] # Reverse backward path
                    return path_f + path_b, count
                    
        # Backward step (Note: Assumes directed graph allows backward traversal, NetworkX allows predecessors)
        node_b = q_b.pop(0)
        count += 1
        for n in graph.predecessors(node_b):
            if n not in vis_b:
                vis_b[n] = node_b
                q_b.append(n)
                if n in vis_f: # Intersection found
                    path_f = reconstruct_path(vis_f, vis_f[n])
                    path_b = reconstruct_path(vis_b, n)[::-1]
                    return path_f + path_b, count
                    
    return None, count

def bidirectional_a_star(graph, start, goal):
    """Simplified Bidirectional A*"""
    pq_f = [(heuristic(graph, start, goal), 0, start)]
    pq_b = [(heuristic(graph, goal, start), 0, goal)]
    
    g_f, g_b = {start: 0}, {goal: 0}
    cf_f, cf_b = {start: None}, {goal: None}
    vis_f, vis_b = set(), set()
    
    count = 0
    best_cost = float('inf')
    best_meet_node = None
    
    while pq_f and pq_b:
        # Check termination condition
        if pq_f[0][0] + pq_b[0][0] >= best_cost:
            break
            
        # Expand forward
        _, cost_f, node_f = heapq.heappop(pq_f)
        count += 1
        if node_f not in vis_f:
            vis_f.add(node_f)
            for n in graph.neighbors(node_f):
                new_g = g_f[node_f] + graph[node_f][n][0]["cost"]
                if n not in g_f or new_g < g_f[n]:
                    g_f[n], cf_f[n] = new_g, node_f
                    f = new_g + heuristic(graph, n, goal)
                    heapq.heappush(pq_f, (f, new_g, n))
                    if n in vis_b and g_f[n] + g_b[n] < best_cost:
                        best_cost = g_f[n] + g_b[n]
                        best_meet_node = n
                        
        # Expand backward
        _, cost_b, node_b = heapq.heappop(pq_b)
        count += 1
        if node_b not in vis_b:
            vis_b.add(node_b)
            for n in graph.predecessors(node_b):
                new_g = g_b[node_b] + graph[n][node_b][0]["cost"] # Cost from n to node_b
                if n not in g_b or new_g < g_b[n]:
                    g_b[n], cf_b[n] = new_g, node_b
                    f = new_g + heuristic(graph, start, n)
                    heapq.heappush(pq_b, (f, new_g, n))
                    if n in vis_f and g_f[n] + g_b[n] < best_cost:
                        best_cost = g_f[n] + g_b[n]
                        best_meet_node = n

    if best_meet_node:
        path_f = reconstruct_path(cf_f, best_meet_node)
        # For backward path, trace from meet_node to goal using cf_b
        curr = cf_b[best_meet_node]
        path_b = []
        while curr is not None:
            path_b.append(curr)
            curr = cf_b.get(curr)
        return path_f + path_b, count
        
    return None, count

def ida_star(graph, start, goal):
    threshold = heuristic(graph, start, goal)
    count = [0] # Passed as list to modify inside recursive function

    def search(node, g, current_threshold, path):
        count[0] += 1
        f = g + heuristic(graph, node, goal)
        if f > current_threshold:
            return None, f
        if node == goal:
            return path, "FOUND"
            
        min_over_threshold = float('inf')
        for n in graph.neighbors(node):
            if n not in path: # prevent cycles
                cost = graph[node][n][0]["cost"]
                res, t = search(n, g + cost, current_threshold, path + [n])
                if t == "FOUND":
                    return res, "FOUND"
                if t < min_over_threshold:
                    min_over_threshold = t
                    
        return None, min_over_threshold

    while True:
        path, t = search(start, 0, threshold, [start])
        if t == "FOUND":
            return path, count[0]
        if t == float('inf'):
            return None, count[0]
        threshold = t

# ============================================================================
# 6. VISUALIZATION & RUNNER
# ============================================================================

def plot_route(G, path, name):
    # Make the UI attractive (Dark mode, cyan neon routes)
    fig, ax = ox.plot_graph_route(
        G,
        path,
        bgcolor="#111111",      # Dark background
        node_color="none",      # Hide nodes for cleaner look
        edge_color="#333333",   # Dark grey edges
        edge_linewidth=0.5,
        route_color="#00FFFF",  # Neon Cyan path
        route_linewidth=4,      # Thicker path
        route_alpha=0.9,
        show=False,
        close=True
    )
    fig.savefig(f"{OUTPUT_FOLDER}/{name}_route.png", dpi=300, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)

def plot_complexities(results):
    names = [r["name"] for r in results]
    times = [r["time"] for r in results]
    nodes = [r["nodes"] for r in results]

    plt.style.use('dark_background') # Attractive charts

    plt.figure(figsize=(10, 5))
    plt.bar(names, times, color="#FF0055")
    plt.yscale("log")
    plt.title("Time Complexity (ms)")
    plt.xticks(rotation=45)
    plt.savefig(f"{OUTPUT_FOLDER}/time_complexity.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(names, nodes, color="#00FF99")
    plt.yscale("log")
    plt.title("Space Complexity (Nodes Explored)")
    plt.xticks(rotation=45)
    plt.savefig(f"{OUTPUT_FOLDER}/space_complexity.png", bbox_inches="tight")
    plt.close()

def run_all():
    G = load_mirpur_map()

    start = ox.distance.nearest_nodes(G, 90.3545, 23.7997)
    goal = ox.distance.nearest_nodes(G, 90.3683, 23.8069)

    algos = [
        ("BFS", lambda: bfs(G, start, goal)),
        ("DFS", lambda: dfs(G, start, goal)),
        ("UCS", lambda: ucs(G, start, goal)),
        ("Greedy", lambda: greedy_bfs(G, start, goal)),
        ("WeightedA*", lambda: weighted_a_star(G, start, goal)),
        ("IDDFS", lambda: iddfs(G, start, goal)),
        ("Bidirectional Search", lambda: bds(G, start, goal)),
        ("Bidirectional A*", lambda: bidirectional_a_star(G, start, goal)),
        ("IDA*", lambda: ida_star(G, start, goal))
    ]

    results = []

    for name, func in algos:
        print(f"Running {name}...")
        t0 = time.time()
        
        path, count = func()

        results.append({
            "name": name,
            "time": (time.time() - t0) * 1000,
            "nodes": count
        })

        if path:
            plot_route(G, path, name)
        else:
            print(f"  -> {name} could not find a path!")

    plot_complexities(results)
    print("DONE → All outputs saved in:", OUTPUT_FOLDER)

if __name__ == "__main__":
    run_all()