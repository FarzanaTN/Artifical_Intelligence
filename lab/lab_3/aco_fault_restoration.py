"""
ACO for Disaster Relief Supply Routing
----------------------------------------
Synthetic network of 10 towns/checkpoints. One road/checkpoint (T5) is
blocked (disaster damage - e.g. collapsed bridge, flooded road).
ACO finds the least-cost route for a relief truck to travel from the
supply depot (T1) to the disaster-hit town (T6), hopping through other
towns/checkpoints, avoiding the blocked one.

This is a realistic use case for multi-hop pathfinding: relief trucks
genuinely travel through intermediate towns/checkpoints along real roads,
unlike the earlier power-grid version where "hopping" was an abstraction.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# ----------------------------
# 1. SETUP: Synthetic Road Network
# ----------------------------

random.seed(42)
np.random.seed(42)

NUM_NODES = 10
NODE_NAMES = [f"T{i+1}" for i in range(NUM_NODES)]  # T = Town/checkpoint

# Random 2D coordinates for towns (represents their geographic layout)
coords = {name: (random.uniform(0, 100), random.uniform(0, 100)) for name in NODE_NAMES}

BLOCKED_NODE = "T5"     # road/checkpoint damaged by disaster (e.g. collapsed bridge)
DEPOT_NODE = "T1"       # relief supply depot (healthy, has supplies)
DISASTER_NODE = "T6"    # disaster-hit town that needs supplies

# Build a semi-connected road network: each town connects to its k nearest neighbors
def euclidean(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

K_NEIGHBORS = 4
adj = {name: {} for name in NODE_NAMES}

for n1 in NODE_NAMES:
    dists = [(n2, euclidean(coords[n1], coords[n2])) for n2 in NODE_NAMES if n2 != n1]
    dists.sort(key=lambda x: x[1])
    for n2, d in dists[:K_NEIGHBORS]:
        adj[n1][n2] = d
        adj[n2][n1] = d  # roads are two-way

# Remove blocked node from graph (no relief truck/ant can pass through it)
for n in list(adj[BLOCKED_NODE].keys()):
    del adj[n][BLOCKED_NODE]
del adj[BLOCKED_NODE]

active_nodes = [n for n in NODE_NAMES if n != BLOCKED_NODE]

# ----------------------------
# 2. ACO PARAMETERS
# ----------------------------

NUM_ANTS = 15
NUM_ITERATIONS = 80
ALPHA = 1.0     # pheromone importance
BETA = 2.0      # heuristic (1/distance) importance
RHO = 0.5       # evaporation rate
Q = 100.0       # pheromone deposit constant

# Initialize pheromone on every road segment
pheromone = {n: {m: 1.0 for m in adj[n]} for n in active_nodes}

# ----------------------------
# 3. ACO CORE FUNCTIONS
# ----------------------------

def construct_solution():
    """One ant (simulated relief truck) builds a route from DEPOT to DISASTER_NODE."""
    current = DEPOT_NODE
    path = [current]
    visited = {current}
    total_cost = 0.0

    while current != DISASTER_NODE:
        neighbors = [n for n in adj[current] if n not in visited]
        if not neighbors:
            return None, None  # dead end, this ant failed to find a route

        # Compute transition probabilities
        weights = []
        for n in neighbors:
            tau = pheromone[current][n] ** ALPHA
            eta = (1.0 / adj[current][n]) ** BETA
            weights.append(tau * eta)

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        next_node = np.random.choice(neighbors, p=probs)
        total_cost += adj[current][next_node]
        path.append(next_node)
        visited.add(next_node)
        current = next_node

        if len(path) > NUM_NODES:  # safety guard against loops
            return None, None

    return path, total_cost


def update_pheromones(all_paths):
    """Evaporate then deposit pheromone based on route quality."""
    # Evaporation
    for n in pheromone:
        for m in pheromone[n]:
            pheromone[n][m] *= (1 - RHO)

    # Deposit
    for path, cost in all_paths:
        if path is None:
            continue
        deposit = Q / cost
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit


# ----------------------------
# 4. MAIN ACO LOOP
# ----------------------------

best_path = None
best_cost = float("inf")
convergence = []  # best cost per iteration

for it in range(NUM_ITERATIONS):
    all_paths = []
    for _ in range(NUM_ANTS):
        path, cost = construct_solution()
        all_paths.append((path, cost))
        if path is not None and cost < best_cost:
            best_cost = cost
            best_path = path

    update_pheromones(all_paths)
    convergence.append(best_cost)

print("Blocked road/checkpoint:", BLOCKED_NODE)
print("Relief depot (source):", DEPOT_NODE)
print("Disaster-hit town (destination):", DISASTER_NODE)
print("Best relief route found:", " -> ".join(best_path))
print("Best route cost (total distance):", round(best_cost, 2))

# ----------------------------
# 5. VISUALIZATIONS
# ----------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Road network with blockage + best route ---
ax = axes[0]
for n in NODE_NAMES:
    x, y = coords[n]
    color = "red" if n == BLOCKED_NODE else ("green" if n == DEPOT_NODE else ("orange" if n == DISASTER_NODE else "steelblue"))
    ax.scatter(x, y, s=300, color=color, zorder=3)
    ax.annotate(n, (x, y), ha="center", va="center", color="white", fontweight="bold", zorder=4)

# draw all roads (faint)
drawn = set()
for n in active_nodes:
    for m in adj[n]:
        if (m, n) not in drawn:
            x_vals = [coords[n][0], coords[m][0]]
            y_vals = [coords[n][1], coords[m][1]]
            ax.plot(x_vals, y_vals, color="lightgray", linewidth=1, zorder=1)
            drawn.add((n, m))

# highlight best relief route
for i in range(len(best_path) - 1):
    a, b = best_path[i], best_path[i + 1]
    x_vals = [coords[a][0], coords[b][0]]
    y_vals = [coords[a][1], coords[b][1]]
    ax.plot(x_vals, y_vals, color="crimson", linewidth=3, zorder=2)

ax.set_title("Relief Route (Green=Depot, Orange=Disaster Town, Red=Blocked Road)")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")

# --- Plot 2: Convergence curve ---
ax2 = axes[1]
ax2.plot(convergence, color="darkblue")
ax2.set_title("ACO Convergence (Best Route Cost vs Iteration)")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Best Route Cost")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("aco_result.png", dpi=150)
print("\nSaved plot to aco_result.png")