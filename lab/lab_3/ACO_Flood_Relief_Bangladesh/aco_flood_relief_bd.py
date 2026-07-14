"""
ACO for Flood-Risk-Aware Relief Convoy Routing
================================================
MOTIVATION (real-world context, not the dataset):
In July 2026, monsoon floods damaged roads and bridges across several
districts in southeastern Bangladesh, isolating towns and forcing the
Army/Navy to reroute relief convoys around destroyed infrastructure.
This assignment models the underlying routing problem using Ant Colony
Optimization. All town names, coordinates, and road data below are
SYNTHETIC -- generated for demonstrating the algorithm within the scope
of a course assignment, not real geographic data.

PROBLEM MODELED:
- A relief depot (Town1) must deliver supplies to a flood-affected town
  (Town8), which still needs to be reached (it is NOT removed from the
  map -- only the direct road connecting it to the depot is destroyed).
- Every remaining road has two properties: physical distance, and a
  flood-risk score (0 = dry/clear, 1 = heavily waterlogged/hazardous).
- A road that is short but heavily flooded can cost MORE effectively
  than a longer but clear road, because trucks move slower and less
  safely through waterlogged routes.
- ACO must find the route that balances distance against flood risk --
  not just the geometrically shortest path.

WHY ACO (not plain Dijkstra):
Dijkstra finds the optimal path efficiently for a FIXED cost function,
but real flood conditions fluctuate (water levels rise/fall over the
relief operation's duration). ACO's population of ants continuously
re-explores the graph every iteration, so it naturally supports
re-optimizing as conditions change, and it also reveals near-optimal
ALTERNATE routes (via the final pheromone map), which is operationally
useful for coordinating multiple simultaneous convoys.

This script also computes the plain shortest-DISTANCE path (ignoring
flood risk) as a baseline, to show how the ACO/risk-aware result differs.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import heapq

# ============================================================
# 1. SETUP: Synthetic road network with distance + flood risk
# ============================================================

random.seed(22)
np.random.seed(22)

NUM_NODES = 12
NODE_NAMES = [f"Town{i+1}" for i in range(NUM_NODES)]

coords = {name: (random.uniform(0, 100), random.uniform(0, 100)) for name in NODE_NAMES}

DEPOT_NODE = "Town1"       # relief depot -- unaffected, has supplies
DISASTER_NODE = "Town8"    # flood-affected town -- still needs to be reached
BLOCKED_EDGE = (DEPOT_NODE, "Town8_direct")  # placeholder, real blocking done below

RISK_WEIGHT = 4.0   # how heavily flood risk penalizes an edge's effective cost

def euclidean(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Build base road network: k-nearest neighbors
K_NEIGHBORS = 4
raw_adj = {name: set() for name in NODE_NAMES}
for n1 in NODE_NAMES:
    dists = sorted(
        [(n2, euclidean(coords[n1], coords[n2])) for n2 in NODE_NAMES if n2 != n1],
        key=lambda x: x[1],
    )
    for n2, d in dists[:K_NEIGHBORS]:
        raw_adj[n1].add(n2)
        raw_adj[n2].add(n1)

# Assign a flood-risk score to every road (0 = dry, 1 = heavily flooded).
# Roads near the disaster town are given elevated risk, reflecting a flood
# that is concentrated around that area.
edges = {}  # (a,b) sorted tuple -> {"dist":.., "risk":..}
for n1 in NODE_NAMES:
    for n2 in sorted(raw_adj[n1]):  # sorted() makes iteration order deterministic across runs
        key = tuple(sorted((n1, n2)))
        if key in edges:
            continue
        dist = euclidean(coords[n1], coords[n2])
        # base random risk, boosted if the edge touches the disaster area
        base_risk = random.uniform(0.05, 0.35)
        if DISASTER_NODE in key:
            base_risk += random.uniform(0.35, 0.55)
        risk = min(base_risk, 0.95)
        edges[key] = {"dist": dist, "risk": risk}

# --- Simulate flood damage: MULTIPLE roads washed out, not just one.
#     Real floods rarely damage a single road in isolation -- water usually
#     cuts multiple crossings/bridges across a region at once. We remove
#     several edges: the direct depot->disaster road, plus the next two
#     shortest roads leading into the disaster town (the most obvious/
#     tempting routes are exactly the ones flooding is likely to hit,
#     since low-lying roads near a flood-affected town are the most
#     exposed). A connectivity check ensures the disaster town is never
#     fully cut off -- only its BEST routes are destroyed, forcing ACO to
#     find a real (if more roundabout) alternative.

def is_connected(adj_dict, source, target):
    if source not in adj_dict or target not in adj_dict:
        return False
    seen = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        if u == target:
            return True
        for v in adj_dict[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return target in seen

# candidate roads leading into the disaster town, shortest first
disaster_edges = sorted(
    [k for k in edges if DISASTER_NODE in k],
    key=lambda k: edges[k]["dist"],
)

NUM_WASHED_OUT = 3  # how many roads the flood destroys
washed_out_edges = []

for key in disaster_edges:
    if len(washed_out_edges) >= NUM_WASHED_OUT:
        break
    # tentatively remove this edge and check the town is still reachable
    trial_edges = {k: v for k, v in edges.items() if k != key and k not in washed_out_edges}
    trial_adj = {n: set() for n in NODE_NAMES}
    for (a, b) in trial_edges:
        trial_adj[a].add(b)
        trial_adj[b].add(a)
    if is_connected(trial_adj, DEPOT_NODE, DISASTER_NODE):
        washed_out_edges.append(key)

for key in washed_out_edges:
    del edges[key]

WASHED_OUT_EDGES = washed_out_edges  # list of (a,b) tuples, for plotting

# Build adjacency dict (undirected) with cost = dist * (1 + RISK_WEIGHT*risk)
adj = {name: {} for name in NODE_NAMES}
for (a, b), info in edges.items():
    cost = info["dist"] * (1 + RISK_WEIGHT * info["risk"])
    adj[a][b] = {"dist": info["dist"], "risk": info["risk"], "cost": cost}
    adj[b][a] = {"dist": info["dist"], "risk": info["risk"], "cost": cost}

# ============================================================
# 2. BASELINE: plain shortest-DISTANCE path (Dijkstra, ignores risk)
# ============================================================

def dijkstra(source, target, weight_key):
    dist = {n: float("inf") for n in NODE_NAMES}
    prev = {n: None for n in NODE_NAMES}
    dist[source] = 0
    pq = [(0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, info in adj[u].items():
            w = info[weight_key]
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path, dist[target]

shortest_dist_path, shortest_dist_cost = dijkstra(DEPOT_NODE, DISASTER_NODE, "dist")
shortest_dist_total_cost = sum(
    adj[shortest_dist_path[i]][shortest_dist_path[i + 1]]["cost"]
    for i in range(len(shortest_dist_path) - 1)
)

# Ground-truth minimum-EFFECTIVE-COST path (Dijkstra on "cost"). This is not
# shown to the ACO -- it exists purely so we can verify, after the fact,
# that ACO actually converged to the true optimum rather than a mediocre
# local minimum.
true_optimal_path, true_optimal_cost = dijkstra(DEPOT_NODE, DISASTER_NODE, "cost")

# ============================================================
# 3. ACO PARAMETERS
# ============================================================

NUM_ANTS = 50
NUM_ITERATIONS = 300
ALPHA = 1.0     # pheromone importance
BETA = 1.0      # heuristic importance -- kept LOW deliberately: with 3 roads
                # washed out, every remaining path into the disaster town
                # must cross a costly "last-mile" edge. A high beta makes
                # ants myopically avoid that edge (it looks bad one hop at
                # a time) even when it's unavoidable and part of the true
                # optimal route. A lower beta lets pheromone (built from
                # completed full-route costs) drive convergence instead.
RHO = 0.3       # evaporation rate
Q = 300.0       # pheromone deposit constant
ELITIST_BOOST = 3.0   # extra pheromone deposit multiplier for the best-so-far path each iteration

pheromone = {n: {m: 1.0 for m in adj[n]} for n in NODE_NAMES}

# ============================================================
# 4. ACO CORE FUNCTIONS
# ============================================================

def construct_solution():
    """One ant (a candidate relief convoy) builds a route from DEPOT to DISASTER_NODE."""
    current = DEPOT_NODE
    path = [current]
    visited = {current}
    total_cost = 0.0

    while current != DISASTER_NODE:
        neighbors = [n for n in adj[current] if n not in visited]
        if not neighbors:
            return None, None

        weights = []
        for n in neighbors:
            tau = pheromone[current][n] ** ALPHA
            eta = (1.0 / adj[current][n]["cost"]) ** BETA
            weights.append(tau * eta)

        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        next_node = np.random.choice(neighbors, p=probs)

        total_cost += adj[current][next_node]["cost"]
        path.append(next_node)
        visited.add(next_node)
        current = next_node

        if len(path) > NUM_NODES:
            return None, None

    return path, total_cost


def update_pheromones(all_paths, global_best_path, global_best_cost):
    # Evaporation
    for n in pheromone:
        for m in pheromone[n]:
            pheromone[n][m] *= (1 - RHO)

    # Regular deposit -- every ant reinforces the road segments it used,
    # proportional to how good its route was
    for path, cost in all_paths:
        if path is None:
            continue
        deposit = Q / cost
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit

    # Elitist reinforcement -- the best route found so far (across all
    # iterations) gets an extra pheromone boost. This is what lets ACO
    # reliably converge on the true optimum on small graphs like this one,
    # instead of drifting between similarly-good routes indefinitely.
    if global_best_path is not None:
        deposit = ELITIST_BOOST * Q / global_best_cost
        for i in range(len(global_best_path) - 1):
            a, b = global_best_path[i], global_best_path[i + 1]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit


# ============================================================
# 5. MAIN ACO LOOP
# ============================================================

best_path = None
best_cost = float("inf")
convergence_best = []      # best-so-far cost (monotonically non-increasing)
convergence_avg = []       # average cost across all ants this iteration

for it in range(NUM_ITERATIONS):
    all_paths = []
    for _ in range(NUM_ANTS):
        path, cost = construct_solution()
        all_paths.append((path, cost))
        if path is not None and cost < best_cost:
            best_cost = cost
            best_path = path
    update_pheromones(all_paths, best_path, best_cost)

    valid_costs = [c for _, c in all_paths if c is not None]
    convergence_avg.append(np.mean(valid_costs) if valid_costs else np.nan)
    convergence_best.append(best_cost)

best_path_dist = sum(
    adj[best_path[i]][best_path[i + 1]]["dist"] for i in range(len(best_path) - 1)
)

print("=" * 60)
print("Depot (relief source):        ", DEPOT_NODE)
print("Flood-affected town (target): ", DISASTER_NODE)
print("Washed-out roads (impassable):")
for a, b in WASHED_OUT_EDGES:
    print(f"   {a} - {b}")
print("-" * 60)
print("Baseline shortest-DISTANCE path (ignores flood risk):")
print("   Route:", " -> ".join(shortest_dist_path))
print("   Total distance:", round(shortest_dist_cost, 2))
print("   Effective cost (with risk):", round(shortest_dist_total_cost, 2))
print("-" * 60)
print("ACO flood-risk-aware best route:")
print("   Route:", " -> ".join(best_path))
print("   Total distance:", round(best_path_dist, 2))
print("   Effective cost (with risk):", round(best_cost, 2))
print("-" * 60)
gap_pct = 100 * (best_cost - true_optimal_cost) / true_optimal_cost
print("Verification against true optimum (Dijkstra on effective cost):")
print("   True optimal route:  ", " -> ".join(true_optimal_path))
print("   True optimal cost:   ", round(true_optimal_cost, 2))
print("   ACO's cost is", f"{gap_pct:.2f}%", "above the true optimum",
      "(0.00% = ACO found the exact optimum)")
print("=" * 60)

# ============================================================
# 6. VISUALIZATION
# ============================================================

# ============================================================
# 6. VISUALIZATION (three separate figures, saved as separate files)
# ============================================================

def draw_nodes(ax):
    """Draw nodes with clear, non-overlapping labels placed beside each node."""
    for n in NODE_NAMES:
        x, y = coords[n]
        if n == DEPOT_NODE:
            color, size = "green", 550
        elif n == DISASTER_NODE:
            color, size = "orange", 550
        else:
            color, size = "steelblue", 380
        ax.scatter(x, y, s=size, color=color, zorder=4, edgecolors="black", linewidths=1.2)
        # label placed just above the node, with a white background so it's
        # always readable regardless of what edges/lines cross behind it
        ax.annotate(
            n, (x, y), xytext=(0, 13), textcoords="offset points",
            ha="center", va="bottom", color="black", fontweight="bold", fontsize=10,
            zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        )

# --- Figure 1: Network map with risk-colored edges + both routes ---
fig1, ax1 = plt.subplots(figsize=(11, 8.5))

cmap = plt.cm.RdYlGn_r
for (a, b), info in edges.items():
    x_vals = [coords[a][0], coords[b][0]]
    y_vals = [coords[a][1], coords[b][1]]
    ax1.plot(x_vals, y_vals, color=cmap(info["risk"]), linewidth=2, alpha=0.6, zorder=1)

# mark every washed-out road with a dotted black line + X at its midpoint
for idx, (a, b) in enumerate(WASHED_OUT_EDGES):
    wx = [coords[a][0], coords[b][0]]
    wy = [coords[a][1], coords[b][1]]
    ax1.plot(wx, wy, color="black", linewidth=2, linestyle=":", zorder=1)
    mid_x, mid_y = (wx[0] + wx[1]) / 2, (wy[0] + wy[1]) / 2
    ax1.scatter([mid_x], [mid_y], marker="x", s=220, color="black", zorder=5, linewidths=3)
    ax1.annotate("washed out", (mid_x, mid_y), textcoords="offset points",
                 xytext=(8, 8), fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

for i in range(len(shortest_dist_path) - 1):
    a, b = shortest_dist_path[i], shortest_dist_path[i + 1]
    ax1.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]],
              color="royalblue", linewidth=3.5, linestyle="--", zorder=2,
              label="Shortest-distance path (ignores risk)" if i == 0 else None)

for i in range(len(best_path) - 1):
    a, b = best_path[i], best_path[i + 1]
    ax1.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]],
              color="crimson", linewidth=4, zorder=3,
              label="ACO flood-risk-aware path" if i == 0 else None)

draw_nodes(ax1)

ax1.set_title("Flood Relief Routing Network\n"
              "Edge color = flood risk (green=safe, red=risky)  |  "
              "Green node=Depot  |  Orange node=Flood-affected town",
              fontsize=11)
ax1.set_xlabel("X coordinate (synthetic)")
ax1.set_ylabel("Y coordinate (synthetic)")
ax1.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("aco_network.png", dpi=150, bbox_inches="tight")
plt.close(fig1)

# --- Figure 2: Convergence curve ---
fig2, ax2 = plt.subplots(figsize=(8, 5.5))
ax2.plot(convergence_avg, color="gray", linewidth=1.5, label="Average cost (all ants, per iteration)")
ax2.plot(convergence_best, color="darkblue", linewidth=2.5, label="Best-so-far cost")
ax2.set_title("ACO Convergence")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Effective Cost")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("aco_convergence.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# --- Figure 3: Route comparison bar chart ---
fig3, ax3 = plt.subplots(figsize=(7, 5.5))
labels = ["Shortest-Distance\n(baseline)", "ACO\n(risk-aware)"]
distances = [shortest_dist_cost, best_path_dist]
eff_costs = [shortest_dist_total_cost, best_cost]

x = np.arange(len(labels))
width = 0.35
ax3.bar(x - width/2, distances, width, label="Physical distance", color="steelblue")
ax3.bar(x + width/2, eff_costs, width, label="Effective cost (risk-weighted)", color="crimson")
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.set_title("Route Comparison")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("aco_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig3)

print("\nSaved: aco_network.png, aco_convergence.png, aco_comparison.png")