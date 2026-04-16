"""
cost_heuristic.py
==================
Two functions used by ALL search algorithms:

  cost(edge, weights)               → g(n) building block
  heuristic(graph, node, goal, w)   → h(n) admissible estimate

Risk factors (safety, traffic, gender, age, lighting, road_quality)
live ONLY inside cost().
heuristic() uses GPS distance only → stays admissible.
"""

import math

# PROFILES = {
#     "rush_hour": {"distance": 1, "time": 5, "cost": 1},
#     "female": {"distance": 1, "time": 2, "safety": 5},
#     "elderly": {"distance": 2, "time": 1, "slope": 5},
#     "budget": {"distance": 1, "time": 1, "cost": 5},
# }

PROFILES = {
    "default": {
        "cost": 5, "time": 7, "safety": 9, "lighting": 4,
        "traffic": 8, "gender": 3, "age": 3, "road_quality": 4,
    },
    "female": {
        "cost": 5, "time": 5, "safety": 9, "lighting": 9,
        "traffic": 4, "gender": 10, "age": 2, "road_quality": 5,
    },
    "elderly": {
        "cost": 6, "time": 4, "safety": 8, "lighting": 7,
        "traffic": 3, "gender": 3, "age": 10, "road_quality": 9,
    },
    "rush_hour": {
        "cost": 5, "time": 10, "safety": 4, "lighting": 3,
        "traffic": 10, "gender": 3, "age": 2, "road_quality": 4,
    },
    "budget": {
        "cost": 10, "time": 5, "safety": 4, "lighting": 3,
        "traffic": 4, "gender": 2, "age": 2, "road_quality": 3,
    },
}

DEFAULT_WEIGHTS = PROFILES["rush_hour"]  # default profile if none specified


def cost(edge, weights: dict) -> float:
    """
    Compute weighted scalar cost of traversing one edge.

    Positive drivers  (higher raw value → higher cost):
      base_cost_c = base_cost (travel time min) × w_cost/5
      time_c      = base_cost × w_time/5
      traffic_c   = traffic × w_traffic/10

    Penalty terms  (LOWER safety/comfort → HIGHER cost):
      safety_c    = (10 - safety)       × w_safety/10
      lighting_c  = (10 - lighting)     × w_lighting/10
      gender_c    = (10 - gender_safe)  × w_gender/10
      age_c       = (10 - age_ease)     × w_age/10
      quality_c   = (10 - road_quality) × w_road_quality/10
    """
    w = weights
    bc = edge.base_cost

    base_cost_c = bc * (w.get("cost",         5) / 5.0)
    time_c      = bc * (w.get("time",         5) / 5.0)
    traffic_c   = edge.traffic       * (w.get("traffic",      5) / 10.0)
    safety_c    = (10 - edge.safety)       * (w.get("safety",       0) / 10.0)
    lighting_c  = (10 - edge.lighting)     * (w.get("lighting",     0) / 10.0)
    gender_c    = (10 - edge.gender_safe)  * (w.get("gender",       0) / 10.0)
    age_c       = (10 - edge.age_ease)     * (w.get("age",          0) / 10.0)
    quality_c   = (10 - edge.road_quality) * (w.get("road_quality", 0) / 10.0)

    return round(
        base_cost_c + time_c + traffic_c +
        safety_c + lighting_c + gender_c + age_c + quality_c + 0.01,
        5
    )


def heuristic(graph, node_id: int, goal_id: int, weights: dict) -> float:
    """
    Admissible h(n): straight-line GPS distance × minimum cost rate.
    Never overestimates → A* remains optimal.
    """
    dist_km = graph.euclidean(node_id, goal_id)
    w_cost  = weights.get("cost", 5)
    w_time  = weights.get("time", 5)
    min_rate = 0.4 * ((w_cost + w_time) / 10.0)
    return round(dist_km * min_rate, 5)


def path_total_cost(graph, path: list, weights: dict) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for nbr, e in graph.neighbors(a):
            if nbr == b:
                total += cost(e, weights)
                break
    return round(total, 4)


def describe_weights(weights: dict) -> str:
    lines = ["  Metric Weights:"]
    for k, v in sorted(weights.items()):
        bar = "█" * v + "░" * (10 - v)
        lines.append(f"    {k:<14} [{bar}] {v:>2}/10")
    return "\n".join(lines)


