"""
Heuristic & Edge Cost Function
================================
Each edge has multiple metric attributes.
The combined edge weight is computed from user-defined metric weights.

Metrics (all weights 0–10, higher = more important):
  cost        : minimize raw travel cost
  time        : minimize travel time (correlated with base_cost)
  safety      : prefer safer roads
  traffic     : avoid congested roads
  gender      : prefer gender-safer roads (for female traveler)
  age         : prefer easier roads (for elderly traveler)
"""


DEFAULT_WEIGHTS = {
    "cost":    8,   # strongly minimize cost
    "time":    7,   # minimize time
    "safety":  5,   # moderate safety concern
    "traffic": 5,   # moderate traffic avoidance
    "gender":  3,   # low (general traveler)
    "age":     3,   # low (general traveler)
}

PROFILES = {
    "default": {"cost": 8, "time": 7, "safety": 5, "traffic": 5, "gender": 3, "age": 3},
    "female":  {"cost": 5, "time": 5, "safety": 9, "traffic": 4, "gender": 9, "age": 2},
    "elderly": {"cost": 6, "time": 4, "safety": 8, "traffic": 3, "gender": 3, "age": 9},
    "rush":    {"cost": 5, "time": 9, "safety": 5, "traffic": 9, "gender": 3, "age": 3},
    "budget":  {"cost":10, "time": 5, "safety": 4, "traffic": 4, "gender": 2, "age": 2},
}


def compute_edge_cost(edge_attr: dict, weights: dict) -> float:
    """
    Compute a single scalar cost for an edge given metric weights.

    Formula:
        cost_component   = base_cost  * (w_cost  / 5)
        time_component   = base_cost  * (w_time  / 5)        # proxy: longer road = more time
        safety_penalty   = (10 - safety)    * (w_safety  / 10)
        traffic_penalty  = traffic          * (w_traffic / 10)
        gender_penalty   = (10 - gender)    * (w_gender  / 10)
        age_penalty      = (10 - age)       * (w_age     / 10)

        total = cost_component + time_component + safety_penalty
                + traffic_penalty + gender_penalty + age_penalty + 0.1  (floor)
    """
    w = weights
    bc = edge_attr["base_cost"]

    cost_c    = bc * (w.get("cost",    5) / 5.0)
    time_c    = bc * (w.get("time",    5) / 5.0)
    safety_p  = (10 - edge_attr["safety"])      * (w.get("safety",  0) / 10.0)
    traffic_p = edge_attr["traffic"]             * (w.get("traffic", 0) / 10.0)
    gender_p  = (10 - edge_attr["gender_safe"]) * (w.get("gender",  0) / 10.0)
    age_p     = (10 - edge_attr["age_ease"])     * (w.get("age",     0) / 10.0)

    return cost_c + time_c + safety_p + traffic_p + gender_p + age_p + 0.1


def heuristic(graph, node: int, goal: int, weights: dict) -> float:
    """
    Admissible heuristic for A* / Greedy.
    Uses Euclidean distance scaled by minimum possible edge cost.
    """
    straight_dist = graph.euclidean(node, goal)
    # Optimistic: assume max safety, min traffic, min cost
    min_possible = 0.5 * (weights.get("cost", 5) / 5.0) * straight_dist
    return max(min_possible, 0.0)


def describe_weights(weights: dict) -> str:
    lines = ["  Metric Weights:"]
    for k, v in weights.items():
        bar = "█" * v + "░" * (10 - v)
        lines.append(f"    {k:<10} [{bar}] {v}/10")
    return "\n".join(lines)
