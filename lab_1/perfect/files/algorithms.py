"""
Search Algorithms
==================
Uninformed:
  1. BFS  – Breadth-First Search
  2. DFS  – Depth-First Search
  3. UCS  – Uniform Cost Search

Informed:
  4. Greedy Best-First Search
  5. A*   – A-Star Search

Each function returns a SearchResult named-tuple:
  path          : list of node ids from source to destination
  visited_order : list of node ids in the order they were expanded
  nodes_visited : total number of nodes expanded
  total_cost    : cumulative edge cost of the found path
  found         : True if path found
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from heuristic import compute_edge_cost, heuristic


@dataclass
class SearchResult:
    algorithm:     str
    path:          List[int]
    visited_order: List[int]
    nodes_visited: int
    total_cost:    float
    found:         bool
    path_length:   int = 0   # number of edges

    def __post_init__(self):
        self.path_length = max(0, len(self.path) - 1)


def _reconstruct(parent, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


# ─────────────────────────────────────────────
# 1. BFS  (Uninformed)
# ─────────────────────────────────────────────
def bfs(graph, source: int, goal: int, weights: dict) -> SearchResult:
    """
    Breadth-First Search.
    Expands nodes level-by-level (FIFO queue).
    Finds shortest path by hop count, not by cost.
    """
    queue = deque([source])
    parent = {source: None}
    visited_order = []
    cost_map = {source: 0.0}

    while queue:
        node = queue.popleft()
        visited_order.append(node)

        if node == goal:
            path = _reconstruct(parent, goal)
            return SearchResult(
                algorithm="BFS",
                path=path,
                visited_order=visited_order,
                nodes_visited=len(visited_order),
                total_cost=cost_map[goal],
                found=True,
            )

        for nb, attr in graph.neighbors(node):
            if nb not in parent:
                parent[nb] = node
                cost_map[nb] = cost_map[node] + compute_edge_cost(attr, weights)
                queue.append(nb)

    return SearchResult("BFS", [], visited_order, len(visited_order), 0, False)


# ─────────────────────────────────────────────
# 2. DFS  (Uninformed)
# ─────────────────────────────────────────────
def dfs(graph, source: int, goal: int, weights: dict) -> SearchResult:
    """
    Depth-First Search.
    Explores as deep as possible before backtracking (LIFO stack).
    May not find optimal path.
    """
    stack = [source]
    parent = {source: None}
    visited_order = []
    cost_map = {source: 0.0}
    visited_set = set()

    while stack:
        node = stack.pop()
        if node in visited_set:
            continue
        visited_set.add(node)
        visited_order.append(node)

        if node == goal:
            path = _reconstruct(parent, goal)
            return SearchResult(
                algorithm="DFS",
                path=path,
                visited_order=visited_order,
                nodes_visited=len(visited_order),
                total_cost=cost_map[goal],
                found=True,
            )

        for nb, attr in reversed(graph.neighbors(node)):
            if nb not in visited_set:
                if nb not in parent:
                    parent[nb] = node
                    cost_map[nb] = cost_map[node] + compute_edge_cost(attr, weights)
                stack.append(nb)

    return SearchResult("DFS", [], visited_order, len(visited_order), 0, False)


# ─────────────────────────────────────────────
# 3. UCS  (Uninformed)
# ─────────────────────────────────────────────
def ucs(graph, source: int, goal: int, weights: dict) -> SearchResult:
    """
    Uniform Cost Search.
    Priority queue ordered by cumulative path cost g(n).
    Guaranteed to find the cheapest path.
    """
    # (cost, node)
    heap = [(0.0, source)]
    parent = {source: None}
    cost_so_far = {source: 0.0}
    visited_order = []
    visited_set = set()

    while heap:
        g, node = heapq.heappop(heap)

        if node in visited_set:
            continue
        visited_set.add(node)
        visited_order.append(node)

        if node == goal:
            path = _reconstruct(parent, goal)
            return SearchResult(
                algorithm="UCS",
                path=path,
                visited_order=visited_order,
                nodes_visited=len(visited_order),
                total_cost=g,
                found=True,
            )

        for nb, attr in graph.neighbors(node):
            new_cost = g + compute_edge_cost(attr, weights)
            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                parent[nb] = node
                heapq.heappush(heap, (new_cost, nb))

    return SearchResult("UCS", [], visited_order, len(visited_order), 0, False)


# ─────────────────────────────────────────────
# 4. Greedy Best-First  (Informed)
# ─────────────────────────────────────────────
def greedy(graph, source: int, goal: int, weights: dict) -> SearchResult:
    """
    Greedy Best-First Search.
    Priority queue ordered ONLY by heuristic h(n).
    Fast but not guaranteed to be optimal.
    """
    # (h, node)
    h0 = heuristic(graph, source, goal, weights)
    heap = [(h0, source)]
    parent = {source: None}
    cost_map = {source: 0.0}
    visited_order = []
    visited_set = set()

    while heap:
        _, node = heapq.heappop(heap)

        if node in visited_set:
            continue
        visited_set.add(node)
        visited_order.append(node)

        if node == goal:
            path = _reconstruct(parent, goal)
            return SearchResult(
                algorithm="Greedy",
                path=path,
                visited_order=visited_order,
                nodes_visited=len(visited_order),
                total_cost=cost_map[goal],
                found=True,
            )

        for nb, attr in graph.neighbors(node):
            if nb not in visited_set:
                new_cost = cost_map[node] + compute_edge_cost(attr, weights)
                if nb not in cost_map or new_cost < cost_map[nb]:
                    cost_map[nb] = new_cost
                    parent[nb] = node
                h = heuristic(graph, nb, goal, weights)
                heapq.heappush(heap, (h, nb))

    return SearchResult("Greedy", [], visited_order, len(visited_order), 0, False)


# ─────────────────────────────────────────────
# 5. A*  (Informed)
# ─────────────────────────────────────────────
def astar(graph, source: int, goal: int, weights: dict) -> SearchResult:
    """
    A* Search.
    Priority queue ordered by f(n) = g(n) + h(n).
    Optimal (when heuristic is admissible) and efficient.
    """
    h0 = heuristic(graph, source, goal, weights)
    heap = [(h0, 0.0, source)]   # (f, g, node)
    parent = {source: None}
    cost_so_far = {source: 0.0}
    visited_order = []
    visited_set = set()

    while heap:
        f, g, node = heapq.heappop(heap)

        if node in visited_set:
            continue
        visited_set.add(node)
        visited_order.append(node)

        if node == goal:
            path = _reconstruct(parent, goal)
            return SearchResult(
                algorithm="A*",
                path=path,
                visited_order=visited_order,
                nodes_visited=len(visited_order),
                total_cost=g,
                found=True,
            )

        for nb, attr in graph.neighbors(node):
            new_g = g + compute_edge_cost(attr, weights)
            if nb not in cost_so_far or new_g < cost_so_far[nb]:
                cost_so_far[nb] = new_g
                parent[nb] = node
                h = heuristic(graph, nb, goal, weights)
                heapq.heappush(heap, (new_g + h, new_g, nb))

    return SearchResult("A*", [], visited_order, len(visited_order), 0, False)
