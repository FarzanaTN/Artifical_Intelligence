import heapq
from collections import deque

# ---------------- BFS ----------------
def bfs(G, start, goal):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return {"path": path}

        if node not in visited:
            visited.add(node)

            for neighbor in G.neighbors(node):
                queue.append((neighbor, path + [neighbor]))

    return None


# ---------------- DFS ----------------
def dfs(G, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()

        if node == goal:
            return {"path": path}

        if node not in visited:
            visited.add(node)

            for neighbor in G.neighbors(node):
                stack.append((neighbor, path + [neighbor]))

    return None


# ---------------- UCS ----------------
def ucs(G, start, goal):
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node == goal:
            return {"path": path, "cost": cost}

        if node not in visited:
            visited.add(node)

            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)[0]
                weight = edge_data.get("length", 1)

                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))

    return None


# ---------------- A* ----------------
def astar(G, start, goal, heuristic_fn):
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        f, node, path = heapq.heappop(pq)

        if node == goal:
            return {"path": path}

        if node not in visited:
            visited.add(node)

            for neighbor in G.neighbors(node):
                h = heuristic_fn(G, neighbor, goal)
                heapq.heappush(pq, (h, neighbor, path + [neighbor]))

    return None