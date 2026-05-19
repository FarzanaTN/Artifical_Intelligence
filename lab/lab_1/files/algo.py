

import heapq
import math

from graph import heuristic, reconstruct_path



def bfs(graph, start, goal):
    
    q   = [start]
    vis = {start}
    cf  = {start: None}
    nodes_explored   = 0
    max_frontier     = 1

    while q:
        nodes_explored += 1
        node = q.pop(0)
        if node == goal:
            return reconstruct_path(cf, goal), nodes_explored, max_frontier, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                q.append(n)
        max_frontier = max(max_frontier, len(q))

    return None, nodes_explored, max_frontier, None


def dfs(graph, start, goal):
    
    stack = [start]
    vis   = {start}
    cf    = {start: None}
    nodes_explored   = 0
    max_frontier     = 1

    while stack:
        nodes_explored += 1
        node = stack.pop()
        if node == goal:
            return reconstruct_path(cf, goal), nodes_explored, max_frontier, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                stack.append(n)
        max_frontier = max(max_frontier, len(stack))

    return None, nodes_explored, max_frontier, None


# def dls(graph, start, goal, limit: int):
#     """
#     Depth-Limited Search
#     f(n): NONE — DFS capped at hop-depth `limit`.
#     Space ~ O(b*limit) — stack depth bounded by limit.

#     IMPORTANT: uses the current PATH as the only cycle guard, NOT a global
#     visited set.  A global visited set would prevent IDDFS from revisiting
#     nodes at deeper levels across iterations.
#     """
#     # Stack entries: (node, depth, path_to_node_as_set)
#     path_set = {start}
#     stack    = [(start, 0, path_set)]
#     cf       = {start: None}
#     nodes_explored = 0
#     max_frontier   = 1

#     while stack:
#         nodes_explored += 1
#         node, depth, cur_path_set = stack.pop()
#         if node == goal:
#             return reconstruct_path(cf, goal), nodes_explored, max_frontier, None
#         if depth < limit:
#             for n in graph.neighbors(node):
#                 if n not in cur_path_set:
#                     new_path_set = cur_path_set | {n}
#                     cf[n]        = node
#                     stack.append((n, depth + 1, new_path_set))
#         max_frontier = max(max_frontier, len(stack))

#     return None, nodes_explored, max_frontier, None


def dls(graph, start, goal, limit):
    stack = [(start, 0)]
    path = {start}

    cf = {start: None}
    nodes_explored = 0
    max_frontier = 1

    while stack:
        node, depth = stack.pop()
        nodes_explored += 1

        if node == goal:
            return reconstruct_path(cf, goal), nodes_explored, max_frontier, None

        if depth < limit:
            for n in graph.neighbors(node):
                if n not in path:
                    path.add(n)
                    cf[n] = node
                    stack.append((n, depth + 1))

        max_frontier = max(max_frontier, len(stack))

    return None, nodes_explored, max_frontier, None

def iddfs(graph, start, goal, max_depth: int = 60):
   
    total_explored  = 0
    overall_max_frontier = 0

    for limit in range(1, max_depth + 1):
        path, explored, max_fr, _ = dls(graph, start, goal, limit)
        total_explored      += explored
        overall_max_frontier = max(overall_max_frontier, max_fr)
        if path is not None:
            return path, total_explored, overall_max_frontier, None

    return None, total_explored, overall_max_frontier, None



def ucs(graph, start, goal):
   
    pq      = [(0, start)]
    g       = {start: 0}
    cf      = {start: None}
    visited = set()
    nodes_explored = 0
    max_frontier   = 1

    while pq:
        cost, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        nodes_explored += 1                # counted on first settlement
        if node == goal:
            return reconstruct_path(cf, goal), nodes_explored, max_frontier, None
        for n in graph.neighbors(node):
            nc = cost + graph[node][n][0]["cost"]
            if n not in g or nc < g[n]:
                g[n]  = nc
                cf[n] = node
                heapq.heappush(pq, (nc, n))
        max_frontier = max(max_frontier, len(pq))

    return None, nodes_explored, max_frontier, None


def greedy_bfs(graph, start, goal):
    
    pq    = [(heuristic(graph, start, goal), start)]
    vis   = {start}
    cf    = {start: None}
    nodes_explored = 0
    max_frontier   = 1

    while pq:
        _, node = heapq.heappop(pq)
        nodes_explored += 1
        if node == goal:
            return reconstruct_path(cf, goal), nodes_explored, max_frontier, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                heapq.heappush(pq, (heuristic(graph, n, goal), n))
        max_frontier = max(max_frontier, len(pq))

    return None, nodes_explored, max_frontier, None


def a_star(graph, start, goal):
   
    pq    = [(heuristic(graph, start, goal), start)]
    g     = {start: 0}
    cf    = {start: None}
    closed = set()
    nodes_explored = 0
    max_frontier   = 1

    while pq:
        _, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        nodes_explored += 1
        if node == goal:
            return reconstruct_path(cf, goal), nodes_explored, max_frontier, None
        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]       # g(n)
            if n not in g or new_g < g[n]:
                g[n]  = new_g
                f     = new_g + heuristic(graph, n, goal)     # f = g + h
                cf[n] = node
                heapq.heappush(pq, (f, n))
        max_frontier = max(max_frontier, len(pq))

    return None, nodes_explored, max_frontier, None


def weighted_a_star(graph, start, goal, weight: float = 1.5):
    
    pq    = [(0, start)]
    g     = {start: 0}
    cf    = {start: None}
    closed = set()
    nodes_explored = 0
    max_frontier   = 1

    while pq:
        _, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        nodes_explored += 1
        if node == goal:
            return reconstruct_path(cf, goal), nodes_explored, max_frontier, None
        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]
            if n not in g or new_g < g[n]:
                g[n]  = new_g
                f     = new_g + weight * heuristic(graph, n, goal)
                cf[n] = node
                heapq.heappush(pq, (f, n))
        max_frontier = max(max_frontier, len(pq))

    return None, nodes_explored, max_frontier, None



def bidirectional_dijkstra(graph, start, goal):
   
    pq_f  = [(0, start)];  g_f = {start: 0};  cf_f = {start: None};  vis_f = set()
    rev_G = graph.reverse(copy=False)
    pq_b  = [(0, goal)];   g_b = {goal: 0};   cf_b = {goal: None};   vis_b = set()
    best  = math.inf
    meet  = None
    nodes_explored = 0
    max_frontier   = 1

    while pq_f or pq_b:
        if pq_f:
            c, u = heapq.heappop(pq_f)
            if u not in vis_f:
                vis_f.add(u)
                nodes_explored += 1
                if u in vis_b:
                    cand = c + g_b.get(u, math.inf)
                    if cand < best:
                        best = cand; meet = u
                for v in graph.neighbors(u):
                    nc = c + graph[u][v][0]["cost"]
                    if v not in g_f or nc < g_f[v]:
                        g_f[v] = nc; cf_f[v] = u
                        heapq.heappush(pq_f, (nc, v))

        if pq_b:
            c, u = heapq.heappop(pq_b)
            if u not in vis_b:
                vis_b.add(u)
                nodes_explored += 1
                if u in vis_f:
                    cand = g_f.get(u, math.inf) + c
                    if cand < best:
                        best = cand; meet = u
                for v in rev_G.neighbors(u):
                    nc = c + rev_G[u][v][0]["cost"]
                    if v not in g_b or nc < g_b[v]:
                        g_b[v] = nc; cf_b[v] = u
                        heapq.heappush(pq_b, (nc, v))

        max_frontier = max(max_frontier, len(pq_f) + len(pq_b))
        if pq_f and pq_b and pq_f[0][0] + pq_b[0][0] >= best:
            break

    if meet is None:
        return None, nodes_explored, max_frontier, None

    fwd = reconstruct_path(cf_f, meet)
    bwd = reconstruct_path(cf_b, meet)
    return fwd + bwd[-2::-1], nodes_explored, max_frontier, meet


def bidirectional_astar(graph, start, goal):
   
    pq_f  = [(heuristic(graph, start, goal), start)]
    g_f   = {start: 0};  cf_f = {start: None};  vis_f = set()
    rev_G = graph.reverse(copy=False)
    pq_b  = [(heuristic(graph, goal, start), goal)]
    g_b   = {goal: 0};   cf_b = {goal: None};   vis_b = set()
    best  = math.inf
    meet  = None
    nodes_explored = 0
    max_frontier   = 1

    while pq_f or pq_b:
        if pq_f:
            _, u = heapq.heappop(pq_f)
            if u not in vis_f:
                vis_f.add(u)
                nodes_explored += 1
                if u in vis_b:
                    cand = g_f[u] + g_b.get(u, math.inf)
                    if cand < best:
                        best = cand; meet = u
                for v in graph.neighbors(u):
                    ng = g_f[u] + graph[u][v][0]["cost"]
                    if v not in g_f or ng < g_f[v]:
                        g_f[v] = ng; cf_f[v] = u
                        heapq.heappush(pq_f, (ng + heuristic(graph, v, goal), v))

        if pq_b:
            _, u = heapq.heappop(pq_b)
            if u not in vis_b:
                vis_b.add(u)
                nodes_explored += 1
                if u in vis_f:
                    cand = g_f.get(u, math.inf) + g_b[u]
                    if cand < best:
                        best = cand; meet = u
                for v in rev_G.neighbors(u):
                    ng = g_b[u] + rev_G[u][v][0]["cost"]
                    if v not in g_b or ng < g_b[v]:
                        g_b[v] = ng; cf_b[v] = u
                        heapq.heappush(pq_b, (ng + heuristic(graph, v, start), v))

        max_frontier = max(max_frontier, len(pq_f) + len(pq_b))
        if pq_f and pq_b and pq_f[0][0] + pq_b[0][0] >= best + 1e-9:
            break

    if meet is None:
        return None, nodes_explored, max_frontier, None

    fwd = reconstruct_path(cf_f, meet)
    bwd = reconstruct_path(cf_b, meet)
    return fwd + bwd[-2::-1], nodes_explored, max_frontier, meet


def ida_star(graph, start, goal):
   
    threshold = heuristic(graph, start, goal)
    path      = [start]
    stats     = {"explored": 0, "max_path": 1}

    def search(g_cost: float, thresh: float):
        node = path[-1]
        f    = g_cost + heuristic(graph, node, goal)
        stats["explored"]  += 1
        stats["max_path"]   = max(stats["max_path"], len(path))

        if f > thresh:
            return f
        if node == goal:
            return "FOUND"

        minimum = math.inf
        for n in graph.neighbors(node):
            if n in path:
                continue
            path.append(n)
            result = search(g_cost + graph[node][n][0]["cost"], thresh)
            if result == "FOUND":
                return "FOUND"
            if result < minimum:
                minimum = result
            path.pop()

        return minimum

    for _ in range(500):
        result = search(0, threshold)
        if result == "FOUND":
            return list(path), stats["explored"], stats["max_path"], None
        if result == math.inf:
            return None, stats["explored"], stats["max_path"], None
        threshold = result

    return None, stats["explored"], stats["max_path"], None



# def ida_star(graph, start, goal):
#     threshold = heuristic(graph, start, goal)
#     path = [start]

#     stats = {
#         "explored": 0,
#         "max_path": 1
#     }

#     def search(g_cost, thresh):
#         node = path[-1]

#         f = g_cost + heuristic(graph, node, goal)
#         stats["explored"] += 1

#         # update path length safely
#         if len(path) > stats["max_path"]:
#             stats["max_path"] = len(path)

#         if f > thresh:
#             return f

#         if node == goal:
#             return "FOUND"

#         minimum = math.inf

#         for n in graph.neighbors(node):
#             if n in path:
#                 continue

#             path.append(n)

#             res = search(g_cost + graph[node][n][0]["cost"], thresh)

#             if res == "FOUND":
#                 return "FOUND"   # safe: we WANT to keep path

#             if res < minimum:
#                 minimum = res

#             path.pop()  # always executed unless FOUND

#         return minimum

#     for _ in range(500):
#         res = search(0, threshold)

#         if res == "FOUND":
#             return list(path), stats["explored"], stats["max_path"], None

#         if res == math.inf:
#             return None, stats["explored"], stats["max_path"], None

#         threshold = res

#     return None, stats["explored"], stats["max_path"], None


