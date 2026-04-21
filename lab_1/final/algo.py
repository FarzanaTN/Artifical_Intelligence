

import heapq
import math

from graph import heuristic, reconstruct_path


# ============================================================================
# UNINFORMED SEARCH
# ============================================================================

def bfs(graph, start, goal):
    """
    Breadth-First Search
    f(n): NONE — FIFO queue ordered by hop-count.
    g(n) = 0, h(n) = 0 effectively.
    Complete, not optimal for weighted graphs.
    """
    q   = [start]
    vis = {start}
    cf  = {start: None}
    count = 0

    while q:
        node = q.pop(0)
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                q.append(n)

    return None, count, None


def dfs(graph, start, goal):
    """
    Depth-First Search
    f(n): NONE — LIFO stack, dives deep ignoring cost.
    g(n) = 0, h(n) = 0 effectively.
    Not complete (may loop), not optimal.
    """
    stack = [start]
    vis   = {start}
    cf    = {start: None}
    count = 0

    while stack:
        node = stack.pop()
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                stack.append(n)

    return None, count, None


def dls(graph, start, goal, limit: int):
    """
    Depth-Limited Search
    f(n): NONE — DFS capped at hop-depth `limit`.
    g(n) = implicit hop depth, h(n) = 0.
    Complete only if solution depth ≤ limit.
    """
    stack = [(start, 0)]
    vis   = {start: 0}
    cf    = {start: None}
    count = 0

    while stack:
        node, depth = stack.pop()
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        if depth < limit:
            for n in graph.neighbors(node):
                if n not in vis:
                    vis[n]  = depth + 1
                    cf[n]   = node
                    stack.append((n, depth + 1))

    return None, count, None


def iddfs(graph, start, goal, max_depth: int = 60):
    """
    Iterative Deepening Depth-First Search
    f(n): NONE — runs DLS with increasing depth limit until goal found.
    Combines BFS completeness with DFS O(bd) memory.
    """
    total = 0
    for limit in range(1, max_depth + 1):
        path, count, _ = dls(graph, start, goal, limit)
        total += count
        if path is not None:
            return path, total, None
    return None, total, None


# ============================================================================
# INFORMED / COST-BASED SEARCH
# ============================================================================

def ucs(graph, start, goal):
    """
    Uniform Cost Search  (Dijkstra)
    f(n) = g(n)   [h(n) = 0]
    Priority queue ordered purely by accumulated risk-weighted cost.
    Optimal when all edge costs ≥ 0.
    """
    pq      = [(0, start)]
    g       = {start: 0}
    cf      = {start: None}
    visited = {}
    count   = 0

    while pq:
        cost, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited[node] = cost
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            nc = cost + graph[node][n][0]["cost"]
            if n not in g or nc < g[n]:
                g[n]  = nc
                cf[n] = node
                heapq.heappush(pq, (nc, n))

    return None, count, None


def greedy_bfs(graph, start, goal):
    """
    Greedy Best-First Search
    f(n) = h(n)   [g(n) = 0]
    Priority queue ordered purely by heuristic estimate.
    Fast but not optimal — ignores accumulated cost.
    """
    pq    = [(heuristic(graph, start, goal), start)]
    vis   = {start}
    cf    = {start: None}
    count = 0

    while pq:
        _, node = heapq.heappop(pq)
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            if n not in vis:
                vis.add(n)
                cf[n] = node
                heapq.heappush(pq, (heuristic(graph, n, goal), n))

    return None, count, None


def a_star(graph, start, goal):
    """
    A* Search
    f(n) = g(n) + h(n)

      g(n) : exact risk-weighted cost from start to n
      h(n) : dynamic heuristic (geodesic distance × local risk)
      f(n) : total estimated cost through n — open-set priority key

    Optimal when h(n) is admissible (never overestimates).
    At each step: pop min-f node, compute new_g = g(current) + edge_cost,
    push f = new_g + h(neighbour) if it improves known g(neighbour).
    """
    pq    = [(heuristic(graph, start, goal), start)]
    g     = {start: 0}
    cf    = {start: None}
    count = 0

    while pq:
        _, node = heapq.heappop(pq)
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]      # g(n)
            if n not in g or new_g < g[n]:
                g[n]  = new_g
                f     = new_g + heuristic(graph, n, goal)    # f = g + h
                cf[n] = node
                heapq.heappush(pq, (f, n))

    return None, count, None


def weighted_a_star(graph, start, goal, weight: float = 1.5):
    """
    Weighted A*
    f(n) = g(n) + W·h(n)

    W > 1 inflates the heuristic, making the search greedier:
    fewer nodes are expanded but the path may be up to W × optimal.
    W = 1 reduces to standard A*.
    """
    pq    = [(0, start)]
    g     = {start: 0}
    cf    = {start: None}
    count = 0

    while pq:
        _, node = heapq.heappop(pq)
        count += 1
        if node == goal:
            return reconstruct_path(cf, goal), count, None
        for n in graph.neighbors(node):
            new_g = g[node] + graph[node][n][0]["cost"]
            if n not in g or new_g < g[n]:
                g[n]  = new_g
                f     = new_g + weight * heuristic(graph, n, goal)
                cf[n] = node
                heapq.heappush(pq, (f, n))

    return None, count, None


# ============================================================================
# BIDIRECTIONAL SEARCH
# ============================================================================

def bidirectional_dijkstra(graph, start, goal):
    """
    Bidirectional Dijkstra (meet-in-the-middle UCS)
    Forward  frontier: f(n) = g_fwd(n)
    Backward frontier: f(n) = g_bwd(n)

    Two UCS frontiers expand simultaneously from start and goal.
    Terminates when the sum of the two frontier tops ≥ best known path cost.
    Returns the meeting node for map highlighting.
    """
    pq_f  = [(0, start)];  g_f = {start: 0};  cf_f = {start: None};  vis_f = {}
    rev_G = graph.reverse(copy=False)
    pq_b  = [(0, goal)];   g_b = {goal: 0};   cf_b = {goal: None};   vis_b = {}
    best  = math.inf
    meet  = None
    count = 0

    while pq_f or pq_b:
        if pq_f:
            c, u = heapq.heappop(pq_f)
            if u not in vis_f:
                vis_f[u] = c
                count += 1
                if u in vis_b and c + vis_b[u] < best:
                    best = c + vis_b[u]
                    meet = u
                for v in graph.neighbors(u):
                    nc = c + graph[u][v][0]["cost"]
                    if v not in g_f or nc < g_f[v]:
                        g_f[v] = nc
                        cf_f[v] = u
                        heapq.heappush(pq_f, (nc, v))

        if pq_b:
            c, u = heapq.heappop(pq_b)
            if u not in vis_b:
                vis_b[u] = c
                count += 1
                if u in vis_f and vis_f[u] + c < best:
                    best = vis_f[u] + c
                    meet = u
                for v in rev_G.neighbors(u):
                    nc = c + rev_G[u][v][0]["cost"]
                    if v not in g_b or nc < g_b[v]:
                        g_b[v] = nc
                        cf_b[v] = u
                        heapq.heappush(pq_b, (nc, v))

        if pq_f and pq_b and pq_f[0][0] + pq_b[0][0] >= best:
            break

    if meet is None:
        return None, count, None

    fwd = reconstruct_path(cf_f, meet)
    bwd = reconstruct_path(cf_b, meet)
    return fwd + bwd[-2::-1], count, meet


def bidirectional_astar(graph, start, goal):
    """
    Bidirectional A*
    Forward  frontier: f(n) = g_fwd(n) + h(n → goal)
    Backward frontier: f(n) = g_bwd(n) + h(n → start)

    Both frontiers use the dynamic heuristic pointing at their respective
    targets.  A*-guided meeting in the middle is faster than plain BiDi.
    Returns the meeting node for map highlighting.
    """
    pq_f  = [(heuristic(graph, start, goal), start)]
    g_f   = {start: 0};  cf_f = {start: None};  vis_f = {}
    rev_G = graph.reverse(copy=False)
    pq_b  = [(heuristic(graph, goal, start), goal)]
    g_b   = {goal: 0};   cf_b = {goal: None};   vis_b = {}
    best  = math.inf
    meet  = None
    count = 0

    while pq_f or pq_b:
        if pq_f:
            _, u = heapq.heappop(pq_f)
            if u not in vis_f:
                vis_f[u] = g_f[u]
                count += 1
                if u in vis_b:
                    cand = g_f[u] + g_b[u]
                    if cand < best:
                        best = cand
                        meet = u
                for v in graph.neighbors(u):
                    ng = g_f[u] + graph[u][v][0]["cost"]
                    if v not in g_f or ng < g_f[v]:
                        g_f[v] = ng
                        cf_f[v] = u
                        heapq.heappush(pq_f, (ng + heuristic(graph, v, goal), v))

        if pq_b:
            _, u = heapq.heappop(pq_b)
            if u not in vis_b:
                vis_b[u] = g_b[u]
                count += 1
                if u in vis_f:
                    cand = g_f.get(u, math.inf) + g_b[u]
                    if cand < best:
                        best = cand
                        meet = u
                for v in rev_G.neighbors(u):
                    ng = g_b[u] + rev_G[u][v][0]["cost"]
                    if v not in g_b or ng < g_b[v]:
                        g_b[v] = ng
                        cf_b[v] = u
                        heapq.heappush(pq_b, (ng + heuristic(graph, v, start), v))

        if pq_f and pq_b and pq_f[0][0] + pq_b[0][0] >= best + 1e-9:
            break

    if meet is None:
        return None, count, None

    fwd = reconstruct_path(cf_f, meet)
    bwd = reconstruct_path(cf_b, meet)
    return fwd + bwd[-2::-1], count, meet


def ida_star(graph, start, goal):
    """
    IDA* (Iterative Deepening A*)
    f(n) = g(n) + h(n)  evaluated at EVERY node.

    A* with iterative deepening over the f-cost threshold instead of memory.
    Each iteration expands all nodes with f(n) ≤ threshold; the threshold
    is then raised to the smallest f-value that exceeded it.
    Memory: O(path length) — only the current path stack is kept.
    """
    threshold = heuristic(graph, start, goal)
    path      = [start]
    count     = [0]

    def search(g_cost: float, thresh: float):
        node  = path[-1]
        f     = g_cost + heuristic(graph, node, goal)
        count[0] += 1

        if f > thresh:
            return f          # new minimum threshold candidate
        if node == goal:
            return "FOUND"

        minimum = math.inf
        for n in graph.neighbors(node):
            if n in path:     # avoid cycles on current path
                continue
            path.append(n)
            result = search(g_cost + graph[node][n][0]["cost"], thresh)
            if result == "FOUND":
                return "FOUND"
            if result < minimum:
                minimum = result
            path.pop()

        return minimum

    for _ in range(500):      # safety cap on iterations
        result = search(0, threshold)
        if result == "FOUND":
            return list(path), count[0], None
        if result == math.inf:
            return None, count[0], None
        threshold = result

    return None, count[0], None

