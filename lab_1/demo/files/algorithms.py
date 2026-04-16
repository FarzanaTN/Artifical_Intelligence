"""
algorithms.py
=============
12 search algorithms operating on CityGraph.

UNINFORMED:  BFS, DFS, DLS, IDDFS, UCS, BiDS
INFORMED:    Greedy, A*, Weighted A*, Bidirectional A*, IDA*, Beam

Each returns SearchResult dataclass.
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any

from cost_heuristic import cost, heuristic


@dataclass
class SearchResult:
    algorithm:     str
    path:          List[int]
    visited_order: List[int]
    nodes_visited: int
    total_cost:    float
    path_length:   int
    found:         bool
    extra:         Dict[str, Any] = field(default_factory=dict)

    def summary(self, graph) -> str:
        status = "✓ FOUND" if self.found else "✗ NOT FOUND"
        lines = [
            f"  Algorithm    : {self.algorithm}",
            f"  Status       : {status}",
            f"  Nodes visited: {self.nodes_visited}",
        ]
        if self.found:
            lines += [
                f"  Path edges   : {self.path_length}",
                f"  Total cost   : {self.total_cost:.4f}",
            ]
        for k, v in self.extra.items():
            lines.append(f"  {k:<14} : {v}")
        return "\n".join(lines)


def _recon(parent, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def _mk(algo, path, vis, g, found, **ex):
    return SearchResult(algo, path, vis, len(vis),
                        round(g, 5), max(0, len(path)-1), found, ex)


# ══════════════════════════════════════════════════════════════════════
# 1. BFS
# ══════════════════════════════════════════════════════════════════════
def bfs(graph, source, goal, weights):
    """Level-by-level FIFO. Finds min-hop path. Ignores edge cost."""
    q = deque([source])
    parent = {source: None}
    g_map = {source: 0.0}
    vis = []
    while q:
        n = q.popleft(); vis.append(n)
        if n == goal:
            return _mk("BFS", _recon(parent, goal), vis, g_map[goal], True)
        for nb, e in graph.neighbors(n):
            if nb not in parent:
                parent[nb] = n
                g_map[nb] = g_map[n] + cost(e, weights)
                q.append(nb)
    return _mk("BFS", [], vis, 0, False)


# ══════════════════════════════════════════════════════════════════════
# 2. DFS
# ══════════════════════════════════════════════════════════════════════
def dfs(graph, source, goal, weights):
    """LIFO stack. Not optimal. May find long path."""
    stack = [source]
    parent = {source: None}
    g_map = {source: 0.0}
    vis, seen = [], set()
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); vis.append(n)
        if n == goal:
            return _mk("DFS", _recon(parent, goal), vis, g_map[goal], True)
        for nb, e in reversed(graph.neighbors(n)):
            if nb not in seen:
                if nb not in parent:
                    parent[nb] = n
                    g_map[nb] = g_map[n] + cost(e, weights)
                stack.append(nb)
    return _mk("DFS", [], vis, 0, False)


# ══════════════════════════════════════════════════════════════════════
# 3. DLS — Depth-Limited Search
# ══════════════════════════════════════════════════════════════════════
def dls(graph, source, goal, weights, depth_limit=40):
    """DFS with hard depth cap."""
    vis = []

    def _r(n, pm, g, d, sp):
        vis.append(n)
        if n == goal: return _recon(pm, goal), g
        if d == 0:    return None, 0
        for nb, e in graph.neighbors(n):
            if nb not in sp:
                pm[nb] = n; sp.add(nb)
                r, c = _r(nb, pm, g + cost(e, weights), d-1, sp)
                if r: return r, c
                sp.discard(nb)
        return None, 0

    pm = {source: None}
    path, g = _r(source, pm, 0.0, depth_limit, {source})
    return _mk("DLS", path or [], vis, g, path is not None,
               depth_limit=depth_limit)


# ══════════════════════════════════════════════════════════════════════
# 4. IDDFS — Iterative Deepening DFS
# ══════════════════════════════════════════════════════════════════════
def iddfs(graph, source, goal, weights, max_depth=10):
    """Runs DLS with increasing limits 0,1,2,… Optimal (hops)."""
    all_vis = []
    for lim in range(max_depth+1):
        vis = []
        def _r(n, pm, g, d, sp):
            vis.append(n)
            if n == goal: return _recon(pm, goal), g
            if d == 0:    return None, 0
            for nb, e in graph.neighbors(n):
                if nb not in sp:
                    pm[nb] = n; sp.add(nb)
                    r, c = _r(nb, pm, g+cost(e,weights), d-1, sp)
                    if r: return r, c
                    sp.discard(nb)
            return None, 0
        pm = {source: None}
        path, g = _r(source, pm, 0.0, lim, {source})
        all_vis.extend(vis)
        if path:
            return _mk("IDDFS", path, all_vis, g, True, depth_found=lim)
    return _mk("IDDFS", [], all_vis, 0, False)


# ══════════════════════════════════════════════════════════════════════
# 5. UCS — Uniform Cost Search
# ══════════════════════════════════════════════════════════════════════
def ucs(graph, source, goal, weights):
    """Priority queue by g(n). Dijkstra variant. Optimal cost."""
    heap = [(0.0, source)]
    parent = {source: None}
    g_map = {source: 0.0}
    vis, seen = [], set()
    while heap:
        g, n = heapq.heappop(heap)
        if n in seen: continue
        seen.add(n); vis.append(n)
        if n == goal:
            return _mk("UCS", _recon(parent, goal), vis, g, True)
        for nb, e in graph.neighbors(n):
            ng = g + cost(e, weights)
            if nb not in g_map or ng < g_map[nb]:
                g_map[nb] = ng; parent[nb] = n
                heapq.heappush(heap, (ng, nb))
    return _mk("UCS", [], vis, 0, False)


# ══════════════════════════════════════════════════════════════════════
# 6. BiDS — Bidirectional BFS
# ══════════════════════════════════════════════════════════════════════
def bids(graph, source, goal, weights):
    """Two simultaneous BFS frontiers. Meets in the middle."""
    if source == goal:
        return _mk("BiDS", [source], [source], 0, True)

    fq = deque([source]); bq = deque([goal])
    fp = {source: None};  bp = {goal: None}
    fc = {source: 0.0};   bc = {goal: 0.0}
    vis = []; meet = None

    while fq or bq:
        if fq:
            n = fq.popleft(); vis.append(n)
            for nb, e in graph.neighbors(n):
                if nb not in fp:
                    fp[nb] = n; fc[nb] = fc[n]+cost(e,weights); fq.append(nb)
            if n in bp: meet=n; break
        if bq:
            n = bq.popleft(); vis.append(n)
            for nb, e in graph.neighbors(n):
                if nb not in bp:
                    bp[nb] = n; bc[nb] = bc[n]+cost(e,weights); bq.append(nb)
            if n in fp: meet=n; break

    if meet is None:
        return _mk("BiDS", [], vis, 0, False)

    fpath = _recon(fp, meet)
    bpath = []
    cur = meet
    while cur is not None:
        bpath.append(cur); cur = bp[cur]
    full = fpath + bpath[1:]
    return _mk("BiDS", full, vis, fc.get(meet,0)+bc.get(meet,0), True,
               meet_node=str(meet))


# ══════════════════════════════════════════════════════════════════════
# 7. Greedy Best-First
# ══════════════════════════════════════════════════════════════════════
def greedy(graph, source, goal, weights):
    """Priority queue by h(n) only. Fast but not optimal."""
    heap = [(heuristic(graph,source,goal,weights), source)]
    parent = {source: None}; g_map = {source: 0.0}
    vis, seen = [], set()
    while heap:
        _, n = heapq.heappop(heap)
        if n in seen: continue
        seen.add(n); vis.append(n)
        if n == goal:
            return _mk("Greedy", _recon(parent,goal), vis, g_map[goal], True)
        for nb, e in graph.neighbors(n):
            if nb not in seen:
                ng = g_map[n]+cost(e,weights)
                if nb not in g_map or ng < g_map[nb]:
                    g_map[nb]=ng; parent[nb]=n
                heapq.heappush(heap,(heuristic(graph,nb,goal,weights),nb))
    return _mk("Greedy", [], vis, 0, False)


# ══════════════════════════════════════════════════════════════════════
# 8. A*
# ══════════════════════════════════════════════════════════════════════
def astar(graph, source, goal, weights):
    """f(n)=g(n)+h(n). Optimal + complete."""
    h0 = heuristic(graph,source,goal,weights)
    heap = [(h0,0.0,source)]
    parent={source:None}; g_map={source:0.0}
    vis,seen=[],set()
    while heap:
        _,g,n = heapq.heappop(heap)
        if n in seen: continue
        seen.add(n); vis.append(n)
        if n==goal:
            return _mk("A*",_recon(parent,goal),vis,g,True)
        for nb,e in graph.neighbors(n):
            ng=g+cost(e,weights)
            if nb not in g_map or ng<g_map[nb]:
                g_map[nb]=ng; parent[nb]=n
                heapq.heappush(heap,(ng+heuristic(graph,nb,goal,weights),ng,nb))
    return _mk("A*",[],vis,0,False)


# ══════════════════════════════════════════════════════════════════════
# 9. Weighted A*
# ══════════════════════════════════════════════════════════════════════
def wastar(graph, source, goal, weights, epsilon=2.5):
    """f=g+ε·h. Faster than A*, within ε× optimal."""
    heap=[(heuristic(graph,source,goal,weights)*epsilon,0.0,source)]
    parent={source:None}; g_map={source:0.0}
    vis,seen=[],set()
    while heap:
        _,g,n=heapq.heappop(heap)
        if n in seen: continue
        seen.add(n); vis.append(n)
        if n==goal:
            return _mk("Weighted A*",_recon(parent,goal),vis,g,True,epsilon=epsilon)
        for nb,e in graph.neighbors(n):
            ng=g+cost(e,weights)
            if nb not in g_map or ng<g_map[nb]:
                g_map[nb]=ng; parent[nb]=n
                heapq.heappush(heap,(ng+epsilon*heuristic(graph,nb,goal,weights),ng,nb))
    return _mk("Weighted A*",[],vis,0,False,epsilon=epsilon)


# ══════════════════════════════════════════════════════════════════════
# 10. Bidirectional A*
# ══════════════════════════════════════════════════════════════════════
def bi_astar(graph, source, goal, weights):
    """Two A* frontiers. Faster on large real-world graphs."""
    if source==goal:
        return _mk("BiA*",[source],[source],0,True)

    fh=[(heuristic(graph,source,goal,weights),0.0,source)]
    bh=[(heuristic(graph,goal,source,weights),0.0,goal)]
    fg={source:0.0}; bg={goal:0.0}
    fp={source:None};bp={goal:None}
    fc,bc=set(),set()
    vis=[]; best=math.inf; meet=None

    while fh or bh:
        if fh:
            _,g,n=heapq.heappop(fh)
            if n not in fc:
                fc.add(n); vis.append(n)
                if n in bc and fg[n]+bg[n]<best:
                    best=fg[n]+bg[n]; meet=n
                for nb,e in graph.neighbors(n):
                    ng=g+cost(e,weights)
                    if nb not in fg or ng<fg[nb]:
                        fg[nb]=ng; fp[nb]=n
                        heapq.heappush(fh,(ng+heuristic(graph,nb,goal,weights),ng,nb))
        if bh:
            _,g,n=heapq.heappop(bh)
            if n not in bc:
                bc.add(n); vis.append(n)
                if n in fc and fg[n]+bg[n]<best:
                    best=fg[n]+bg[n]; meet=n
                for nb,e in graph.neighbors(n):
                    ng=g+cost(e,weights)
                    if nb not in bg or ng<bg[nb]:
                        bg[nb]=ng; bp[nb]=n
                        heapq.heappush(bh,(ng+heuristic(graph,nb,source,weights),ng,nb))
        ft=fh[0][0] if fh else math.inf
        bt=bh[0][0] if bh else math.inf
        if meet and ft+bt>=best: break

    if meet is None:
        return _mk("BiA*",[],vis,0,False)
    fpath=_recon(fp,meet)
    bpart=[]
    cur=meet
    while cur is not None: bpart.append(cur); cur=bp[cur]
    return _mk("BiA*",fpath+bpart[1:],vis,best,True,meet_node=str(meet))


# ══════════════════════════════════════════════════════════════════════
# 11. IDA*
# ══════════════════════════════════════════════════════════════════════
def idastar(graph, source, goal, weights):
    """Memory-efficient A*. Threshold increases each iteration."""
    threshold=heuristic(graph,source,goal,weights)
    path=[source]; all_vis=[]

    def _s(g,bound):
        n=path[-1]; all_vis.append(n)
        f=g+heuristic(graph,n,goal,weights)
        if f>bound: return f,False
        if n==goal: return g,True
        mn=math.inf
        for nb,e in graph.neighbors(n):
            if nb not in path:
                path.append(nb)
                t,found=_s(g+cost(e,weights),bound)
                if found: return t,True
                mn=min(mn,t); path.pop()
        return mn,False

    for i in range(200):
        r,found=_s(0.0,threshold)
        if found:
            return _mk("IDA*",list(path),all_vis,r,True,iterations=i+1)
        if r==math.inf: break
        threshold=r
    return _mk("IDA*",[],all_vis,0,False)


# ══════════════════════════════════════════════════════════════════════
# 12. Beam Search
# ══════════════════════════════════════════════════════════════════════
def beam_search(graph, source, goal, weights, beam_width=4):
    """BFS keeping only top-k nodes by h(n) per level."""
    beam=[(heuristic(graph,source,goal,weights),source,{source:None},{source:0.0})]
    vis=[]; seen=set()
    while beam:
        nxt=[]
        for _,n,pm,gm in beam:
            if n in seen: continue
            seen.add(n); vis.append(n)
            if n==goal:
                return _mk("Beam",_recon(pm,goal),vis,gm[goal],True,beam_width=beam_width)
            for nb,e in graph.neighbors(n):
                if nb not in seen:
                    ng=gm[n]+cost(e,weights)
                    npm=dict(pm); npm[nb]=n
                    ngm=dict(gm); ngm[nb]=ng
                    nxt.append((heuristic(graph,nb,goal,weights),nb,npm,ngm))
        nxt.sort(key=lambda x:x[0])
        beam=nxt[:beam_width]
    return _mk("Beam",[],vis,0,False,beam_width=beam_width)


# ─────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────
def run_all(graph, source, goal, weights):
    print("  Running BFS..."); r1=bfs(graph,source,goal,weights)
    print("  Running DFS..."); r2=dfs(graph,source,goal,weights)
    print("  Running DLS..."); r3=dls(graph,source,goal,weights,depth_limit=12)
    print("  Running IDDFS..."); r4=iddfs(graph,source,goal,weights)
    print("  Running UCS..."); r5=ucs(graph,source,goal,weights)
    print("  Running BiDS..."); r6=bids(graph,source,goal,weights)
    print("  Running Greedy..."); r7=greedy(graph,source,goal,weights)
    print("  Running A*..."); r8=astar(graph,source,goal,weights)
    print("  Running Weighted A*..."); r9=wastar(graph,source,goal,weights,epsilon=2.5)
    print("  Running BiA*..."); r10=bi_astar(graph,source,goal,weights)
    print("  Running IDA*..."); r11=idastar(graph,source,goal,weights)
    print("  Running Beam..."); r12=beam_search(graph,source,goal,weights,beam_width=4)
    return [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12]
