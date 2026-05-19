"""
=======================================================
  FUEL ALLOCATION CSP  —  Mirpur, Dhaka
  Constraint Optimization Problem (COP)
=======================================================
  Algorithms implemented :
    1. Pure Backtracking
    2. Backtracking + MRV + LCV + Forward Checking
       + AC-3 Preprocessing
       + Intelligent Backtracking (Conflict-Directed Backjumping)
    3. Local Search (Min-Conflicts Heuristic)

  NEW ADDITIONS vs original:
    • AC-3 Arc Consistency  — prunes domains globally before
      any search begins; visibly shrinks the constraint graph.
    • Conflict-Directed Backjumping (CBJ) — replaces chronological
      backtracking; on failure, jumps back to the deepest variable
      actually responsible for the conflict (conflict set), skipping
      irrelevant choices and reducing nodes explored drastically.

  CSP Formulation:
    Variables  : X_i = fuel allocated to station i  (i = 0..6)
    Domain     : D_i = {min_alloc, min+50, ..., max_alloc}  [discrete, finite]
    Constraints:
      C1 (global)  : sum(X_i) <= TOTAL_SUPPLY
      C2 (unary)   : X_i >= MIN_ALLOC
      C3 (unary)   : X_i <= MAX_ALLOC
      C4 (unary)   : X_i <= station_demand_i
      C5 (binary)  : |X_i - X_j| / max(X_i,X_j) <= 0.20  for adjacent stations
    Objective (COP): maximize sum(X_i / demand_i)
=======================================================
"""

import os
import time
import random
import copy
import collections
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

random.seed(42)
matplotlib.rcParams['figure.facecolor'] = '#0f1117'
matplotlib.rcParams['text.color'] = '#e8eaf0'
matplotlib.rcParams['axes.facecolor'] = '#1a1d27'
matplotlib.rcParams['axes.edgecolor'] = '#2e3250'
matplotlib.rcParams['axes.labelcolor'] = '#8b8fa8'
matplotlib.rcParams['xtick.color'] = '#8b8fa8'
matplotlib.rcParams['ytick.color'] = '#8b8fa8'
matplotlib.rcParams['grid.color'] = '#2e3250'


STATIONS = [
    {"id": 0, "name": "Mirpur-1\nPadma",    "x": 0.22, "y": 0.80, "demand": 900,  "priority": 3},
    {"id": 1, "name": "Mirpur-2\nMeghna",   "x": 0.55, "y": 0.82, "demand": 750,  "priority": 2},
    {"id": 2, "name": "Mirpur-10\nCircle",  "x": 0.38, "y": 0.60, "demand": 1100, "priority": 5},
    {"id": 3, "name": "Kazipara\nJamuna",   "x": 0.72, "y": 0.65, "demand": 650,  "priority": 2},
    {"id": 4, "name": "Shewrapara\nRupsa",  "x": 0.20, "y": 0.38, "demand": 850,  "priority": 3},
    {"id": 5, "name": "Pallabi\nBrahma",    "x": 0.58, "y": 0.35, "demand": 950,  "priority": 4},
    {"id": 6, "name": "Mirpur-14\nTista",   "x": 0.80, "y": 0.28, "demand": 700,  "priority": 2},
]

EDGES = [(0,2),(1,2),(2,3),(2,4),(2,5),(3,5),(4,5),(5,6),(1,3)]

N = len(STATIONS)

TOTAL_SUPPLY  = 5000
MAX_PER_STAT  = 1200
MIN_PER_STAT  = 200
DOMAIN_STEP   = 50
MAX_BINARY_DIFF = 0.20

COLORS = ["#4f8ef7","#7c5cf7","#fb923c","#34d399","#fbbf24","#f87171","#a78bfa"]

# Adjacency list (neighbours per station)
NEIGHBOURS = {i: set() for i in range(N)}
for (a, b) in EDGES:
    NEIGHBOURS[a].add(b)
    NEIGHBOURS[b].add(a)


# ─────────────────────────────────────────────────────────────────
#  DOMAIN & UTILITIES
# ─────────────────────────────────────────────────────────────────

def build_domain(min_v=MIN_PER_STAT, max_v=MAX_PER_STAT, step=DOMAIN_STEP):
    return list(range(min_v, max_v + 1, step))


def binary_ok(alloc, i, j):
    """C5: adjacent stations must not differ by more than 20%."""
    ai, aj = alloc[i], alloc[j]
    if ai < 0 or aj < 0:
        return True
    mx = max(ai, aj)
    if mx == 0:
        return True
    return abs(ai - aj) / mx <= MAX_BINARY_DIFF


def is_consistent(alloc, idx):
    """Check all constraints for a partially assigned allocation at index idx."""
    if alloc[idx] < 0:
        return True
    if alloc[idx] < MIN_PER_STAT or alloc[idx] > MAX_PER_STAT:
        return False
    if alloc[idx] > STATIONS[idx]["demand"]:
        return False
    for (a, b) in EDGES:
        if a == idx or b == idx:
            if not binary_ok(alloc, a, b):
                return False
    return True


def total_used(alloc):
    return sum(v for v in alloc if v >= 0)


def objective(alloc):
    """COP objective: maximize sum of satisfaction ratios."""
    return sum(min(alloc[i], STATIONS[i]["demand"]) / STATIONS[i]["demand"]
               for i in range(N))


def satisfaction_pct(alloc):
    return objective(alloc) / N * 100


def count_violations(alloc):
    v = 0
    if sum(alloc) > TOTAL_SUPPLY:
        v += 1
    for i in range(N):
        if alloc[i] < MIN_PER_STAT: v += 1
        if alloc[i] > MAX_PER_STAT: v += 1
        if alloc[i] > STATIONS[i]["demand"]: v += 1
    for (a, b) in EDGES:
        if not binary_ok(alloc, a, b):
            v += 1
    return v


# ─────────────────────────────────────────────────────────────────
#  AC-3 ARC CONSISTENCY  (NEW)
# ─────────────────────────────────────────────────────────────────

def ac3(domains):
    """
    AC-3 Arc Consistency algorithm.
    
    Enforces arc consistency across all binary constraints (C5).
    An arc (Xi, Xj) is consistent if for every value in Domain(Xi)
    there exists at least one value in Domain(Xj) that satisfies C5.
    
    Also enforces unary constraints C2/C3/C4 first (node consistency).
    
    Returns:
        domains  : pruned domain dict  {var_index: [values]}
        removed  : dict of {var: [removed_values]}  for reporting
        arc_log  : list of (arc, action) for visualisation
        success  : False if any domain is wiped out (problem unsolvable)
    """
    # ── Step 0: Node consistency (unary constraints) ──────────────
    # Prune any value that violates C2/C3/C4 before we even start AC-3.
    removed = {i: [] for i in range(N)}
    for i in range(N):
        cap = min(MAX_PER_STAT, STATIONS[i]["demand"])
        new_d = [v for v in domains[i] if MIN_PER_STAT <= v <= cap]
        removed[i] = [v for v in domains[i] if v not in new_d]
        domains[i] = new_d
        if not domains[i]:
            return domains, removed, [], False

    arc_log = []  # records pruning events for the visualisation

    # ── Step 1: Initialise queue with ALL directed arcs ───────────
    queue = collections.deque()
    for (a, b) in EDGES:
        queue.append((a, b))
        queue.append((b, a))

    # ── Step 2: Process arcs ──────────────────────────────────────
    while queue:
        (xi, xj) = queue.popleft()

        if _revise(domains, xi, xj, removed, arc_log):
            if not domains[xi]:
                # Domain wipe-out → problem has no solution with current state
                return domains, removed, arc_log, False
            # Re-add all arcs (xk, xi) for every neighbour xk ≠ xj
            for xk in NEIGHBOURS[xi]:
                if xk != xj:
                    queue.append((xk, xi))

    return domains, removed, arc_log, True


def _revise(domains, xi, xj, removed, arc_log):
    """
    Remove values from Domain(xi) that have no support in Domain(xj).
    Returns True if any value was removed.
    
    Support check for C5:
      value vi in D(xi) has support iff ∃ vj ∈ D(xj) s.t. binary_ok({xi:vi, xj:vj})
    """
    revised = False
    new_domain = []
    for vi in domains[xi]:
        # Check if there is ANY vj in D(xj) consistent with vi
        has_support = False
        for vj in domains[xj]:
            # Build a minimal mock-alloc for the binary_ok check
            mock = [-1] * N
            mock[xi] = vi
            mock[xj] = vj
            if binary_ok(mock, xi, xj):
                has_support = True
                break
        if has_support:
            new_domain.append(vi)
        else:
            removed[xi].append(vi)
            arc_log.append((xi, xj, vi, "pruned"))
            revised = True

    domains[xi] = new_domain
    return revised


def print_ac3_report(original_domains, pruned_domains, removed, arc_log):
    """Print a detailed AC-3 preprocessing report."""
    print("\n" + "="*55)
    print("  AC-3 ARC CONSISTENCY PREPROCESSING")
    print("="*55)
    total_before = sum(len(original_domains[i]) for i in range(N))
    total_after  = sum(len(pruned_domains[i])   for i in range(N))
    total_pruned = total_before - total_after
    print(f"  Total domain values before : {total_before}")
    print(f"  Total domain values after  : {total_after}")
    print(f"  Values pruned              : {total_pruned}  "
          f"({total_pruned/total_before*100:.1f}% reduction)")
    print(f"  Arcs processed (log)       : {len(arc_log)}")
    print()
    for i, s in enumerate(STATIONS):
        name = s["name"].replace("\n", " ")
        before = len(original_domains[i])
        after  = len(pruned_domains[i])
        pruned = before - after
        flag   = "  ← PRUNED" if pruned else ""
        lo = pruned_domains[i][0]  if pruned_domains[i] else "∅"
        hi = pruned_domains[i][-1] if pruned_domains[i] else "∅"
        print(f"    {name:22s}: {before:3d} → {after:3d} values "
              f"[{lo}..{hi}]{flag}")
        if removed[i]:
            print(f"       Removed: {removed[i][:6]}"
                  + (" …" if len(removed[i]) > 6 else ""))


# ─────────────────────────────────────────────────────────────────
#  ALGORITHM 1 — PURE BACKTRACKING  (unchanged)
# ─────────────────────────────────────────────────────────────────

def run_backtracking():
    """
    Pure Backtracking — no heuristics, no inference, no AC-3.
    Assigns variables sequentially, checks constraints after each
    assignment, backtracks chronologically on violation.
    """
    print("\n" + "="*55)
    print("  ALGORITHM 1: Pure Backtracking")
    print("="*55)

    domain = build_domain()
    alloc  = [-1] * N
    best   = {"alloc": None, "obj": -1}
    stats  = {"nodes": 0}
    NODE_LIMIT = 80_000

    def backtrack(idx, remaining):
        if stats["nodes"] >= NODE_LIMIT:
            return
        stats["nodes"] += 1

        if idx == N:
            obj = objective(alloc)
            if obj > best["obj"]:
                best["obj"]   = obj
                best["alloc"] = alloc[:]
            return

        cap = min(MAX_PER_STAT, STATIONS[idx]["demand"], remaining)
        for val in domain:
            if val > cap:
                break
            alloc[idx] = val
            if is_consistent(alloc, idx):
                backtrack(idx + 1, remaining - val)
            alloc[idx] = -1

    t0 = time.time()
    backtrack(0, TOTAL_SUPPLY)
    elapsed = (time.time() - t0) * 1000

    result = best["alloc"] or [MIN_PER_STAT] * N
    sat    = satisfaction_pct(result)
    viol   = count_violations(result)
    nodes  = stats["nodes"]

    print(f"  Nodes explored : {nodes:,}")
    print(f"  Time           : {elapsed:.1f} ms")
    print(f"  Satisfaction   : {sat:.1f}%")
    print(f"  Violations     : {viol}")
    for i, s in enumerate(STATIONS):
        bar = "█" * int(result[i] / 50)
        print(f"    {s['name'].replace(chr(10),' '):22s}  {result[i]:4d}L  {bar}")

    return {"alloc": result, "nodes": nodes, "time": elapsed,
            "sat": sat, "viol": viol, "name": "Backtracking"}


# ─────────────────────────────────────────────────────────────────
#  ALGORITHM 2 — MRV + LCV + FC + AC-3 + INTELLIGENT BACKTRACKING
# ─────────────────────────────────────────────────────────────────

def run_heuristic():
    """
    Combines four improvements over pure backtracking:

    1. AC-3 Preprocessing
       ─ Before search starts, run AC-3 to prune domains globally.
       ─ Every inconsistent value eliminated here saves exponential
         work inside the search tree.

    2. MRV  (Minimum Remaining Values)
       ─ Always expand the variable with the fewest legal values.
       ─ "Fail-first" — catches dead ends earliest.

    3. LCV  (Least Constraining Value)
       ─ Try values that eliminate the fewest options for neighbours
         first, leaving maximum flexibility for the rest.

    4. Forward Checking (FC)
       ─ After each assignment, immediately prune neighbour domains.
       ─ Detects domain-wipe-out (DWO) before recursing.

    5. Conflict-Directed Backjumping (CBJ)  ← NEW
       ─ Maintains a conflict set for each variable: the set of
         earlier variables whose assignments caused failures here.
       ─ On exhausting all values for X_i (dead end), instead of
         backtracking to X_{i-1} (chronological), jumps directly
         to the deepest variable in the conflict set.
       ─ Dramatically reduces nodes when the true cause of failure
         is far up the search tree.
    """
    print("\n" + "="*55)
    print("  ALGORITHM 2: AC-3 + MRV + LCV + FC + CBJ")
    print("="*55)

    # ── AC-3 preprocessing ────────────────────────────────────────
    base_domain = build_domain()
    original_domains = {i: base_domain[:] for i in range(N)}
    ac3_domains, removed, arc_log, feasible = ac3(
        {i: base_domain[:] for i in range(N)}
    )
    print_ac3_report(original_domains, ac3_domains, removed, arc_log)

    if not feasible:
        print("  ✗ AC-3 detected infeasibility — no solution exists!")
        fallback = [MIN_PER_STAT] * N
        return {"alloc": fallback, "nodes": 0, "time": 0,
                "sat": satisfaction_pct(fallback), "viol": count_violations(fallback),
                "name": "AC3+MRV+LCV+FC+CBJ",
                "ac3_removed": removed, "arc_log": arc_log,
                "original_domains": original_domains, "ac3_domains": ac3_domains}

    # Working copies of domains (modified during search, restored on backtrack)
    domains = {i: ac3_domains[i][:] for i in range(N)}
    alloc   = [-1] * N
    best    = {"alloc": None, "obj": -1}
    stats   = {"nodes": 0, "jumps": 0}
    NODE_LIMIT = 120_000

    # ── Variable ordering (fixed for CBJ index tracking) ──────────
    # We still use MRV dynamically but we need a stable index space.

    # def select_var(assigned):
    #     """MRV: pick unassigned variable with smallest remaining domain."""
    #     best_i, best_size = -1, float("inf")
    #     for i in range(N):
    #         if not assigned[i] and len(domains[i]) < best_size:
    #             best_size = len(domains[i])
    #             best_i = i
    #     return best_i
    
    def select_var(assigned):
        """
        MRV with Degree heuristic as tie-breaker.
        
        Primary   : MRV  — fewest remaining legal values (fail-first)
        Tie-break : Degree — most constraints on unassigned neighbours
        """
        best_i     = -1
        best_size  = float("inf")
        best_degree = -1

        for i in range(N):
            if assigned[i]:
                continue

            mrv = len(domains[i])

            # Degree = number of constraints with unassigned neighbours
            degree = sum(1 for nb in NEIGHBOURS[i] if not assigned[nb])

            if (mrv < best_size) or (mrv == best_size and degree > best_degree):
                best_size   = mrv
                best_degree = degree
                best_i      = i

        return best_i

    def order_values(idx, remaining):
        """LCV: sort by fewest neighbour values eliminated."""
        cap = min(MAX_PER_STAT, STATIONS[idx]["demand"], remaining)
        candidates = [v for v in domains[idx] if v <= cap]

        def lcv_score(val):
            eliminated = 0
            for nb in NEIGHBOURS[idx]:
                if alloc[nb] >= 0:
                    continue
                for nv in domains[nb]:
                    mock = alloc[:]
                    mock[idx], mock[nb] = val, nv
                    if not binary_ok(mock, idx, nb):
                        eliminated += 1
            return eliminated

        return sorted(candidates, key=lcv_score)

    def forward_check(idx, val, remaining):
        """
        After assigning val to X_idx, prune neighbour domains.
        Returns (saved_domains, conflict_vars) or (None, conflict_vars).
        conflict_vars: set of variable indices that caused DWO (for CBJ).
        """
        saved = {}
        conflict_vars = set()

        for nb in NEIGHBOURS[idx]:
            if alloc[nb] >= 0:
                continue
            saved[nb] = domains[nb][:]
            new_domain = []
            cap_nb = min(MAX_PER_STAT, STATIONS[nb]["demand"], remaining - val)
            for nv in domains[nb]:
                if nv > cap_nb:
                    continue
                mock = alloc[:]
                mock[idx], mock[nb] = val, nv
                if binary_ok(mock, idx, nb):
                    new_domain.append(nv)
            domains[nb] = new_domain
            if not domains[nb]:
                # DWO: record idx as a conflicting variable for CBJ
                conflict_vars.add(idx)
                # Restore and signal failure
                for k, v in saved.items():
                    domains[k] = v
                return None, conflict_vars

        return saved, conflict_vars

    # ── CBJ: conflict sets ─────────────────────────────────────────
    # conf_set[i] = set of variable indices that have caused a failure
    #               while X_i was being processed.
    conf_set = [set() for _ in range(N)]

    # assignment_order tracks which variable was assigned at each depth level
    assignment_order = []  # list of variable indices in order assigned

    JUMP_SENTINEL = object()   # sentinel returned to signal a backjump

    def backtrack(assigned, remaining):
        """
        CBJ-augmented backtracking.
        Returns None on normal completion, or (jump_target, conf) to signal
        a backjump to variable jump_target with merged conflict set conf.
        """
        if stats["nodes"] >= NODE_LIMIT:
            return None
        stats["nodes"] += 1

        if all(assigned):
            obj = objective(alloc)
            if obj > best["obj"]:
                best["obj"]   = obj
                best["alloc"] = alloc[:]
            return None

        idx  = select_var(assigned)
        vals = order_values(idx, remaining)

        assigned[idx] = True
        assignment_order.append(idx)

        exhausted_all = True

        for val in vals:
            alloc[idx] = val

            saved, fc_conflicts = forward_check(idx, val, remaining)

            if saved is None:
                # FC detected DWO — merge conflicts into conf_set[idx]
                conf_set[idx].update(fc_conflicts)
                alloc[idx] = -1
                continue  # try next value

            result = backtrack(assigned, remaining - val)

            # Restore forward-checked domains
            for k, v in saved.items():
                domains[k] = v

            if result is None:
                # Normal (non-jump) return — success or exhausted deeper vars
                alloc[idx] = -1
                exhausted_all = False
                # Keep searching for better objective (optimization)
                continue

            # A backjump was requested
            jump_target, jump_conf = result
            alloc[idx] = -1

            if jump_target != idx:
                # We are not the target — propagate jump upward
                conf_set[idx].update(jump_conf)
                assigned[idx] = False
                if assignment_order and assignment_order[-1] == idx:
                    assignment_order.pop()
                stats["jumps"] += 1
                return jump_target, conf_set[idx]
            else:
                # We ARE the target — resume search here with merged conf set
                conf_set[idx].update(jump_conf)
                # continue trying remaining values

        alloc[idx] = -1
        assigned[idx] = False
        if assignment_order and assignment_order[-1] == idx:
            assignment_order.pop()

        if exhausted_all and conf_set[idx]:
            # All values exhausted — backjump to deepest variable in conf set
            # that appears in the current assignment order
            jump_target = -1
            for v in conf_set[idx]:
                if v in assignment_order:
                    pos = assignment_order.index(v)
                    if pos > jump_target:
                        jump_target = pos
            if jump_target >= 0:
                actual_var = assignment_order[jump_target]
                # Merge conflict sets minus idx itself
                merged = conf_set[idx] - {idx}
                stats["jumps"] += 1
                return actual_var, merged

        return None

    t0 = time.time()
    backtrack([False] * N, TOTAL_SUPPLY)
    elapsed = (time.time() - t0) * 1000

    result = best["alloc"] or [MIN_PER_STAT] * N
    sat    = satisfaction_pct(result)
    viol   = count_violations(result)
    nodes  = stats["nodes"]
    jumps  = stats["jumps"]

    print(f"\n  Nodes explored     : {nodes:,}")
    print(f"  Backjumps (CBJ)    : {jumps:,}")
    print(f"  Time               : {elapsed:.1f} ms")
    print(f"  Satisfaction       : {sat:.1f}%")
    print(f"  Violations         : {viol}")
    for i, s in enumerate(STATIONS):
        bar = "█" * int(result[i] / 50)
        print(f"    {s['name'].replace(chr(10),' '):22s}  {result[i]:4d}L  {bar}")

    return {"alloc": result, "nodes": nodes, "time": elapsed,
            "sat": sat, "viol": viol, "name": "AC3+MRV+LCV+FC+CBJ",
            "ac3_removed": removed, "arc_log": arc_log,
            "original_domains": original_domains, "ac3_domains": ac3_domains,
            "jumps": jumps}


# ─────────────────────────────────────────────────────────────────
#  ALGORITHM 3 — LOCAL SEARCH (MIN-CONFLICTS)  (unchanged)
# ─────────────────────────────────────────────────────────────────

def run_local_search():
    """
    Local Search — Min-Conflicts Heuristic.
    Starts from a complete (possibly inconsistent) assignment.
    Repeatedly picks the most conflicted variable and reassigns it
    to the value minimising total violations.
    Random restarts escape local optima.
    Very fast — near-constant time regardless of problem size.
    """
    print("\n" + "="*55)
    print("  ALGORITHM 3: Local Search — Min-Conflicts")
    print("="*55)

    MAX_ITER   = 8000
    RESTART_AT = 1500

    def make_initial():
        a = []
        for s in STATIONS:
            v = min(MAX_PER_STAT, s["demand"])
            v = max(MIN_PER_STAT, round(v * random.uniform(0.5, 0.9) / DOMAIN_STEP) * DOMAIN_STEP)
            a.append(v)
        total = sum(a)
        if total > TOTAL_SUPPLY:
            scale = TOTAL_SUPPLY / total
            a = [max(MIN_PER_STAT, round(v * scale / DOMAIN_STEP) * DOMAIN_STEP) for v in a]
        return a

    def conflict_count(a, idx):
        c = 0
        if a[idx] < MIN_PER_STAT or a[idx] > MAX_PER_STAT: c += 2
        if a[idx] > STATIONS[idx]["demand"]: c += 1
        for nb in NEIGHBOURS[idx]:
            if not binary_ok(a, idx, nb):
                c += 1
        return c

    def total_conflicts(a):
        c = 0
        if sum(a) > TOTAL_SUPPLY: c += max(1, (sum(a) - TOTAL_SUPPLY) // 100)
        for i in range(N): c += conflict_count(a, i)
        return c

    def min_conflict_value(a, idx):
        cap = min(MAX_PER_STAT, STATIONS[idx]["demand"])
        best_val, best_c, best_obj = a[idx], float("inf"), -1
        for v in range(MIN_PER_STAT, cap + 1, DOMAIN_STEP):
            tmp = a[:]
            tmp[idx] = v
            excess = sum(tmp) - TOTAL_SUPPLY
            if excess > 0:
                others = [j for j in range(N) if j != idx and tmp[j] > MIN_PER_STAT]
                if others:
                    pick = random.choice(others)
                    tmp[pick] = max(MIN_PER_STAT,
                                    round((tmp[pick] - excess) / DOMAIN_STEP) * DOMAIN_STEP)
            c = total_conflicts(tmp)
            o = objective(tmp)
            if c < best_c or (c == best_c and o > best_obj):
                best_val, best_c, best_obj = v, c, o
        return best_val

    current    = make_initial()
    best_a     = current[:]
    best_obj   = objective(current)
    no_improve = 0
    iterations = 0

    t0 = time.time()

    while iterations < MAX_ITER:
        iterations += 1

        conf_scores = [conflict_count(current, i) for i in range(N)]
        max_conf    = max(conf_scores)

        if max_conf > 0 and random.random() < 0.85:
            conflicted = [i for i, c in enumerate(conf_scores) if c == max_conf]
            idx = random.choice(conflicted)
        else:
            idx = random.randint(0, N - 1)

        new_val      = min_conflict_value(current, idx)
        current[idx] = new_val

        excess = sum(current) - TOTAL_SUPPLY
        if excess > 0:
            others = sorted([j for j in range(N) if j != idx],
                            key=lambda j: -current[j])
            for j in others:
                cut = min(excess, current[j] - MIN_PER_STAT)
                current[j] -= round(cut / DOMAIN_STEP) * DOMAIN_STEP
                excess = sum(current) - TOTAL_SUPPLY
                if excess <= 0:
                    break

        obj = objective(current)
        if obj > best_obj:
            best_obj   = obj
            best_a     = current[:]
            no_improve = 0
        else:
            no_improve += 1

        if total_conflicts(current) == 0 and no_improve > 600:
            break

        if no_improve >= RESTART_AT:
            current    = make_initial()
            no_improve = 0

    elapsed = (time.time() - t0) * 1000
    sat     = satisfaction_pct(best_a)
    viol    = count_violations(best_a)

    print(f"  Iterations     : {iterations:,}")
    print(f"  Time           : {elapsed:.1f} ms")
    print(f"  Satisfaction   : {sat:.1f}%")
    print(f"  Violations     : {viol}")
    for i, s in enumerate(STATIONS):
        bar = "█" * int(best_a[i] / 50)
        print(f"    {s['name'].replace(chr(10),' '):22s}  {best_a[i]:4d}L  {bar}")

    return {"alloc": best_a, "nodes": iterations, "time": elapsed,
            "sat": sat, "viol": viol, "name": "Min-Conflicts LS"}


# ─────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────

def draw_map(ax, alloc, title):
    ax.set_facecolor("#1a1d27")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, color="#e8eaf0", fontsize=10, pad=6)
    ax.axis("off")

    for y in [0.15, 0.38, 0.55, 0.72]:
        ax.axhline(y, color="#2e3250", lw=5, zorder=0)
    for x in [0.2, 0.38, 0.58, 0.75]:
        ax.axvline(x, color="#2e3250", lw=5, zorder=0)

    for (a, b) in EDGES:
        sa, sb = STATIONS[a], STATIONS[b]
        ax.plot([sa["x"], sb["x"]], [sa["y"], sb["y"]],
            color="#7ab3f7", lw=2.0, linestyle="-", zorder=2,
            alpha=0.75)
    for i, s in enumerate(STATIONS):
        pct = alloc[i] / s["demand"] if alloc else 0
        col = "#34d399" if pct >= 0.85 else ("#fbbf24" if pct >= 0.55 else "#f87171")
        ax.scatter(s["x"], s["y"], s=350, color=col, zorder=4, edgecolors="#fff", linewidths=1.5)
        ax.scatter(s["x"], s["y"], s=900, color=col + "30", zorder=3)
        label = f"{s['name']}\n{alloc[i]}L" if alloc else s["name"]
        ax.text(s["x"], s["y"] - 0.08, label, ha="center", va="top",
                fontsize=7, color="#e8eaf0", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1d2799", edgecolor="none"))


def draw_ac3_graph(ax, original_domains, ac3_domains, removed):
    """
    Visualise how AC-3 changed the constraint graph.
    Left half = before, right half = after.
    Each station node shows domain size; pruned arcs highlighted.
    """
    ax.set_facecolor("#1a1d27")
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.05, 1.1)
    ax.set_title("AC-3: Domain Pruning Effect", color="#e8eaf0", fontsize=9, pad=6)
    ax.axis("off")

    # Offset: left = before, right = after
    offsets = [(-0.22, 0), (0.22, 0)]
    panel_labels = ["Before AC-3", "After AC-3"]
    panel_cols   = ["#4f8ef7aa", "#34d399aa"]

    for pi, (dx, dy) in enumerate(offsets):
        dom = original_domains if pi == 0 else ac3_domains
        label_col = panel_cols[pi]
        ax.text(0.5 + dx, 1.07, panel_labels[pi],
                ha="center", color=label_col, fontsize=8, fontweight="bold")

        # Draw edges
        for (a, b) in EDGES:
            sa, sb = STATIONS[a], STATIONS[b]
            x0, y0 = sa["x"] * 0.56 + 0.22 + dx, sa["y"] * 0.85 + dy
            x1, y1 = sb["x"] * 0.56 + 0.22 + dx, sb["y"] * 0.85 + dy
            ax.plot([x0, x1], [y0, y1], color="#7ab3f7", lw=2.0,
                linestyle="-", zorder=2, alpha=0.75)
        # Draw nodes
        for i, s in enumerate(STATIONS):
            x = s["x"] * 0.56 + 0.22 + dx
            y = s["y"] * 0.85 + dy
            dsize = len(dom[i])
            orig_size = len(original_domains[i])
            # Colour: red if domain shrank, green if unchanged
            if pi == 1 and dsize < orig_size:
                col = "#f87171"
            else:
                col = "#4f8ef7"
            ax.scatter(x, y, s=200 + dsize * 8, color=col, zorder=4,
                       edgecolors="#fff", linewidths=1.0, alpha=0.85)
            ax.text(x, y - 0.07, f"{s['name'].split(chr(10))[0]}\n|D|={dsize}",
                    ha="center", va="top", fontsize=6, color="#e8eaf0", zorder=5)

    # Divider
    ax.axvline(0.5, color="#2e3250", lw=1, linestyle=":")


def plot_results(results_list, r2_detail):
    """
    Full 4-row dashboard.
    Row 0: Maps
    Row 1: AC-3 graph + domain size comparison
    Row 2: Allocation bars per algorithm
    Row 3: Summary metrics
    """
    names   = [r["name"] for r in results_list]
    allocs  = [r["alloc"] for r in results_list]
    nodes   = [r["nodes"] for r in results_list]
    times   = [r["time"]  for r in results_list]
    sats    = [r["sat"]   for r in results_list]
    viols   = [r["viol"]  for r in results_list]

    algo_colors = ["#4f8ef7", "#34d399", "#fb923c"]

    fig = plt.figure(figsize=(22, 18), facecolor="#0f1117")
    fig.suptitle(
        "Fuel Allocation CSP — Mirpur, Dhaka\n",
        color="#e8eaf0", fontsize=14, fontweight="bold", y=0.99)

    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.40,
                          left=0.05, right=0.97, top=0.95, bottom=0.04)

    # ── Row 0: Maps ───────────────────────────────────────────────
    map_labels = ["Backtracking", "AC3+MRV+LCV+FC+CBJ", "Min-Conflicts LS"]
    for col, r in enumerate(results_list):
        ax = fig.add_subplot(gs[0, col])
        draw_map(ax, r["alloc"], map_labels[col])

    ax_leg = fig.add_subplot(gs[0, 3])
    ax_leg.set_facecolor("#1a1d27"); ax_leg.axis("off")
    ax_leg.set_title("Map Legend", color="#8b8fa8", fontsize=9)
    patches = [
        mpatches.Patch(color="#34d399", label="≥ 85% satisfied"),
        mpatches.Patch(color="#fbbf24", label="55–84% satisfied"),
        mpatches.Patch(color="#f87171", label="< 55% satisfied"),
        mlines.Line2D([], [], color="#4f8ef740", linestyle="--", label="Constraint edge"),
    ]
    ax_leg.legend(handles=patches, loc="center", fontsize=8,
                  facecolor="#1a1d27", edgecolor="#2e3250",
                  labelcolor="#e8eaf0", framealpha=0.9)

    # ── Row 1: AC-3 detail ────────────────────────────────────────
    # AC-3 before/after graph
    # ax_ac3 = fig.add_subplot(gs[1, 0:2])
    # draw_ac3_graph(ax_ac3,
    #                r2_detail["original_domains"],
    #                r2_detail["ac3_domains"],
    #                r2_detail["ac3_removed"])

    # Domain size per station: before vs after AC-3
    ax_dom = fig.add_subplot(gs[1, 2])
    ax_dom.set_facecolor("#1a1d27")
    snames = [s["name"].split("\n")[0] for s in STATIONS]
    x = np.arange(N)
    w = 0.35
    before_sizes = [len(r2_detail["original_domains"][i]) for i in range(N)]
    after_sizes  = [len(r2_detail["ac3_domains"][i])       for i in range(N)]
    ax_dom.bar(x - w/2, before_sizes, w, label="Before AC-3",
               color="#4f8ef7aa", edgecolor="#4f8ef7", linewidth=0.8)
    ax_dom.bar(x + w/2, after_sizes,  w, label="After AC-3",
               color="#34d399aa", edgecolor="#34d399", linewidth=0.8)
    ax_dom.set_title("Domain Size per Station (AC-3)", color="#e8eaf0", fontsize=9)
    ax_dom.set_xticks(x)
    ax_dom.set_xticklabels(snames, rotation=35, ha="right", fontsize=7)
    ax_dom.set_ylabel("Domain Size", fontsize=7)
    ax_dom.yaxis.set_tick_params(labelsize=7)
    ax_dom.grid(axis="y", alpha=0.3)
    ax_dom.spines[["top","right"]].set_visible(False)
    ax_dom.legend(fontsize=7, facecolor="#1a1d27", edgecolor="#2e3250", labelcolor="#e8eaf0")

    # CBJ jumps annotation panel
    ax_cbj = fig.add_subplot(gs[1, 3])
    ax_cbj.set_facecolor("#1a1d27"); ax_cbj.axis("off")
    ax_cbj.set_title("CBJ vs Chronological BT", color="#e8eaf0", fontsize=9)
    cbj_lines = [
        "Conflict-Directed Backjumping",
        "",
        "• Tracks a conflict set per var",
        "  (variables causing failure)",
        "",
        "• On dead-end: jumps directly",
        "  to deepest guilty variable",
        "  instead of just X_{i-1}",
        "",
        "• Skips irrelevant assignments",
        "  → far fewer nodes explored",
        "",
        f"  Backjumps this run: "
        f"{r2_detail.get('jumps', 0):,}",
    ]
    for li, line in enumerate(cbj_lines):
        col = "#34d399" if "Backjumps" in line else "#8b8fa8"
        ax_cbj.text(0.05, 0.92 - li * 0.068, line,
                    transform=ax_cbj.transAxes, fontsize=7.5,
                    color=col, va="top", family="monospace")

    # ── Row 2: Allocation bars per algorithm ──────────────────────
    demands = [s["demand"] for s in STATIONS]
    x = np.arange(N)
    w = 0.3
    for col, r in enumerate(results_list):
        ax = fig.add_subplot(gs[2, col])
        ax.bar(x - w/2, demands, w, label="Demand", color="#ffffff18",
               edgecolor="#ffffff40", linewidth=0.8)
        ax.bar(x + w/2, r["alloc"], w, label="Allocated",
               color=[COLORS[i] + "cc" for i in range(N)],
               edgecolor=[COLORS[i] for i in range(N)], linewidth=0.8)
        ax.set_title(r["name"], color="#e8eaf0", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([s["name"].split("\n")[0] for s in STATIONS],
                           rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Litres", fontsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        if col == 0:
            d_patch = mpatches.Patch(color="#ffffff30", label="Demand")
            a_patch = mpatches.Patch(color="#4f8ef7", label="Allocated")
            ax.legend(handles=[d_patch, a_patch], fontsize=7,
                      facecolor="#1a1d27", edgecolor="#2e3250", labelcolor="#e8eaf0")

    # Nodes compared
    ax = fig.add_subplot(gs[2, 3])
    bars = ax.bar(names, nodes, color=[c + "cc" for c in algo_colors],
                  edgecolor=algo_colors, linewidth=1.5)
    ax.set_title("Nodes / Iterations Explored", color="#e8eaf0", fontsize=9)
    ax.set_ylabel("Count", fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7, rotation=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, nodes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(nodes)*0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=7, color="#e8eaf0")

    # ── Row 3: Summary metrics ────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    bars = ax.bar(names, sats, color=[c + "cc" for c in algo_colors],
                  edgecolor=algo_colors, linewidth=1.5)
    ax.set_title("Satisfaction %", color="#e8eaf0", fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("%", fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7, rotation=10)
    ax.axhline(100, color="#ffffff30", lw=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, sats):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7, color="#e8eaf0")

    ax = fig.add_subplot(gs[3, 1])
    bars = ax.bar(names, times, color=[c + "cc" for c in algo_colors],
                  edgecolor=algo_colors, linewidth=1.5)
    ax.set_title("Execution Time (ms)", color="#e8eaf0", fontsize=9)
    ax.set_ylabel("ms", fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7, rotation=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#e8eaf0")

    ax = fig.add_subplot(gs[3, 2])
    bars = ax.bar(names, viols, color=[c + "cc" for c in algo_colors],
                  edgecolor=algo_colors, linewidth=1.5)
    ax.set_title("Constraint Violations", color="#e8eaf0", fontsize=9)
    ax.set_ylabel("Count", fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7, rotation=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, viols):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(val), ha="center", va="bottom", fontsize=8, color="#e8eaf0")

    # Summary table
    ax = fig.add_subplot(gs[3, 3])
    ax.set_facecolor("#1a1d27"); ax.axis("off")
    ax.set_title("Summary", color="#e8eaf0", fontsize=9)
    col_labels = ["Algorithm", "Nodes", "Time\n(ms)", "Sat%", "Viol"]
    row_data = [[r["name"], f"{r['nodes']:,}", f"{r['time']:.1f}",
                 f"{r['sat']:.1f}%", str(r["viol"])] for r in results_list]
    tbl = ax.table(cellText=row_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2e3250")
        if row == 0:
            cell.set_facecolor("#2e3250")
            cell.set_text_props(color="#8b8fa8", fontweight="bold")
        else:
            cell.set_facecolor("#1a1d27")
            cell.set_text_props(color="#e8eaf0")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    save_path = os.path.join(base_dir, "csp_results.png")

    plt.savefig(save_path,
                dpi=150,
                bbox_inches="tight",
                facecolor="#0f1117")
    print("\n  ✓  Plot saved → csp_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────
#  CONSTRAINT VERIFICATION REPORT
# ─────────────────────────────────────────────────────────────────

def print_constraint_report(r):
    alloc = r["alloc"]
    print(f"\n  {'─'*50}")
    print(f"  Constraint Report — {r['name']}")
    print(f"  {'─'*50}")
    total = sum(alloc)
    ok = "✓" if total <= TOTAL_SUPPLY else "✗"
    print(f"  {ok} C1 (supply)   : Σ = {total}L  ≤  {TOTAL_SUPPLY}L")
    for i, s in enumerate(STATIONS):
        n   = s["name"].replace("\n", " ")
        ok2 = "✓" if MIN_PER_STAT <= alloc[i] <= MAX_PER_STAT else "✗"
        ok3 = "✓" if alloc[i] <= s["demand"] else "✗"
        print(f"  {ok2} C2/C3 {n:20s}: {alloc[i]:4d}L ∈ [{MIN_PER_STAT},{MAX_PER_STAT}]")
        print(f"  {ok3} C4       {n:20s}: {alloc[i]:4d}L ≤ demand {s['demand']}L")
    for (a, b) in EDGES:
        ok4 = "✓" if binary_ok(alloc, a, b) else "✗"
        na  = STATIONS[a]["name"].replace("\n", " ")
        nb  = STATIONS[b]["name"].replace("\n", " ")
        diff = abs(alloc[a] - alloc[b])
        mx   = max(alloc[a], alloc[b])
        pct  = diff / mx * 100 if mx else 0
        print(f"  {ok4} C5 {na:17s}↔{nb:17s}: Δ={pct:.1f}% (≤20%)")


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FUEL ALLOCATION CSP  —  Mirpur, Dhaka             ║")
    print("║   Constraint Optimization Problem                    ║")
    print("║   + AC-3 Preprocessing  + CBJ                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n  Total supply  : {TOTAL_SUPPLY} L")
    print(f"  Stations      : {N}")
    print(f"  Domain step   : {DOMAIN_STEP} L")
    print(f"  Domain size   : {len(build_domain())} values per variable")
    print(f"  Constraint C5 : adjacent station diff ≤ {int(MAX_BINARY_DIFF*100)}%")

    r1 = run_backtracking()
    r2 = run_heuristic()       # AC-3 + MRV + LCV + FC + CBJ
    r3 = run_local_search()

    results = [r1, r2, r3]

    for r in results:
        print_constraint_report(r)

    print("\n" + "="*55)
    print("  ALGORITHM COMPARISON SUMMARY")
    print("="*55)
    print(f"  {'Algorithm':<25} {'Nodes':>8} {'Time(ms)':>10} {'Sat%':>8} {'Viol':>6}")
    print("  " + "─"*57)
    for r in results:
        print(f"  {r['name']:<25} {r['nodes']:>8,} {r['time']:>10.1f} "
              f"{r['sat']:>8.1f} {r['viol']:>6}")
    if "jumps" in r2:
        print(f"\n  CBJ backjumps (Algorithm 2): {r2['jumps']:,}")

    print("\n  AC-3 Effect:")
    total_before = sum(len(r2["original_domains"][i]) for i in range(N))
    total_after  = sum(len(r2["ac3_domains"][i])       for i in range(N))
    print(f"    Domain values before: {total_before}")
    print(f"    Domain values after : {total_after}  "
          f"({(total_before-total_after)/total_before*100:.1f}% pruned before search)")

    best = max(results, key=lambda x: x["sat"])
    print(f"\n  ★  Best algorithm: {best['name']}  ({best['sat']:.1f}% satisfaction)")
    print("\n  Note: Algorithm 2 shows the effect of AC-3 graph pruning")
    print("  and CBJ jump count.  Local Search is fastest by iteration count.")

    plot_results(results, r2)


if __name__ == "__main__":
    main()