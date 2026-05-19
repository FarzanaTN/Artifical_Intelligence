"""
=======================================================
  FUEL ALLOCATION CSP  —  Mirpur, Dhaka
  Constraint Optimization Problem (COP)
=======================================================
  Algorithms implemented :
    1. Pure Backtracking
    2. Backtracking + MRV + LCV + Forward Checking
    3. Local Search (Min-Conflicts Heuristic)
   

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

import time
import random
import copy
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
#  ALGORITHM 1 — PURE BACKTRACKING
# ─────────────────────────────────────────────────────────────────

def run_backtracking():
    """
     "Backtracking search for solving CSP"
    - Assigns variables sequentially (no smart ordering)
    - Checks constraints after each assignment
    - Backtracks on violation
    - No inference (no forward checking)
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
        print(f"    {s['name'].replace(chr(10),' '):20s}  {result[i]:4d}L  {bar}")

    return {"alloc": result, "nodes": nodes, "time": elapsed,
            "sat": sat, "viol": viol, "name": "Backtracking"}


# ─────────────────────────────────────────────────────────────────
#  ALGORITHM 2 — BACKTRACKING + MRV + LCV + FORWARD CHECKING
# ─────────────────────────────────────────────────────────────────

def run_heuristic():
    """
      Heuristic 1 → MRV (minimum remaining values / fail-fast)
      Heuristic 3 → LCV (least constraining value)
      Inference   → Forward Checking (interleaving search & inference)
    """
    print("\n" + "="*55)
    print("  ALGORITHM 2: MRV + LCV + Forward Checking")
    print("="*55)

    base_domain = build_domain()
    domains = [base_domain[:] for _ in range(N)]
    alloc   = [-1] * N
    best    = {"alloc": None, "obj": -1}
    stats   = {"nodes": 0}
    NODE_LIMIT = 120_000

    def select_var(assigned):
        """MRV: pick unassigned variable with fewest remaining legal values."""
        best_i, best_size = -1, float("inf")
        for i in range(N):
            if not assigned[i] and len(domains[i]) < best_size:
                best_size = len(domains[i])
                best_i = i
        return best_i

    def order_values(idx, remaining):
        """LCV: sort values by how few choices they eliminate in neighbours."""
        cap = min(MAX_PER_STAT, STATIONS[idx]["demand"], remaining)
        candidates = [v for v in domains[idx] if v <= cap]

        def lcv_score(val):
            """Count how many neighbour values would be eliminated."""
            eliminated = 0
            for (a, b) in EDGES:
                nb = b if a == idx else (a if b == idx else -1)
                if nb == -1 or alloc[nb] >= 0:
                    continue
                for nv in domains[nb]:
                    tmp = alloc[:]
                    tmp[idx], tmp[nb] = val, nv
                    if not binary_ok(tmp, idx, nb):
                        eliminated += 1
            return eliminated

        return sorted(candidates, key=lcv_score)

    def forward_check(idx, val, remaining):
        """
        Forward Checking: after assigning val to X_idx,
        remove inconsistent values from unassigned neighbours.
        Returns saved domains dict (for undo), or None if any domain empties.
        """
        saved = {}
        for (a, b) in EDGES:
            nb = b if a == idx else (a if b == idx else -1)
            if nb == -1 or alloc[nb] >= 0:
                continue
            saved[nb] = domains[nb][:]
            new_domain = []
            cap_nb = min(MAX_PER_STAT, STATIONS[nb]["demand"], remaining - val)
            for nv in domains[nb]:
                if nv > cap_nb:
                    continue
                tmp = alloc[:]
                tmp[idx], tmp[nb] = val, nv
                if binary_ok(tmp, idx, nb):
                    new_domain.append(nv)
            domains[nb] = new_domain
            if not domains[nb]:
                # Restore and signal failure (DWO — domain wipe-out)
                for k, v in saved.items():
                    domains[k] = v
                return None
        return saved

    def backtrack(assigned, remaining):
        if stats["nodes"] >= NODE_LIMIT:
            return
        stats["nodes"] += 1

        if all(assigned):
            obj = objective(alloc)
            if obj > best["obj"]:
                best["obj"]   = obj
                best["alloc"] = alloc[:]
            return

        idx  = select_var(assigned)
        vals = order_values(idx, remaining)

        for val in vals:
            alloc[idx]    = val
            assigned[idx] = True

            saved = forward_check(idx, val, remaining)
            if saved is not None:
                backtrack(assigned, remaining - val)
                for k, v in saved.items():
                    domains[k] = v

            alloc[idx]    = -1
            assigned[idx] = False

    t0 = time.time()
    backtrack([False] * N, TOTAL_SUPPLY)
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
        print(f"    {s['name'].replace(chr(10),' '):20s}  {result[i]:4d}L  {bar}")

    return {"alloc": result, "nodes": nodes, "time": elapsed,
            "sat": sat, "viol": viol, "name": "MRV+LCV+FC"}


# ─────────────────────────────────────────────────────────────────
#  ALGORITHM 3 — LOCAL SEARCH (MIN-CONFLICTS)
# ─────────────────────────────────────────────────────────────────

def run_local_search():
    """
     "Local Search for CSPs — Min-conflict Heuristic"
    - Starts from a complete (possibly inconsistent) assignment
    - Repeatedly picks a conflicted variable
    - Reassigns it to the value minimising constraint violations
    - Random restarts to escape local optima
    - Very fast — near-constant time regardless of problem size
    """
    print("\n" + "="*55)
    print("  ALGORITHM 3: Local Search — Min-Conflicts")
    print("="*55)

    MAX_ITER   = 8000
    RESTART_AT = 1500

    def make_initial():
        """Greedy initial complete assignment."""
        a = []
        for s in STATIONS:
            v = min(MAX_PER_STAT, s["demand"])
            v = max(MIN_PER_STAT, round(v * random.uniform(0.5, 0.9) / DOMAIN_STEP) * DOMAIN_STEP)
            a.append(v)
        # Scale down if over supply
        total = sum(a)
        if total > TOTAL_SUPPLY:
            scale = TOTAL_SUPPLY / total
            a = [max(MIN_PER_STAT, round(v * scale / DOMAIN_STEP) * DOMAIN_STEP) for v in a]
        return a

    def conflict_count(a, idx):
        """How many constraints does station idx currently violate?"""
        c = 0
        if a[idx] < MIN_PER_STAT or a[idx] > MAX_PER_STAT: c += 2
        if a[idx] > STATIONS[idx]["demand"]: c += 1
        for (x, y) in EDGES:
            nb = y if x == idx else (x if y == idx else -1)
            if nb != -1 and not binary_ok(a, idx, nb):
                c += 1
        return c

    def total_conflicts(a):
        c = 0
        if sum(a) > TOTAL_SUPPLY: c += max(1, (sum(a) - TOTAL_SUPPLY) // 100)
        for i in range(N): c += conflict_count(a, i)
        return c

    def min_conflict_value(a, idx):
        """Return the value for X_idx that minimises total conflicts."""
        cap = min(MAX_PER_STAT, STATIONS[idx]["demand"])
        best_val, best_c, best_obj = a[idx], float("inf"), -1
        for v in range(MIN_PER_STAT, cap + 1, DOMAIN_STEP):
            tmp = a[:]
            tmp[idx] = v
            # Honour supply constraint by adjusting another station
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

    current   = make_initial()
    best_a    = current[:]
    best_obj  = objective(current)
    no_improve = 0
    iterations = 0

    t0 = time.time()

    while iterations < MAX_ITER:
        iterations += 1

        # Pick a conflicted variable (min-conflicts style from slide)
        conf_scores = [conflict_count(current, i) for i in range(N)]
        max_conf    = max(conf_scores)

        if max_conf > 0 and random.random() < 0.85:
            conflicted = [i for i, c in enumerate(conf_scores) if c == max_conf]
            idx = random.choice(conflicted)
        else:
            idx = random.randint(0, N - 1)

        new_val        = min_conflict_value(current, idx)
        current[idx]   = new_val

        # Enforce supply constraint (reduce largest other station if needed)
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

        # Early stop: no violations and no improvement possible
        if total_conflicts(current) == 0 and no_improve > 600:
            break

        # Random restart to escape local optimum
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
        print(f"    {s['name'].replace(chr(10),' '):20s}  {best_a[i]:4d}L  {bar}")

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

    # Road grid
    for y in [0.15, 0.38, 0.55, 0.72]:
        ax.axhline(y, color="#2e3250", lw=5, zorder=0)
    for x in [0.2, 0.38, 0.58, 0.75]:
        ax.axvline(x, color="#2e3250", lw=5, zorder=0)

    # Edges
    for (a, b) in EDGES:
        sa, sb = STATIONS[a], STATIONS[b]
        ax.plot([sa["x"], sb["x"]], [sa["y"], sb["y"]],
                color="#4f8ef740", lw=1.2, linestyle="--", zorder=1)

    # Stations
    for i, s in enumerate(STATIONS):
        pct = alloc[i] / s["demand"] if alloc else 0
        col = "#34d399" if pct >= 0.85 else ("#fbbf24" if pct >= 0.55 else "#f87171")

        ax.scatter(s["x"], s["y"], s=350, color=col, zorder=4, edgecolors="#fff", linewidths=1.5)
        ax.scatter(s["x"], s["y"], s=900, color=col + "30", zorder=3)

        label = f"{s['name']}\n{alloc[i]}L" if alloc else s["name"]
        ax.text(s["x"], s["y"] - 0.08, label, ha="center", va="top",
                fontsize=7, color="#e8eaf0", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1d2799", edgecolor="none"))


def plot_results(results_list):
    names   = [r["name"] for r in results_list]
    allocs  = [r["alloc"] for r in results_list]
    nodes   = [r["nodes"] for r in results_list]
    times   = [r["time"]  for r in results_list]
    sats    = [r["sat"]   for r in results_list]
    viols   = [r["viol"]  for r in results_list]
    snames  = [s["name"].replace("\n", " ") for s in STATIONS]
    demands = [s["demand"] for s in STATIONS]

    algo_colors = ["#4f8ef7", "#34d399", "#fb923c"]

    fig = plt.figure(figsize=(20, 13), facecolor="#0f1117")
    fig.suptitle("Fuel Allocation CSP — Mirpur, Dhaka\nAlgorithm Comparison",
                 color="#e8eaf0", fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 4, hspace=0.48, wspace=0.38,
                          left=0.05, right=0.97, top=0.93, bottom=0.05)

    # ── Row 0: Maps ───────────────────────────────────────────────
    map_labels = ["Backtracking", "MRV+LCV+FC", "Min-Conflicts LS"]
    for col, r in enumerate(results_list):
        ax = fig.add_subplot(gs[0, col])
        draw_map(ax, r["alloc"], map_labels[col])

    # Legend for maps
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

    # ── Row 1 col 0-2: Allocation bars per algorithm ──────────────
    x = np.arange(N)
    w = 0.3
    for col, r in enumerate(results_list):
        ax = fig.add_subplot(gs[1, col])
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

    # ── Row 1 col 3: Nodes explored comparison ────────────────────
    ax = fig.add_subplot(gs[1, 3])
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

    # ── Row 2: Summary metrics ────────────────────────────────────
    # Satisfaction %
    ax = fig.add_subplot(gs[2, 0])
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

    # Time
    ax = fig.add_subplot(gs[2, 1])
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

    # Violations
    ax = fig.add_subplot(gs[2, 2])
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
    ax = fig.add_subplot(gs[2, 3])
    ax.set_facecolor("#1a1d27"); ax.axis("off")
    ax.set_title("Summary", color="#e8eaf0", fontsize=9)
    col_labels = ["Algorithm", "Nodes", "Time\n(ms)", "Sat%", "Viol"]
    row_data = [[r["name"], f"{r['nodes']:,}", f"{r['time']:.1f}",
                 f"{r['sat']:.1f}%", str(r["viol"])] for r in results_list]
    tbl = ax.table(cellText=row_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2e3250")
        if row == 0:
            cell.set_facecolor("#2e3250")
            cell.set_text_props(color="#8b8fa8", fontweight="bold")
        else:
            cell.set_facecolor("#1a1d27")
            cell.set_text_props(color="#e8eaf0")

    plt.savefig("csp_results.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    print("\n  ✓  Plot saved → csp_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────
#  CONSTRAINT VERIFICATION REPORT
# ─────────────────────────────────────────────────────────────────

def print_constraint_report(r):
    alloc = r["alloc"]
    print(f"\n  {'─'*48}")
    print(f"  Constraint Report — {r['name']}")
    print(f"  {'─'*48}")
    total = sum(alloc)
    ok = "✓" if total <= TOTAL_SUPPLY else "✗"
    print(f"  {ok} C1 (supply)   : Σ = {total}L  ≤  {TOTAL_SUPPLY}L")
    for i, s in enumerate(STATIONS):
        n = s["name"].replace("\n", " ")
        ok2 = "✓" if MIN_PER_STAT <= alloc[i] <= MAX_PER_STAT else "✗"
        ok3 = "✓" if alloc[i] <= s["demand"] else "✗"
        print(f"  {ok2} C2/C3 {n:18s}: {alloc[i]:4d}L ∈ [{MIN_PER_STAT},{MAX_PER_STAT}]")
        print(f"  {ok3} C4       {n:18s}: {alloc[i]:4d}L ≤ demand {s['demand']}L")
    for (a, b) in EDGES:
        ok4 = "✓" if binary_ok(alloc, a, b) else "✗"
        na = STATIONS[a]["name"].replace("\n", " ")
        nb = STATIONS[b]["name"].replace("\n", " ")
        diff = abs(alloc[a] - alloc[b])
        mx   = max(alloc[a], alloc[b])
        pct  = diff / mx * 100 if mx else 0
        print(f"  {ok4} C5 {na:15s}↔{nb:15s}: Δ={pct:.1f}% (≤20%)")


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FUEL ALLOCATION CSP  —  Mirpur, Dhaka             ║")
    print("║   Constraint Optimization Problem                    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n  Total supply  : {TOTAL_SUPPLY} L")
    print(f"  Stations      : {N}")
    print(f"  Domain step   : {DOMAIN_STEP} L")
    print(f"  Domain size   : {len(build_domain())} values per variable")
    print(f"  Constraint C5 : adjacent station diff ≤ {int(MAX_BINARY_DIFF*100)}%")

    r1 = run_backtracking()
    r2 = run_heuristic()
    r3 = run_local_search()

    results = [r1, r2, r3]

    # Constraint reports
    for r in results:
        print_constraint_report(r)

    # Comparison summary
    print("\n" + "="*55)
    print("  ALGORITHM COMPARISON SUMMARY")
    print("="*55)
    print(f"  {'Algorithm':<20} {'Nodes':>8} {'Time(ms)':>10} {'Sat%':>8} {'Viol':>6}")
    print("  " + "─"*52)
    for r in results:
        print(f"  {r['name']:<20} {r['nodes']:>8,} {r['time']:>10.1f} "
              f"{r['sat']:>8.1f} {r['viol']:>6}")

    best = max(results, key=lambda x: x["sat"])
    print(f"\n  ★  Best algorithm for this run: {best['name']}  "
          f"({best['sat']:.1f}% satisfaction)")
    print("\n  Note: Local Search should show dramatically fewer")
    print("  nodes/iterations vs Backtracking — that is the key")

    plot_results(results)


if __name__ == "__main__":
    main()