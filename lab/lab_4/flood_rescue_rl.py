"""
Static Flood Rescue GridWorld - RL Assignment
-------------------------------------------------
Scenario: A rescue boat must travel from a relief camp to an evacuation
shelter across a flooded Dhaka neighborhood. Unlike a rising-flood version,
the flood extent here is FIXED (static) - some cells are permanently deep
water (impassable/dangerous), some are shallow water (passable but slower/
riskier), and the rest are dry land. This keeps the environment a simple,
standard, stationary MDP: state = just the boat's grid position.

Grid legend:
    S = start (relief camp)
    G = goal (evacuation shelter)
    X = deep water (terminal, boat capsizes, reward -10)
    ~ = shallow water (passable, reward -0.3)
    . = dry land (passable, reward -0.04)

  S . . . . .
  . . ~ X ~ .
  . ~ X X X ~
  . ~ X X X ~
  . . ~ X ~ .
  . . . . . G

We solve this MDP with Value Iteration and Q-Learning, and compare the
results across different discount factors (gamma) and, for Q-learning,
different learning rates (alpha).
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. ENVIRONMENT
# ---------------------------------------------------------
GRID_SIZE = 6
START = (0, 0)
GOAL = (5, 5)

DEEP_WATER = {
    (1, 3),
    (2, 3),
    (3, 3)
}

SHALLOW_WATER = {
    (2, 1),
    (2, 2),
    (3, 2),
    (4, 2),
    (4, 3),
    (4, 4)
}

# DEEP_WATER = {(1, 3), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 3)}
# SHALLOW_WATER = {(1, 2), (1, 4), (2, 1), (2, 5), (3, 1), (3, 5), (4, 2), (4, 4)}

ACTIONS = ["up", "down", "left", "right"]
ACTION_MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

STEP_REWARD = -0.04
SHALLOW_REWARD = -0.05  #0.3
DEEP_REWARD = -10
GOAL_REWARD = 10


def cell_type(pos):
    if pos == GOAL:
        return "goal"
    if pos in DEEP_WATER:
        return "deep"
    if pos in SHALLOW_WATER:
        return "shallow"
    return "dry"


def all_states():
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def is_terminal(state):
    return state == GOAL or state in DEEP_WATER


def step(state, action):
    """Deterministic transition. Returns (next_state, reward)."""
    if is_terminal(state):
        return state, 0

    r, c = state
    dr, dc = ACTION_MOVES[action]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr, nc = r, c  # wall -> stay

    next_state = (nr, nc)
    ctype = cell_type(next_state)

    if ctype == "goal":
        reward = GOAL_REWARD
    elif ctype == "deep":
        reward = DEEP_REWARD
    elif ctype == "shallow":
        reward = SHALLOW_REWARD
    else:
        reward = STEP_REWARD

    return next_state, reward


# ---------------------------------------------------------
# 2. VALUE ITERATION
# ---------------------------------------------------------
def value_iteration(gamma, theta=1e-4, max_iter=1000):
    V = {s: 0.0 for s in all_states()}
    history = []

    for i in range(max_iter):
        delta = 0
        for s in all_states():
            if is_terminal(s):
                continue
            values = []
            for a in ACTIONS:
                s_next, r = step(s, a)
                values.append(r + gamma * V[s_next])
            best = max(values)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        history.append(np.mean(list(V.values())))
        if delta < theta:
            break

    policy = {}
    for s in all_states():
        if is_terminal(s):
            policy[s] = "-"
            continue
        best_a, best_val = None, -np.inf
        for a in ACTIONS:
            s_next, r = step(s, a)
            val = r + gamma * V[s_next]
            if val > best_val:
                best_val = val
                best_a = a
        policy[s] = best_a

    return V, policy, history, i + 1


# ---------------------------------------------------------
# 3. Q-LEARNING (with epsilon-decay for cleaner convergence)
# ---------------------------------------------------------
def q_learning(gamma, alpha=0.1, epsilon_start=0.3, epsilon_end=0.02,
               episodes=3000, max_steps=25):
    Q = {(s, a): 0.0 for s in all_states() for a in ACTIONS}
    reward_history = []

    for ep in range(episodes):
        frac = ep / max(episodes - 1, 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

        s = START
        total_reward = 0
        for t in range(max_steps):
            if random.random() < epsilon:
                a = random.choice(ACTIONS)
            else:
                a = max(ACTIONS, key=lambda act: Q[(s, act)])

            s_next, r = step(s, a)
            total_reward += r

            best_next = max(Q[(s_next, a2)] for a2 in ACTIONS)
            Q[(s, a)] += alpha * (r + gamma * best_next - Q[(s, a)])

            s = s_next
            if is_terminal(s):
                break

        reward_history.append(total_reward)

    policy = {}
    for s in all_states():
        if is_terminal(s):
            policy[s] = "-"
            continue
        policy[s] = max(ACTIONS, key=lambda act: Q[(s, act)])

    return Q, policy, reward_history


# ---------------------------------------------------------
# 4. PATH TRACING
# ---------------------------------------------------------
def trace_path(policy, max_steps=25):
    s = START
    path = [s]
    rewards = []
    for t in range(max_steps):
        if is_terminal(s):
            break
        a = policy[s]
        s_next, r = step(s, a)
        rewards.append(r)
        path.append(s_next)
        s = s_next

    if s == GOAL:
        outcome = "REACHED SHELTER"
    elif s in DEEP_WATER:
        outcome = "BOAT CAPSIZED (deep water)"
    else:
        outcome = "DID NOT FINISH (ran out of steps)"

    return path, sum(rewards), outcome


# ---------------------------------------------------------
# 5. PLOTTING
# ---------------------------------------------------------
COLOR_DRY = "#e8e8e8"
COLOR_SHALLOW = "#7ec8e3"
COLOR_DEEP = "#1a4d8f"
COLOR_GOAL = "#2e7d32"
COLOR_START = "#f9a825"


def draw_grid(ax):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ctype = cell_type((r, c))
            color = {"dry": COLOR_DRY, "shallow": COLOR_SHALLOW,
                     "deep": COLOR_DEEP, "goal": COLOR_GOAL}[ctype]
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="gray"))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def plot_static_map(filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_grid(ax)
    ax.plot(START[1] + 0.5, START[0] + 0.5, marker="s", color=COLOR_START,
            markersize=16, markeredgecolor="black", zorder=6)
    ax.set_title("Flooded Neighborhood Map (static)")
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=COLOR_START, label="Start (relief camp)"),
        mpatches.Patch(color=COLOR_DRY, label="Dry land"),
        mpatches.Patch(color=COLOR_SHALLOW, label="Shallow water"),
        mpatches.Patch(color=COLOR_DEEP, label="Deep water (danger)"),
        mpatches.Patch(color=COLOR_GOAL, label="Shelter (goal)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9)
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_paths(paths, titles, outcomes, filename):
    fig, axes = plt.subplots(1, len(paths), figsize=(4.4 * len(paths), 4.8))
    if len(paths) == 1:
        axes = [axes]
    for ax, path, title, outcome in zip(axes, paths, titles, outcomes):
        draw_grid(ax)

        if outcome == "REACHED SHELTER" or outcome == "BOAT CAPSIZED (deep water)":
            visit_count = {}
            xs, ys = [], []
            for p in path:
                n = visit_count.get(p, 0)
                visit_count[p] = n + 1
                offset_x = 0.12 * n * (1 if (p[0] + p[1]) % 2 == 0 else -1)
                offset_y = 0.12 * n * (1 if p[0] % 2 == 0 else -1)
                xs.append(p[1] + 0.5 + offset_x)
                ys.append(p[0] + 0.5 + offset_y)

            ax.plot(xs, ys, color="black", linewidth=1.6, zorder=4, alpha=0.8)
            ax.scatter(xs, ys, c=range(len(xs)), cmap="autumn_r", s=55,
                       edgecolors="black", linewidths=0.6, zorder=5)
            for i, (x, y) in enumerate(zip(xs, ys)):
                ax.annotate(str(i), (x, y), fontsize=6.5, ha="center", va="center",
                            zorder=6, fontweight="bold")
        else:
            # Policy never reaches a real outcome (goal or capsize) - drawing the
            # raw step trace here would be misleading, since with such a low gamma
            # the "policy" is really just arbitrary tie-breaking, not a real route.
            ax.text(GRID_SIZE / 2, GRID_SIZE / 2,
                    "No coherent path -\npolicy is essentially\nundirected (gamma too\nlow to reach the goal)",
                    ha="center", va="center", fontsize=9, color="black",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
                    zorder=6)

        ax.plot(START[1] + 0.5, START[0] + 0.5, marker="s", color=COLOR_START, markersize=14,
                markeredgecolor="black", zorder=6)
        ax.set_title(f"{title}\n{outcome} ({len(path)-1} steps)", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_convergence(histories, labels, filename, ylabel, title):
    plt.figure(figsize=(6, 4.5))
    for h, lab in zip(histories, labels):
        plt.plot(h, label=lab)
    plt.xlabel("Iteration / Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ---------------------------------------------------------
# 6. EXPERIMENTS
# ---------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    plot_static_map("static_flood_map.png")

    gammas = [0.1, 0.5, 0.9, 0.99]

    print("=" * 70)
    print("VALUE ITERATION across different gamma values")
    print("=" * 70)

    vi_paths, vi_titles, vi_outcomes, vi_histories = [], [], [], []
    for g in gammas:
        V, policy, history, sweeps = value_iteration(gamma=g)
        path, total_r, outcome = trace_path(policy)
        print(f"gamma={g:<5} sweeps={sweeps:<4} steps={len(path)-1:<4} "
              f"total_reward={total_r:.2f}  outcome={outcome}")
        vi_paths.append(path)
        vi_titles.append(f"VI, γ={g}")
        vi_outcomes.append(outcome)
        vi_histories.append(history)

    plot_paths(vi_paths, vi_titles, vi_outcomes, "vi_paths_by_gamma.png")
    plot_convergence(vi_histories, [f"γ={g}" for g in gammas],
                      "vi_convergence_by_gamma.png",
                      ylabel="Mean V(s) across all states",
                      title="Value Iteration Convergence vs Gamma")

    print("\n" + "=" * 70)
    print("Q-LEARNING across different gamma values")
    print("=" * 70)

    ql_paths, ql_titles, ql_outcomes, ql_reward_histories = [], [], [], []
    for g in gammas:
        Q, policy, reward_history = q_learning(gamma=g, alpha=0.1, episodes=3000)
        path, total_r, outcome = trace_path(policy)
        print(f"gamma={g:<5} steps={len(path)-1:<4} total_reward={total_r:.2f}  outcome={outcome}")
        ql_paths.append(path)
        ql_titles.append(f"QL, γ={g}")
        ql_outcomes.append(outcome)
        ql_reward_histories.append(reward_history)

    plot_paths(ql_paths, ql_titles, ql_outcomes, "ql_paths_by_gamma.png")

    def moving_avg(x, w=50):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    smoothed = [moving_avg(h) for h in ql_reward_histories]
    plot_convergence(smoothed, [f"γ={g}" for g in gammas],
                      "ql_reward_curve_by_gamma.png",
                      ylabel="Total reward per episode (smoothed)",
                      title="Q-Learning Reward Curve vs Gamma")

    print("\n" + "=" * 70)
    print("Q-LEARNING: effect of learning rate (alpha), gamma fixed = 0.9")
    print("=" * 70)

    alphas = [0.01, 0.1, 0.5]
    alpha_paths, alpha_titles, alpha_outcomes, alpha_reward_histories = [], [], [], []
    for a in alphas:
        Q, policy, reward_history = q_learning(gamma=0.9, alpha=a, episodes=3000)
        path, total_r, outcome = trace_path(policy)
        print(f"alpha={a:<5} steps={len(path)-1:<4} total_reward={total_r:.2f}  outcome={outcome}")
        alpha_paths.append(path)
        alpha_titles.append(f"QL, α={a}")
        alpha_outcomes.append(outcome)
        alpha_reward_histories.append(reward_history)

    plot_paths(alpha_paths, alpha_titles, alpha_outcomes, "ql_paths_by_alpha.png")
    smoothed_alpha = [moving_avg(h) for h in alpha_reward_histories]
    plot_convergence(smoothed_alpha, [f"α={a}" for a in alphas],
                      "ql_reward_curve_by_alpha.png",
                      ylabel="Total reward per episode (smoothed)",
                      title="Q-Learning Reward Curve vs Learning Rate (γ=0.9)")

    print("\nAll experiments finished. Check the saved PNG files for plots.")