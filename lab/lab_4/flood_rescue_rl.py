"""
Flood Rescue GridWorld RL Assignment
--------------------------------------
Problem: A rescue boat must navigate a 6x6 flooded neighborhood (inspired by
Dhaka monsoon flooding) from a starting point to a shelter, while water
keeps RISING the longer the agent takes.

Environment design:
- Grid: 6x6 cells
- Start: top-left (0,0)  -> the rescue boat's launch point
- Shelter (goal): bottom-right (5,5) -> reward +10, terminal, always safe
- Flood source: a fixed cell where flooding originates
- Flood grows outward by 1 "ring" (Manhattan distance) every timestep the
  agent takes. So the flood_stage increases every action, capped at MAX_STAGE.
- Cell types (computed from distance to flood source, given current stage):
    deep water    -> distance <= stage   -> TERMINAL, reward -10 (boat capsizes)
    shallow water -> distance == stage+1 -> reward -0.3 (passable but risky/slow)
    dry land      -> otherwise            -> reward -0.04 (normal cost)

State = (row, col, flood_stage)  -- this keeps the environment Markovian:
given the current state and action, the next state and reward are fully
determined (flood_stage always advances by exactly 1 per action, capped).

Because flood_stage saturates at MAX_STAGE, this is a valid stationary MDP
(no need for special finite-horizon handling) so normal discounted
Value Iteration and normal Q-learning both apply directly.

We compare Value Iteration and Q-Learning across different discount factors
(gamma) and visualize the ACTUAL PATH each resulting policy takes from start
to goal, plus how much of the grid is flooded by the time it gets there.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# ---------------------------------------------------------
# 1. ENVIRONMENT
# ---------------------------------------------------------
GRID_SIZE = 6
START = (0, 0)
GOAL = (5, 5)
FLOOD_SOURCE = (1, 4)  # off the main diagonal, threatens a shortcut without ever reaching start/goal
FLOOD_INTERVAL = 3    # flood advances by 1 ring every this many agent actions
MAX_STAGE = 3         # flood radius caps here (source is distance 5 from both start & goal, so they stay dry)
T_MAX = 20            # elapsed-steps counter saturates here (state stays finite)

ACTIONS = ["up", "down", "left", "right"]
ACTION_MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

STEP_REWARD = -0.04
SHALLOW_REWARD = -0.3
DEEP_REWARD = -10
GOAL_REWARD = 10


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def stage_from_t(t):
    return min(t // FLOOD_INTERVAL, MAX_STAGE)


def cell_type(pos, t):
    """Return 'deep', 'shallow', or 'dry' for a cell at a given elapsed-time t."""
    if pos == GOAL:
        return "goal"
    stage = stage_from_t(t)
    d = manhattan(pos, FLOOD_SOURCE)
    if d <= stage:
        return "deep"
    elif d == stage + 1:
        return "shallow"
    else:
        return "dry"


def all_states():
    return [(r, c, t) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
            for t in range(T_MAX + 1)]


def is_terminal(state):
    r, c, t = state
    pos = (r, c)
    return pos == GOAL or cell_type(pos, t) == "deep"


def step(state, action):
    """Deterministic transition. Returns (next_state, reward)."""
    if is_terminal(state):
        return state, 0

    r, c, t = state
    dr, dc = ACTION_MOVES[action]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr, nc = r, c  # wall -> stay

    next_t = min(t + 1, T_MAX)
    next_pos = (nr, nc)
    ctype = cell_type(next_pos, next_t)

    if ctype == "goal":
        reward = GOAL_REWARD
    elif ctype == "deep":
        reward = DEEP_REWARD
    elif ctype == "shallow":
        reward = SHALLOW_REWARD
    else:
        reward = STEP_REWARD

    return (nr, nc, next_t), reward


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
# 3. Q-LEARNING
# ---------------------------------------------------------
def q_learning(gamma, alpha=0.1, epsilon=0.2, episodes=4000, max_steps=25):
    Q = {(s, a): 0.0 for s in all_states() for a in ACTIONS}
    reward_history = []

    for ep in range(episodes):
        s = (START[0], START[1], 0)
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
# 4. PATH TRACING (follow greedy policy from start to goal/failure)
# ---------------------------------------------------------
def trace_path(policy, max_steps=25):
    s = (START[0], START[1], 0)
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

    r, c, t_final = s
    if (r, c) == GOAL:
        outcome = "REACHED SHELTER"
    elif cell_type((r, c), t_final) == "deep":
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


def plot_flood_stages(filename):
    # pick t values that land on each flood ring (t = stage * FLOOD_INTERVAL)
    t_to_show = [0, FLOOD_INTERVAL * 1, FLOOD_INTERVAL * 2, FLOOD_INTERVAL * MAX_STAGE]
    fig, axes = plt.subplots(1, len(t_to_show), figsize=(4 * len(t_to_show), 4.3))
    for ax, t in zip(axes, t_to_show):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                ctype = cell_type((r, c), t)
                color = {"dry": COLOR_DRY, "shallow": COLOR_SHALLOW,
                         "deep": COLOR_DEEP, "goal": COLOR_GOAL}[ctype]
                if (r, c) == START:
                    color = COLOR_START
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="gray"))
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.invert_yaxis()
        ax.set_title(f"After {t} steps (flood ring={stage_from_t(t)})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    patches = [
        mpatches.Patch(color=COLOR_START, label="Start"),
        mpatches.Patch(color=COLOR_DRY, label="Dry land"),
        mpatches.Patch(color=COLOR_SHALLOW, label="Shallow water"),
        mpatches.Patch(color=COLOR_DEEP, label="Deep water (danger)"),
        mpatches.Patch(color=COLOR_GOAL, label="Shelter (goal)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_paths(paths, titles, outcomes, filename):
    fig, axes = plt.subplots(1, len(paths), figsize=(4.2 * len(paths), 4.6))
    if len(paths) == 1:
        axes = [axes]
    for ax, path, title, outcome in zip(axes, paths, titles, outcomes):
        final_t = path[-1][2]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                ctype = cell_type((r, c), final_t)
                color = {"dry": COLOR_DRY, "shallow": COLOR_SHALLOW,
                         "deep": COLOR_DEEP, "goal": COLOR_GOAL}[ctype]
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="gray"))

        xs = [p[1] + 0.5 for p in path]
        ys = [p[0] + 0.5 for p in path]
        ax.plot(xs, ys, color="black", linewidth=2, marker="o", markersize=5, zorder=5)
        ax.plot(xs[0], ys[0], marker="s", color=COLOR_START, markersize=12,
                markeredgecolor="black", zorder=6)

        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
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

    plot_flood_stages("flood_stages.png")

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
        Q, policy, reward_history = q_learning(gamma=g, alpha=0.1, epsilon=0.2, episodes=3000)
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
        Q, policy, reward_history = q_learning(gamma=0.9, alpha=a, epsilon=0.2, episodes=3000)
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
