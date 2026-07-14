"""
GridWorld RL Assignment
------------------------
Problem: 5x5 GridWorld
- Agent starts at top-left (0,0)
- Goal at bottom-right (4,4)  -> reward +10
- Pit at (2,2)                -> reward -10
- Every other move            -> reward -0.04 (small cost to encourage shortest path)
- Actions: up, down, left, right (deterministic; hitting a wall = stay in place)

We solve this environment two ways:
1. Value Iteration  (planning, needs full model of environment)
2. Q-Learning       (model-free, learns from trial and error)

Then we repeat both algorithms with different discount factors (gamma)
to see how gamma changes the learned policy / values.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. ENVIRONMENT
# ---------------------------------------------------------
GRID_SIZE = 5
GOAL = (4, 4)
PIT = (2, 2)
START = (0, 0)

ACTIONS = ["up", "down", "left", "right"]
ACTION_MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

STEP_REWARD = -0.04
GOAL_REWARD = 10
PIT_REWARD = -10


def is_terminal(state):
    return state == GOAL or state == PIT


def step(state, action):
    """Deterministic transition. Returns (next_state, reward)."""
    if is_terminal(state):
        return state, 0  # no movement from terminal states

    dr, dc = ACTION_MOVES[action]
    r, c = state
    nr, nc = r + dr, c + dc

    # hitting a wall -> stay in place
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr, nc = r, c

    next_state = (nr, nc)

    if next_state == GOAL:
        return next_state, GOAL_REWARD
    elif next_state == PIT:
        return next_state, PIT_REWARD
    else:
        return next_state, STEP_REWARD


def all_states():
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


# ---------------------------------------------------------
# 2. VALUE ITERATION
# ---------------------------------------------------------
def value_iteration(gamma, theta=1e-4, max_iter=1000):
    V = {s: 0.0 for s in all_states()}
    history = []  # track avg value per sweep, for convergence plot

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

    # derive greedy policy from V
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

    return V, policy, history, i + 1  # i+1 = sweeps to converge


# ---------------------------------------------------------
# 3. Q-LEARNING
# ---------------------------------------------------------
def q_learning(gamma, alpha=0.1, epsilon=0.2, episodes=2000, max_steps=100):
    Q = {(s, a): 0.0 for s in all_states() for a in ACTIONS}
    reward_history = []

    for ep in range(episodes):
        s = START
        total_reward = 0
        for t in range(max_steps):
            # epsilon-greedy action choice
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

    # derive greedy policy from Q
    policy = {}
    V = {}
    for s in all_states():
        if is_terminal(s):
            policy[s] = "-"
            V[s] = 0.0
            continue
        best_a = max(ACTIONS, key=lambda act: Q[(s, act)])
        policy[s] = best_a
        V[s] = max(Q[(s, a)] for a in ACTIONS)

    return Q, policy, V, reward_history


# ---------------------------------------------------------
# 4. HELPERS: printing / plotting
# ---------------------------------------------------------
ARROW = {"up": "^", "down": "v", "left": "<", "right": ">", "-": "*"}


def print_policy(policy, title=""):
    print(f"\n{title}")
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            if s == GOAL:
                row.append(" G ")
            elif s == PIT:
                row.append(" X ")
            else:
                row.append(f" {ARROW[policy[s]]} ")
        print("".join(row))


def policy_to_grid(policy):
    """Return grid of symbols for plotting."""
    grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = (r, c)
            if s == GOAL:
                grid[r, c] = "G"
            elif s == PIT:
                grid[r, c] = "X"
            else:
                grid[r, c] = ARROW[policy[s]]
    return grid


def plot_policies(policies, titles, filename):
    fig, axes = plt.subplots(1, len(policies), figsize=(4 * len(policies), 4.2))
    if len(policies) == 1:
        axes = [axes]
    for ax, policy, title in zip(axes, policies, titles):
        grid = policy_to_grid(policy)
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.invert_yaxis()
        ax.set_xticks(range(GRID_SIZE + 1))
        ax.set_yticks(range(GRID_SIZE + 1))
        ax.grid(True)
        ax.set_title(title, fontsize=11)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                sym = grid[r, c]
                color = "black"
                if sym == "G":
                    color = "green"
                elif sym == "X":
                    color = "red"
                ax.text(c + 0.5, r + 0.5, sym, ha="center", va="center",
                         fontsize=16, color=color, fontweight="bold")
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
# 5. EXPERIMENTS: vary discount factor (gamma)
# ---------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    gammas = [0.1, 0.5, 0.9, 0.99]

    print("=" * 60)
    print("VALUE ITERATION across different gamma values")
    print("=" * 60)

    vi_policies, vi_titles, vi_histories = [], [], []
    for g in gammas:
        V, policy, history, sweeps = value_iteration(gamma=g)
        print_policy(policy, title=f"Gamma = {g}  (converged in {sweeps} sweeps)")
        vi_policies.append(policy)
        vi_titles.append(f"VI, γ={g}")
        vi_histories.append(history)

    plot_policies(vi_policies, vi_titles, "vi_policies_by_gamma.png")
    plot_convergence(vi_histories, [f"γ={g}" for g in gammas],
                      "vi_convergence_by_gamma.png",
                      ylabel="Mean V(s) across grid",
                      title="Value Iteration Convergence vs Gamma")

    print("\n" + "=" * 60)
    print("Q-LEARNING across different gamma values")
    print("=" * 60)

    ql_policies, ql_titles, ql_reward_histories = [], [], []
    for g in gammas:
        Q, policy, V, reward_history = q_learning(gamma=g, alpha=0.1, epsilon=0.2, episodes=2000)
        print_policy(policy, title=f"Gamma = {g}")
        ql_policies.append(policy)
        ql_titles.append(f"QL, γ={g}")
        ql_reward_histories.append(reward_history)

    plot_policies(ql_policies, ql_titles, "ql_policies_by_gamma.png")

    # smooth reward curves for readability (moving average)
    def moving_avg(x, w=50):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    smoothed = [moving_avg(h) for h in ql_reward_histories]
    plot_convergence(smoothed, [f"γ={g}" for g in gammas],
                      "ql_reward_curve_by_gamma.png",
                      ylabel="Total reward per episode (smoothed)",
                      title="Q-Learning Reward Curve vs Gamma")

    # ------------------------------------------------------
    # Bonus: vary alpha (learning rate) at fixed gamma=0.9
    # ------------------------------------------------------
    print("\n" + "=" * 60)
    print("Q-LEARNING: effect of learning rate (alpha), gamma fixed = 0.9")
    print("=" * 60)

    alphas = [0.01, 0.1, 0.5]
    alpha_reward_histories = []
    for a in alphas:
        Q, policy, V, reward_history = q_learning(gamma=0.9, alpha=a, epsilon=0.2, episodes=2000)
        print_policy(policy, title=f"Alpha = {a}")
        alpha_reward_histories.append(reward_history)

    smoothed_alpha = [moving_avg(h) for h in alpha_reward_histories]
    plot_convergence(smoothed_alpha, [f"α={a}" for a in alphas],
                      "ql_reward_curve_by_alpha.png",
                      ylabel="Total reward per episode (smoothed)",
                      title="Q-Learning Reward Curve vs Learning Rate (γ=0.9)")

    print("\nAll experiments finished. Check the saved PNG files for plots.")
