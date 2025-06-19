from Astar import AstarBaseline
from DNQ import TrafficRoutingEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

GRID_ROWS = 8
GRID_COLS = 8
NUM_AGENTS = 4

def plot_astar_trajectories(trajectories, goals, grid_size):
    H, W = grid_size
    plt.figure(figsize=(H, W))

    for x in range(H + 1):
        plt.plot([0, W], [x, x], color='gray', linewidth=0.5)
    for y in range(W + 1):
        plt.plot([y, y], [0, H], color='gray', linewidth=0.5)

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i, traj in trajectories.items():
        xs = [pos[1] + 0.5 for pos in traj]
        ys = [(H - 1 - pos[0]) + 0.5 for pos in traj]
        plt.plot(xs, ys, marker='o', color=colors[i], label=f'Agent {i}')

        start = traj[0]
        sx, sy = start[1] + 0.5, (H - 1 - start[0]) + 0.5
        plt.scatter([sx], [sy], color=colors[i], marker='s', s=80)

        goal = goals[i]
        gx, gy = goal[1] + 0.5, (H - 1 - goal[0]) + 0.5
        plt.scatter([gx], [gy], color=colors[i], marker='*', s=120)

    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.gca().set_aspect('equal')
    plt.xticks([]);
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.title('A* Baseline Multi-Agent Trajectories')
    plt.tight_layout()
    plt.show()

def animate(trajectories, goals, grid_size):
    H, W = grid_size
    fig, ax = plt.subplots(figsize=(H, W))

    for x in range(H + 1):
        ax.plot([0, W], [x, x], color='gray', linewidth=0.5)
    for y in range(W + 1):
        ax.plot([y, y], [0, H], color='gray', linewidth=0.5)

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i, traj in trajectories.items():
        start = traj[0]
        sx, sy = start[1] + 0.5, (H - 1 - start[0]) + 0.5
        ax.scatter([sx], [sy], color=colors[i], marker='s', s=80)

        goal = goals[i]
        gx, gy = goal[1] + 0.5, (H - 1 - goal[0]) + 0.5
        ax.scatter([gx], [gy], color=colors[i], marker='*', s=120)

    lines = []
    for i in range(len(trajectories)):
        line, = ax.plot([], [], marker='o', color=colors[i], label=f'Agent {i}')
        lines.append(line)

    max_steps = max(len(traj) for traj in trajectories.values())

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            current_traj = trajectories[i][:frame+1]
            if len(current_traj) > 0:
                xs = [pos[1] + 0.5 for pos in current_traj]
                ys = [(H - 1 - pos[0]) + 0.5 for pos in current_traj]
                line.set_data(xs, ys)
        return lines

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    ax.set_title('A* Baseline Multi-Agent Trajectories Animation')

    ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, interval=500, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    ani.save('animate_results/GreedyTrajectoriesBaselineAStar.gif', writer='pillow', fps=2, dpi=100)

    return ani

if __name__ == '__main__':
    env = TrafficRoutingEnv(grid_size=(GRID_ROWS, GRID_COLS), num_agents=NUM_AGENTS)
    baseline = AstarBaseline(env)

    trajectories, goals = baseline.run_baseline()

    print('A* Baseline Trajectories:')
    for i, traj in trajectories.items():
        print(f'Agent {i}: {traj}')

    plot_astar_trajectories(trajectories, goals, env.grid_size)