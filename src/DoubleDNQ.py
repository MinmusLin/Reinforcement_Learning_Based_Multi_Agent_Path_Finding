import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import time

GRID_ROWS = 8
GRID_COLS = 8
NUM_AGENTS = 4
NUM_EPISODES = 400

class TrafficRoutingEnv:

    def __init__(self, grid_size=(5, 5), num_agents=2, obstacles=None):
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)

        self.grid_size = grid_size
        self.num_agents = num_agents

        self._initial_positions = {
            i: (np.random.randint(self.grid_size[0]),
                np.random.randint(self.grid_size[1]))
            for i in range(self.num_agents)
        }

        self.destinations = {
            i: (np.random.randint(self.grid_size[0]),
                np.random.randint(self.grid_size[1]))
            for i in range(self.num_agents)
        }

        self.agent_positions = dict(self._initial_positions)

        self._arrived = {i: False for i in range(self.num_agents)}
        self.steps = 0

        if obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = obstacles.copy()

        for obs in self.obstacles:
            for i in range(self.num_agents):
                if obs == self._initial_positions[i]:
                    raise ValueError(f'障碍物 {obs} 与 Agent {i} 的初始位置冲突！')
                if obs == self.destinations[i]:
                    raise ValueError(f'障碍物 {obs} 与 Agent {i} 的目的地冲突！')

    def reset(self):
        self.agent_positions = dict(self._initial_positions)
        self._arrived = {i: False for i in range(self.num_agents)}
        self.steps = 0

        return self._get_observations()

    def _get_observations(self):
        obs = {}
        for i in range(self.num_agents):
            obs[i] = {
                'position': self.agent_positions[i],
                'destination': self.destinations[i],

                'obstacles': self.obstacles
            }

        return obs

    def step(self, actions):
        old_positions = {i: self.agent_positions[i] for i in range(self.num_agents)}

        new_positions = {}
        for i, action in actions.items():
            if self._arrived.get(i, False):
                new_positions[i] = old_positions[i]
                continue

            x, y = old_positions[i]
            if action == 0 and x > 0:
                x -= 1
            elif action == 1 and x < self.grid_size[0] - 1:
                x += 1
            elif action == 2 and y > 0:
                y -= 1
            elif action == 3 and y < self.grid_size[1] - 1:
                y += 1

            if (x, y) in self.obstacles:
                new_positions[i] = old_positions[i]
            else:
                new_positions[i] = (x, y)

        swap_blocked = set()
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (new_positions[i] == old_positions[j] and new_positions[j] == old_positions[i]):
                    swap_blocked.add(i)
                    swap_blocked.add(j)

        from collections import defaultdict
        desired_cells = defaultdict(list)
        for i in range(self.num_agents):
            if i in swap_blocked:
                continue
            desired_cells[new_positions[i]].append(i)

        collision_blocked = set()
        for cell, agents in desired_cells.items():
            if len(agents) > 1:
                for a in agents:
                    collision_blocked.add(a)

        num_collisions = len(swap_blocked) + len(collision_blocked)

        blocked = swap_blocked.union(collision_blocked)

        final_positions = {}
        for i in range(self.num_agents):
            if i in blocked:
                final_positions[i] = old_positions[i]
            else:
                final_positions[i] = new_positions[i]

        self.agent_positions = final_positions
        self.steps += 1

        raw_rewards = {}
        dones = {}
        for i in range(self.num_agents):
            if self.agent_positions[i] == self.destinations[i]:
                if not self._arrived[i]:
                    raw_rewards[i] = +10
                    self._arrived[i] = True
                else:
                    raw_rewards[i] = 0
                dones[i] = True
            else:
                raw_rewards[i] = -1
                dones[i] = False

        shaped_rewards = {}
        gamma = 0.99
        for i in range(self.num_agents):
            ox, oy = old_positions[i]
            gx, gy = self.destinations[i]
            dist_old = abs(ox - gx) + abs(oy - gy)

            nx, ny = self.agent_positions[i]
            dist_new = abs(nx - gx) + abs(ny - gy)

            phi_old = -dist_old
            phi_new = -dist_new

            reward32 = np.float32(raw_rewards[i] + (gamma * phi_new - phi_old))
            shaped_rewards[i] = reward32

        done = all(dones.values()) or (self.steps >= 50)

        return self._get_observations(), shaped_rewards, done, {}, num_collisions

    def render(self):
        H, W = self.grid_size
        grid = np.zeros((H, W), dtype=int)

        for (ox, oy) in self.obstacles:
            grid[ox, oy] = 3

        for i in range(self.num_agents):
            x, y = self.destinations[i]
            if grid[x, y] == 3:
                raise ValueError(f'目的地 {(x, y)} 与障碍物冲突！')
            grid[x, y] = 1

        for i in range(self.num_agents):
            x, y = self.agent_positions[i]
            if grid[x, y] == 3:
                raise ValueError(f'Agent {i} 企图进入障碍物 {(x, y)}！')
            grid[x, y] = 2

        cmap = colors.ListedColormap(['white', 'blue', 'red', 'black'])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(5, 5))
        plt.imshow(grid, cmap=cmap, norm=norm)
        plt.title(f'Step: {self.steps}')
        plt.show()

class DuelingDQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DuelingDQN, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.value_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.adv_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        h = self.shared(x)
        v = self.value_branch(h)
        a = self.adv_branch(h)
        q = v + (a - a.mean(dim=1, keepdim=True))

        return q

class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def obs_to_state(obs, grid_size):
    state_dict = {}
    H, W = grid_size

    occupancy = np.zeros((H, W), dtype=np.float32)

    if 'obstacles' in next(iter(obs.values())):
        for (ox, oy) in next(iter(obs.values()))['obstacles']:
            occupancy[ox, oy] = 1.0

    for j in range(len(obs)):
        rx, ry = obs[j]['position']
        occupancy[rx, ry] = 1.0

    for agent_id, info in obs.items():
        x, y = info['position']
        dx, dy = info['destination']

        pos_vec = np.array([x/(H-1), y/(W-1), dx/(H-1), dy/(W-1)], dtype=np.float32)

        local = np.zeros((3, 3), dtype=np.float32)
        for dx_off in (-1, 0, 1):
            for dy_off in (-1, 0, 1):
                nx, ny = x + dx_off, y + dy_off
                if 0 <= nx < H and 0 <= ny < W:
                    local[dx_off+1, dy_off+1] = occupancy[nx, ny]
        local_flat = local.flatten()

        dist = abs(x - dx) + abs(y - dy)
        dist_norm = np.array([dist / ((H - 1) + (W - 1))], dtype=np.float32)

        state = np.concatenate([pos_vec, local_flat, dist_norm], axis=0)
        state_dict[agent_id] = state

    return state_dict

class MADQNTrainer:

    def __init__(self, env, num_agents, state_dim, action_dim, buffer_capacity=5000, batch_size=64, gamma=0.99, lr=1e-3, target_update=10):
        self.env = env
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update

        self.policy_nets = {}
        self.target_nets = {}
        self.optimizers = {}
        self.replay_buffers = {}

        for i in range(num_agents):
            policy_net = DuelingDQN(state_dim, action_dim, hidden_dim=64)
            target_net = DuelingDQN(state_dim, action_dim, hidden_dim=64)

            target_net.load_state_dict(policy_net.state_dict())
            optimizer = optim.Adam(policy_net.parameters(), lr=lr)
            buffer = ReplayBuffer(buffer_capacity)

            self.policy_nets[i] = policy_net
            self.target_nets[i] = target_net
            self.optimizers[i] = optimizer
            self.replay_buffers[i] = buffer

        self.steps_done = 0

        self.loss_history = {i: [] for i in range(num_agents)}
        self.collision_history = []
        self.episode_collisions = 0

    def select_action(self, agent_id, state, eps, done):
        if done:
            return 4

        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_nets[agent_id](state_tensor)
            return int(q_values.argmax().item())

    def optimize_agent(self, agent_id):
        buffer = self.replay_buffers[agent_id]
        if len(buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions).unsqueeze(1).long()
        rewards = torch.from_numpy(rewards).unsqueeze(1)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones.astype(np.uint8)).unsqueeze(1)

        current_q = self.policy_nets[agent_id](states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_nets[agent_id](next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_nets[agent_id](next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        criterion = nn.MSELoss()
        loss = criterion(current_q, target_q)

        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()

        loss_value = loss.item()
        self.loss_history[agent_id].append(loss_value)

        return loss_value

    def update_targets(self):
        for i in range(self.num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())

    def _plot_training_stats(self, returns, losses, collisions):
        plt.figure(figsize=(15, 5))

        window_size = 10
        returns_smooth = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
        losses_smooth = np.convolve([x for x in losses if x > 0], np.ones(window_size)/window_size, mode='valid')
        collisions_smooth = np.convolve(collisions, np.ones(window_size)/window_size, mode='valid')

        plt.subplot(1, 3, 1)
        plt.plot(returns, alpha=0.3, label='Raw Return')
        plt.plot(range(window_size-1, len(returns)), returns_smooth, label=f'MA({window_size}) Return', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Returns')
        plt.legend()
        plt.grid(True)

        valid_losses = [x for x in losses if x > 0]
        plt.subplot(1, 3, 2)
        plt.plot(valid_losses, alpha=0.3, label='Raw Loss', color='orange')
        plt.plot(range(window_size-1, len(valid_losses)), losses_smooth, label=f'MA({window_size}) Loss', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(collisions, alpha=0.3, label='Raw Collisions', color='red')
        plt.plot(range(window_size-1, len(collisions)), collisions_smooth, label=f'MA({window_size}) Collisions', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Collisions per Episode')
        plt.title('Collision Count')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_results/DoubleDQN.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train(self, num_episodes=200, max_steps=50, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        eps = eps_start
        episode_returns = []
        avg_loss_history = []

        start_time = time.time()
        convergence_episode = None
        convergence_threshold = 0.9
        max_return = 0

        for episode in range(1, num_episodes + 1):
            obs = self.env.reset()
            state_dict = obs_to_state(obs, self.env.grid_size)
            done_dict = {i: False for i in range(self.num_agents)}
            total_reward = 0
            self.episode_collisions = 0

            for step in range(max_steps):
                actions = {}
                for i in range(self.num_agents):
                    actions[i] = self.select_action(i, state_dict[i], eps, done_dict[i])

                next_obs, rewards, done_all, _, num_collisions = self.env.step(actions)
                self.episode_collisions += num_collisions
                next_state_dict = obs_to_state(next_obs, self.env.grid_size)

                for i in range(self.num_agents):
                    self.replay_buffers[i].push(
                        state_dict[i],
                        actions[i],
                        rewards[i],
                        next_state_dict[i],
                        done_dict[i]
                    )

                    if rewards[i] == 10:
                        done_dict[i] = True

                    total_reward += rewards[i]

                state_dict = next_state_dict

                for i in range(self.num_agents):
                    self.optimize_agent(i)

                if done_all:
                    break

            episode_returns.append(total_reward)
            self.collision_history.append(self.episode_collisions)

            if total_reward > max_return:
                max_return = total_reward

            if len(episode_returns) >= 10 and convergence_episode is None:
                recent_avg = np.mean(episode_returns[-10:])
                if recent_avg >= max_return * convergence_threshold:
                    convergence_episode = episode
                    convergence_time = time.time() - start_time

            episode_losses = []
            for i in range(self.num_agents):
                if self.loss_history[i]:
                    episode_losses.append(np.mean(self.loss_history[i][-max_steps:]))
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            avg_loss_history.append(avg_loss)

            eps = max(eps * eps_decay, eps_end)

            if episode % self.target_update == 0:
                self.update_targets()

            if episode % 10 == 0:
                last10_avg = np.mean(episode_returns[-10:])
                last10_loss = np.mean(avg_loss_history[-10:])
                last10_coll = np.mean(self.collision_history[-10:])
                print(f'Episode {episode}/{num_episodes}, '
                      f'Epsilon: {eps:.3f}, '
                      f'AvgReturn(last10): {last10_avg:.2f}, '
                      f'AvgLoss(last10): {last10_loss:.4f}, '
                      f'AvgCollisions(last10): {last10_coll:.1f}')

        if convergence_episode is not None:
            print(f'\n算法在 {convergence_episode} 轮后收敛，花费时间: {convergence_time:.2f} 秒')
            print(f'最终10轮平均回报: {np.mean(episode_returns[-10:]):.2f}')
        else:
            print('\n算法在指定轮数内未达到收敛标准')
            print(f'最终10轮平均回报: {np.mean(episode_returns[-10:]):.2f}')
        self._plot_training_stats(episode_returns, avg_loss_history, self.collision_history)

        return episode_returns

def animate(trainer, env):
    H, W = env.grid_size
    agent_trajectories = {i: [] for i in range(env.num_agents)}
    done_dict = {i: False for i in range(env.num_agents)}

    obs = env.reset()
    state_dict = obs_to_state(obs, env.grid_size)
    for i in range(env.num_agents):
        agent_trajectories[i].append(env.agent_positions[i])

    all_steps = []
    step = 0
    done = False

    while not done and step < 50:
        actions = {
            i: trainer.select_action(i, state_dict[i], eps=0.0, done=done_dict[i])
            for i in range(env.num_agents)
        }
        next_obs, rewards, done, _, num_collisions = env.step(actions)
        next_state_dict = obs_to_state(next_obs, env.grid_size)

        for i in range(env.num_agents):
            if rewards[i] == 10 or env.agent_positions[i] == env.destinations[i]:
                done_dict[i] = True
            agent_trajectories[i].append(env.agent_positions[i])

        all_steps.append({i: env.agent_positions[i] for i in range(env.num_agents)})
        state_dict = next_state_dict
        step += 1

    print('Final Greedy Trajectories (row, col) for each agent:')
    for i, traj in agent_trajectories.items():
        print(f'Agent {i}: {traj}')

    fig, ax = plt.subplots(figsize=(8, 8))

    for x in range(H + 1):
        ax.plot([0, W], [x, x], color='gray', linewidth=0.5)
    for y in range(W + 1):
        ax.plot([y, y], [0, H], color='gray', linewidth=0.5)

    for (ox, oy) in env.obstacles:
        rect = Rectangle((oy, H - 1 - ox), 1, 1, facecolor='black', edgecolor='black')
        ax.add_patch(rect)

    base_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#FFA500']
    colors_list = [base_colors[i % len(base_colors)] for i in range(env.num_agents)]

    destinations = {}
    for i in range(env.num_agents):
        goal = env.destinations[i]
        gx, gy = goal[1] + 0.5, (H - 1 - goal[0]) + 0.5
        destinations[i] = ax.scatter([gx], [gy], color=colors_list[i], marker='*', s=120, zorder=3)

    agents = {}
    for i in range(env.num_agents):
        start = agent_trajectories[i][0]
        sx, sy = start[1] + 0.5, (H - 1 - start[0]) + 0.5
        agents[i] = ax.scatter([sx], [sy], color=colors_list[i], marker='s', s=80, zorder=3)

    lines = {}
    for i in range(env.num_agents):
        lines[i], = ax.plot([], [], color=colors_list[i], marker='o', markersize=6, linewidth=2, zorder=2)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Multi-Agent Trajectories Animation (Greedy Policy)')

    step_text = ax.text(W/2, H+0.5, 'Step: 0', ha='center', va='center', fontsize=12)

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='black', edgecolor='black', label='Obstacle'),
        Line2D([0], [0], marker='s', color='gray', label='Start', markerfacecolor='none', linestyle='None', markersize=10),
        Line2D([0], [0], marker='*', color='gray', label='Goal', markerfacecolor='none', linestyle='None', markersize=12)
    ]
    for i, color in enumerate(colors_list):
        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f'Agent {i}'))

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

    def update(frame):
        nonlocal step_text

        step_text.set_text(f'Step: {frame+1}')

        for i in range(env.num_agents):
            current_traj = agent_trajectories[i][:frame+2]

            if frame < len(all_steps):
                x, y = all_steps[frame][i]
                agents[i].set_offsets([y + 0.5, (H - 1 - x) + 0.5])

            xs = [pos[1] + 0.5 for pos in current_traj]
            ys = [(H - 1 - pos[0]) + 0.5 for pos in current_traj]
            lines[i].set_data(xs, ys)

        return list(agents.values()) + list(lines.values()) + [step_text]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(all_steps),
        interval=500,
        blit=True,
        repeat=False
    )

    plt.tight_layout()
    plt.show()

    ani.save('animate_results/GreedyTrajectoriesDoubleDQN.gif', writer='pillow', fps=2, dpi=100)

    return ani

if __name__ == '__main__':
    fixed_obstacles = [
        (1, 1),
        (3, 0),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 2),
        (5, 5),
        (6, 2),
    ]

    env = TrafficRoutingEnv(grid_size=(GRID_ROWS, GRID_COLS), num_agents=NUM_AGENTS, obstacles=fixed_obstacles)

    state_dim = 14
    action_dim = 5

    trainer = MADQNTrainer( env=env, num_agents=NUM_AGENTS, state_dim=state_dim, action_dim=action_dim)
    trainer.train(num_episodes=NUM_EPISODES)

    animate(trainer, env)