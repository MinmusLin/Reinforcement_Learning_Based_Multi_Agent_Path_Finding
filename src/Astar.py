import heapq
import numpy as np
from collections import defaultdict

class AstarBaseline:

    def __init__(self, env):
        self.env = env
        self.grid_size = env.grid_size
        self.num_agents = env.num_agents

    @staticmethod
    def _heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _astar_search(self, start, goal):
        H, W = self.grid_size
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, goal), 0, start, None))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            f_curr, g_curr, current, parent = heapq.heappop(open_set)
            if current in came_from:
                continue
            came_from[current] = parent

            if current == goal:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                return path[::-1]

            x, y = current
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    neighbors.append((nx, ny))

            for neighbor in neighbors:
                tentative_g = g_curr + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))

        return None

    def run_baseline(self, max_steps=50):
        obs = self.env.reset()
        start_positions = {i: obs[i]['position'] for i in range(self.num_agents)}
        goals = {i: obs[i]['destination'] for i in range(self.num_agents)}

        paths = {}
        for i in range(self.num_agents):
            path = self._astar_search(start_positions[i], goals[i])
            if path is None:
                paths[i] = [start_positions[i]]
            else:
                paths[i] = path

        agent_positions = dict(start_positions)
        agent_trajectories = {i: [agent_positions[i]] for i in range(self.num_agents)}

        for step in range(1, max_steps + 1):
            desired = {}
            for i in range(self.num_agents):
                current = agent_positions[i]
                path = paths[i]
                if current == goals[i]:
                    desired[i] = current
                else:
                    idx = path.index(current)
                    if idx + 1 < len(path):
                        desired[i] = path[idx + 1]
                    else:
                        desired[i] = current

            new_positions = {}
            occupied = {}
            for i, pos in desired.items():
                if pos in occupied:
                    new_positions[i] = agent_positions[i]
                    other = occupied[pos]
                    new_positions[other] = agent_positions[other]
                else:
                    occupied[pos] = i
                    new_positions[i] = pos

            agent_positions = new_positions
            for i in range(self.num_agents):
                agent_trajectories[i].append(agent_positions[i])
            if all(agent_positions[i] == goals[i] for i in range(self.num_agents)):
                break

        return agent_trajectories, goals