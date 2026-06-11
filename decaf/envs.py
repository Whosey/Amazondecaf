from dataclasses import dataclass

import numpy as np

from .metrics import make_fairness


@dataclass(frozen=True)
class Candidate:
    action: int
    features: np.ndarray
    consumption: np.ndarray
    utility_prior: float


class BaseEnv:
    name = "base"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    @property
    def resources(self):
        return np.array([1.0])

    @property
    def min_resources(self):
        return np.array([0.0])

    def candidate_lists(self):
        raise NotImplementedError

    def step(self, chosen_actions):
        raise NotImplementedError

    def _z_stats(self, agent_idx):
        z = self.fairness.z
        return float(z[agent_idx]), float(np.mean(z))


def _move_toward(position, target, distance):
    delta = target - position
    norm = float(np.linalg.norm(delta))
    if norm <= distance or norm == 0.0:
        return target.copy()
    return position + (delta / norm) * distance


def _grid_next(position, action, grid_size):
    row, col = position
    if action == 1:
        row -= 1
    elif action == 2:
        row += 1
    elif action == 3:
        col -= 1
    elif action == 4:
        col += 1
    row = int(np.clip(row, 0, grid_size - 1))
    col = int(np.clip(col, 0, grid_size - 1))
    return np.array([row, col], dtype=int)


class BiasedDMEnv(BaseEnv):
    """Paper's BiasedDM domain: one resource, five agents, biased utility."""

    name = "biaseddm"

    def __init__(self, seed=0, episode_steps=100, fairness_metric="variance"):
        super().__init__(seed)
        self.n_agents = 5
        self.episode_steps = episode_steps
        self.fairness = make_fairness(
            self.n_agents,
            metric=fairness_metric,
            warm_start=2.0,
            past_discount=0.999,
            rng=self.rng,
        )
        self.t = 0

    def reset(self):
        self.t = 0
        self.fairness.reset()
        return self.candidate_lists()

    @property
    def min_resources(self):
        return np.array([1.0])

    def candidate_lists(self):
        lists = []
        for i in range(self.n_agents):
            z_i, z_mean = self._z_stats(i)
            agent_norm = i / max(1, self.n_agents - 1)
            t_norm = self.t / self.episode_steps
            candidates = []
            for action in (0, 1):
                features = np.array(
                    [agent_norm, float(action), t_norm, z_i, z_mean], dtype=float
                )
                candidates.append(
                    Candidate(
                        action=action,
                        features=features,
                        consumption=np.array([float(action)]),
                        utility_prior=float(action) * 0.2 * (i + 1),
                    )
                )
            lists.append(candidates)
        return lists

    def step(self, chosen_actions):
        utility = np.zeros(self.n_agents, dtype=float)
        resource_values = np.zeros(self.n_agents, dtype=float)
        for i, action in enumerate(chosen_actions):
            if action == 1:
                utility[i] = 0.2 * (i + 1)
                resource_values[i] = 1.0

        fair_rewards = self.fairness.update(resource_values)
        self.t += 1
        done = self.t >= self.episode_steps
        info = {"variance": self.fairness.variance, "fairness": self.fairness.fairness}
        return utility, fair_rewards, self.candidate_lists(), done, info


class JobAllocEnv(BaseEnv):
    """Simplified Job allocation environment from Appendix C.3."""

    name = "joballoc"

    def __init__(self, seed=0, episode_steps=100, fairness_metric="variance"):
        super().__init__(seed)
        self.n_agents = 4
        self.episode_steps = episode_steps
        self.fairness = make_fairness(
            self.n_agents,
            metric=fairness_metric,
            warm_start=3.0,
            past_discount=0.995,
            rng=self.rng,
        )
        self.t = 0
        self.occupant = -1

    def reset(self):
        self.t = 0
        self.occupant = -1
        self.fairness.reset()
        return self.candidate_lists()

    def candidate_lists(self):
        lists = []
        for i in range(self.n_agents):
            z_i, z_mean = self._z_stats(i)
            occupied_self = 1.0 if self.occupant == i else 0.0
            occupied_any = 1.0 if self.occupant >= 0 else 0.0
            base = [
                i / max(1, self.n_agents - 1),
                occupied_self,
                occupied_any,
                self.t / self.episode_steps,
                z_i,
                z_mean,
            ]

            allowed_actions = [0, 1] if self.occupant in (-1, i) else [0]
            candidates = []
            for action in allowed_actions:
                consumes = 1.0 if action == 1 else 0.0
                features = np.array(base + [float(action)], dtype=float)
                candidates.append(
                    Candidate(
                        action=action,
                        features=features,
                        consumption=np.array([consumes]),
                        utility_prior=float(action),
                    )
                )
            lists.append(candidates)
        return lists

    def step(self, chosen_actions):
        if self.occupant == -1:
            claimers = [i for i, action in enumerate(chosen_actions) if action == 1]
            self.occupant = claimers[0] if claimers else -1
        else:
            if chosen_actions[self.occupant] == 0:
                self.occupant = -1

        utility = np.zeros(self.n_agents, dtype=float)
        if self.occupant >= 0:
            utility[self.occupant] = 1.0
        fair_rewards = self.fairness.update(utility)

        self.t += 1
        done = self.t >= self.episode_steps
        info = {
            "occupant": self.occupant,
            "variance": self.fairness.variance,
            "fairness": self.fairness.fairness,
        }
        return utility, fair_rewards, self.candidate_lists(), done, info


class JobEnv(BaseEnv):
    """Grid Job domain from Appendix C.2."""

    name = "job"

    def __init__(self, seed=0, episode_steps=100, fairness_metric="variance"):
        super().__init__(seed)
        self.n_agents = 4
        self.grid_size = 7
        self.episode_steps = episode_steps
        self.fairness = make_fairness(
            self.n_agents,
            metric=fairness_metric,
            warm_start=3.0,
            past_discount=0.995,
            rng=self.rng,
        )
        self.job = np.array([3, 3], dtype=int)
        self.t = 0
        self.positions = np.zeros((self.n_agents, 2), dtype=int)

    @property
    def resources(self):
        return np.ones(self.grid_size * self.grid_size)

    def reset(self):
        self.t = 0
        self.positions = np.array([[0, 0], [0, 6], [6, 0], [6, 6]], dtype=int)
        self.fairness.reset()
        return self.candidate_lists()

    def _cell_idx(self, position):
        return int(position[0] * self.grid_size + position[1])

    def candidate_lists(self):
        lists = []
        for i in range(self.n_agents):
            z_i, z_mean = self._z_stats(i)
            candidates = []
            for action in range(5):
                next_pos = _grid_next(self.positions[i], action, self.grid_size)
                consumption = np.zeros(self.grid_size * self.grid_size)
                consumption[self._cell_idx(next_pos)] = 1.0
                dist = float(np.abs(next_pos - self.job).sum())
                features = np.array(
                    [
                        i / max(1, self.n_agents - 1),
                        self.positions[i, 0] / (self.grid_size - 1),
                        self.positions[i, 1] / (self.grid_size - 1),
                        next_pos[0] / (self.grid_size - 1),
                        next_pos[1] / (self.grid_size - 1),
                        self.job[0] / (self.grid_size - 1),
                        self.job[1] / (self.grid_size - 1),
                        dist / (2 * (self.grid_size - 1)),
                        self.t / self.episode_steps,
                        z_i,
                        z_mean,
                    ],
                    dtype=float,
                )
                candidates.append(
                    Candidate(
                        action=action,
                        features=features,
                        consumption=consumption,
                        utility_prior=1.0 if dist == 0.0 else 1.0 / (dist + 1.0),
                    )
                )
            lists.append(candidates)
        return lists

    def step(self, chosen_actions):
        for i, action in enumerate(chosen_actions):
            self.positions[i] = _grid_next(self.positions[i], action, self.grid_size)

        utility = np.array(
            [1.0 if np.array_equal(pos, self.job) else 0.0 for pos in self.positions],
            dtype=float,
        )
        fair_rewards = self.fairness.update(utility)
        self.t += 1
        done = self.t >= self.episode_steps
        info = {"variance": self.fairness.variance, "fairness": self.fairness.fairness}
        return utility, fair_rewards, self.candidate_lists(), done, info


class MatthewEnv(BaseEnv):
    """Continuous resource collection domain from Appendix C.1."""

    name = "matthew"

    def __init__(self, seed=0, episode_steps=200, fairness_metric="variance"):
        super().__init__(seed)
        self.n_agents = 10
        self.n_resources = 3
        self.episode_steps = episode_steps
        self.fairness = make_fairness(
            self.n_agents,
            metric=fairness_metric,
            warm_start=5.0,
            past_discount=0.995,
            rng=self.rng,
        )
        self.t = 0
        self.positions = np.zeros((self.n_agents, 2), dtype=float)
        self.resources_pos = np.zeros((self.n_resources, 2), dtype=float)
        self.sizes = np.ones(self.n_agents, dtype=float)

    @property
    def resources(self):
        return np.ones(self.n_resources)

    def reset(self):
        self.t = 0
        self.positions = self.rng.uniform(0.0, 1.0, size=(self.n_agents, 2))
        self.resources_pos = self.rng.uniform(0.0, 1.0, size=(self.n_resources, 2))
        self.sizes = np.ones(self.n_agents, dtype=float)
        self.sizes[:4] = 1.5
        self.fairness.reset()
        return self.candidate_lists()

    def _speed(self, agent_idx):
        return min(0.08, 0.018 * self.sizes[agent_idx])

    def candidate_lists(self):
        lists = []
        for i in range(self.n_agents):
            z_i, z_mean = self._z_stats(i)
            candidates = [
                Candidate(
                    action=0,
                    features=np.array(
                        [
                            i / max(1, self.n_agents - 1),
                            self.positions[i, 0],
                            self.positions[i, 1],
                            0.0,
                            0.0,
                            0.0,
                            self.sizes[i] / 3.0,
                            self.t / self.episode_steps,
                            z_i,
                            z_mean,
                        ],
                        dtype=float,
                    ),
                    consumption=np.zeros(self.n_resources),
                    utility_prior=0.0,
                )
            ]
            for resource_idx in range(self.n_resources):
                target = self.resources_pos[resource_idx]
                dist = float(np.linalg.norm(target - self.positions[i]))
                consumption = np.zeros(self.n_resources)
                consumption[resource_idx] = 1.0
                features = np.array(
                    [
                        i / max(1, self.n_agents - 1),
                        self.positions[i, 0],
                        self.positions[i, 1],
                        target[0],
                        target[1],
                        dist,
                        self.sizes[i] / 3.0,
                        self.t / self.episode_steps,
                        z_i,
                        z_mean,
                    ],
                    dtype=float,
                )
                candidates.append(
                    Candidate(
                        action=resource_idx + 1,
                        features=features,
                        consumption=consumption,
                        utility_prior=1.0 / (dist + 0.05),
                    )
                )
            lists.append(candidates)
        return lists

    def step(self, chosen_actions):
        utility = np.zeros(self.n_agents, dtype=float)
        for i, action in enumerate(chosen_actions):
            if action <= 0:
                continue
            resource_idx = action - 1
            target = self.resources_pos[resource_idx]
            self.positions[i] = _move_toward(self.positions[i], target, self._speed(i))
            if np.linalg.norm(self.positions[i] - target) < 1e-9:
                utility[i] = 1.0
                self.sizes[i] = min(3.0, self.sizes[i] + 0.08)
                self.resources_pos[resource_idx] = self.rng.uniform(0.0, 1.0, size=2)

        fair_rewards = self.fairness.update(utility)
        self.t += 1
        done = self.t >= self.episode_steps
        info = {"variance": self.fairness.variance, "fairness": self.fairness.fairness}
        return utility, fair_rewards, self.candidate_lists(), done, info


class PlantEnv(BaseEnv):
    """Grid resource-combination domain from Appendix C.4."""

    name = "plant"

    def __init__(self, seed=0, episode_steps=200, fairness_metric="variance"):
        super().__init__(seed)
        self.n_agents = 5
        self.grid_size = 8
        self.n_resources = 8
        self.episode_steps = episode_steps
        self.requirements = np.array(
            [[2, 1, 0], [1, 0, 1], [1, 0, 0], [1, 3, 0], [0, 1, 2]],
            dtype=int,
        )
        self.fairness = make_fairness(
            self.n_agents,
            metric=fairness_metric,
            warm_start=1.0,
            past_discount=0.995,
            rng=self.rng,
        )
        self.t = 0
        self.positions = np.zeros((self.n_agents, 2), dtype=int)
        self.resource_positions = np.zeros((self.n_resources, 2), dtype=int)
        self.resource_types = np.zeros(self.n_resources, dtype=int)
        self.inventory = np.zeros((self.n_agents, 3), dtype=int)

    @property
    def resources(self):
        return np.ones(self.n_resources)

    def reset(self):
        self.t = 0
        self.positions = self.rng.integers(
            0, self.grid_size, size=(self.n_agents, 2), endpoint=False
        )
        self.resource_positions = self.rng.integers(
            0, self.grid_size, size=(self.n_resources, 2), endpoint=False
        )
        self.resource_types = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=int)
        self.rng.shuffle(self.resource_types)
        self.inventory = np.zeros((self.n_agents, 3), dtype=int)
        self.fairness.reset()
        return self.candidate_lists()

    def candidate_lists(self):
        lists = []
        for i in range(self.n_agents):
            z_i, z_mean = self._z_stats(i)
            base = [
                i / max(1, self.n_agents - 1),
                self.positions[i, 0] / (self.grid_size - 1),
                self.positions[i, 1] / (self.grid_size - 1),
                self.t / self.episode_steps,
                z_i,
                z_mean,
            ]
            candidates = [
                Candidate(
                    action=0,
                    features=np.array(
                        base + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0]
                        + (self.inventory[i] / 4.0).tolist()
                        + (self.requirements[i] / 4.0).tolist(),
                        dtype=float,
                    ),
                    consumption=np.zeros(self.n_resources),
                    utility_prior=0.0,
                )
            ]
            for resource_idx in range(self.n_resources):
                target = self.resource_positions[resource_idx]
                resource_type = int(self.resource_types[resource_idx])
                dist = float(np.abs(target - self.positions[i]).sum())
                needed = max(0, self.requirements[i, resource_type] - self.inventory[i, resource_type])
                consumption = np.zeros(self.n_resources)
                consumption[resource_idx] = 1.0
                type_onehot = [1.0 if resource_type == k else 0.0 for k in range(3)]
                features = np.array(
                    base
                    + [
                        target[0] / (self.grid_size - 1),
                        target[1] / (self.grid_size - 1),
                        dist / (2 * (self.grid_size - 1)),
                    ]
                    + type_onehot
                    + (self.inventory[i] / 4.0).tolist()
                    + (self.requirements[i] / 4.0).tolist(),
                    dtype=float,
                )
                candidates.append(
                    Candidate(
                        action=resource_idx + 1,
                        features=features,
                        consumption=consumption,
                        utility_prior=(1.0 + needed) / (dist + 1.0),
                    )
                )
            lists.append(candidates)
        return lists

    def _step_grid_toward(self, position, target):
        next_pos = position.copy()
        delta = target - position
        axis = 0 if abs(delta[0]) >= abs(delta[1]) else 1
        if delta[axis] > 0:
            next_pos[axis] += 1
        elif delta[axis] < 0:
            next_pos[axis] -= 1
        return next_pos

    def step(self, chosen_actions):
        utility = np.zeros(self.n_agents, dtype=float)
        for i, action in enumerate(chosen_actions):
            if action <= 0:
                continue
            resource_idx = action - 1
            target = self.resource_positions[resource_idx]
            self.positions[i] = self._step_grid_toward(self.positions[i], target)
            if np.array_equal(self.positions[i], target):
                resource_type = int(self.resource_types[resource_idx])
                self.inventory[i, resource_type] += 1
                self.resource_positions[resource_idx] = self.rng.integers(
                    0, self.grid_size, size=2, endpoint=False
                )
                if np.all(self.inventory[i] >= self.requirements[i]):
                    utility[i] = 1.0
                    self.inventory[i] -= self.requirements[i]

        fair_rewards = self.fairness.update(utility)
        self.t += 1
        done = self.t >= self.episode_steps
        info = {"variance": self.fairness.variance, "fairness": self.fairness.fairness}
        return utility, fair_rewards, self.candidate_lists(), done, info


def make_env(name, seed=0, episode_steps=None, fairness_metric="variance"):
    name = name.lower()
    if name == "biaseddm":
        return BiasedDMEnv(
            seed=seed, episode_steps=episode_steps or 100, fairness_metric=fairness_metric
        )
    if name == "joballoc":
        return JobAllocEnv(
            seed=seed, episode_steps=episode_steps or 100, fairness_metric=fairness_metric
        )
    if name == "job":
        return JobEnv(
            seed=seed, episode_steps=episode_steps or 100, fairness_metric=fairness_metric
        )
    if name == "matthew":
        return MatthewEnv(
            seed=seed, episode_steps=episode_steps or 200, fairness_metric=fairness_metric
        )
    if name == "plant":
        return PlantEnv(
            seed=seed, episode_steps=episode_steps or 200, fairness_metric=fairness_metric
        )
    raise ValueError(f"Unknown environment: {name}")
