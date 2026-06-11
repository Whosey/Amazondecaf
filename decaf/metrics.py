import numpy as np


class FairnessTracker:
    """Discounted payoff tracker for DECAF fairness rewards."""

    def __init__(
        self,
        n_agents,
        metric="variance",
        warm_start=0.0,
        past_discount=0.995,
        rng=None,
        alpha=1.0,
    ):
        self.n_agents = n_agents
        self.metric = metric.lower()
        self.warm_start = warm_start
        self.past_discount = past_discount
        self.rng = rng or np.random.default_rng()
        self.alpha = alpha
        self.z = np.zeros(n_agents, dtype=float)

    def reset(self):
        if self.warm_start > 0:
            self.z = self.rng.uniform(0.0, self.warm_start, size=self.n_agents)
        else:
            self.z = np.zeros(self.n_agents, dtype=float)
        return self.z.copy()

    def value(self, z=None):
        z = self.z if z is None else np.asarray(z, dtype=float)
        if self.metric == "variance":
            return -float(np.var(z))
        if self.metric == "alpha":
            safe_z = np.maximum(z, 1e-8)
            if abs(self.alpha - 1.0) < 1e-12:
                return float(np.sum(np.log(safe_z)))
            return float(np.sum((safe_z ** (1.0 - self.alpha)) / (1.0 - self.alpha)))
        if self.metric == "ggf":
            weights = 0.5 ** np.arange(self.n_agents, dtype=float)
            return float(np.sum(weights * np.sort(z)))
        if self.metric == "maximin":
            return float(np.min(z))
        raise ValueError(f"Unknown fairness metric: {self.metric}")

    def update(self, resource_values):
        old_z = self.z.copy()
        new_z = self.past_discount * old_z + np.asarray(resource_values, dtype=float)
        self.z = new_z

        if self.metric == "variance":
            return self._variance_rewards(old_z, new_z)
        if self.metric in {"alpha", "ggf"}:
            return self._equal_delta_rewards(old_z, new_z)
        if self.metric == "maximin":
            return self._maximin_rewards(old_z, new_z)
        raise ValueError(f"Unknown fairness metric: {self.metric}")

    def _variance_rewards(self, old_z, new_z):
        old_mean = float(np.mean(old_z))
        new_mean = float(np.mean(new_z))
        n = self.n_agents
        return (
            -((new_z - new_mean) ** 2) / n
            + ((old_z - old_mean) ** 2) / n
        ).astype(float)

    def _equal_delta_rewards(self, old_z, new_z):
        delta = self.value(new_z) - self.value(old_z)
        return np.full(self.n_agents, delta / self.n_agents, dtype=float)

    def _maximin_rewards(self, old_z, new_z):
        delta = self.value(new_z) - self.value(old_z)
        rewards = np.full(self.n_agents, delta / self.n_agents, dtype=float)
        old_min = float(np.min(old_z))
        new_min = float(np.min(new_z))

        rewards += np.where(np.isclose(old_z, old_min), new_z - old_z, 0.0)
        rewards += np.where(np.isclose(new_z, new_min), new_z - old_z, 0.0)

        total = float(np.sum(rewards))
        if abs(total) > 1e-12:
            rewards = rewards * (delta / total)
        return rewards.astype(float)

    @property
    def variance(self):
        return float(np.var(self.z))

    @property
    def fairness(self):
        return self.value()


VarianceFairness = FairnessTracker


def make_fairness(n_agents, metric, warm_start, past_discount, rng=None):
    metric = metric.lower()
    if metric == "alpha":
        warm_start = 0.0
        past_discount = 1.0
    elif metric == "ggf":
        warm_start = 0.1
        past_discount = 1.0
    return FairnessTracker(
        n_agents=n_agents,
        metric=metric,
        warm_start=warm_start,
        past_discount=past_discount,
        rng=rng,
    )

