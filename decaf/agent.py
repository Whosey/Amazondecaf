import numpy as np

from .network import MLP
from .solver import solve_allocation


def flatten_candidates(candidate_lists):
    features = [np.stack([c.features for c in candidates]) for candidates in candidate_lists]
    consumptions = [
        np.stack([c.consumption for c in candidates]) for candidates in candidate_lists
    ]
    priors = [np.array([c.utility_prior for c in candidates]) for candidates in candidate_lists]
    actions = [[c.action for c in candidates] for candidates in candidate_lists]
    return features, consumptions, priors, actions


def _metric_pair_from_features(features):
    if len(features) == 5:
        return float(features[3]), float(features[4])
    if len(features) == 7:
        return float(features[4]), float(features[5])
    if len(features) == 10:
        return float(features[8]), float(features[9])
    if len(features) == 11:
        return float(features[9]), float(features[10])
    if len(features) == 15:
        return float(features[4]), float(features[5])
    return 0.0, 0.0


def _uses_resource(consumption):
    return float(np.sum(consumption)) > 1e-9


class DecafAgent:
    def __init__(
        self,
        method,
        input_dim,
        beta=0.5,
        gamma=0.99,
        lr=3e-4,
        seed=0,
    ):
        self.method = method.lower()
        self.beta = beta
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        if self.method == "jo":
            self.q = MLP(input_dim, lr=lr, seed=seed)
            self.target_q = self.q.clone()
        elif self.method == "so":
            self.u = MLP(input_dim, lr=lr, seed=seed)
            self.f = MLP(input_dim, lr=lr, seed=seed + 1)
            self.target_u = self.u.clone()
            self.target_f = self.f.clone()
        elif self.method == "fo":
            self.f = MLP(input_dim, lr=lr, seed=seed + 1)
            self.target_f = self.f.clone()
        elif self.method in {"fen", "soto"}:
            pass
        else:
            raise ValueError("method must be one of: jo, so, fo, fen, soto")

    def q_values(self, features, priors=None, target=False):
        values = []
        for i, feats in enumerate(features):
            if self.method == "jo":
                net = self.target_q if target else self.q
                q = net.predict(feats)
            elif self.method == "so":
                u_net = self.target_u if target else self.u
                f_net = self.target_f if target else self.f
                q = (1.0 - self.beta) * u_net.predict(feats) + self.beta * f_net.predict(feats)
            elif self.method == "fo":
                f_net = self.target_f if target else self.f
                q = (1.0 - self.beta) * priors[i] + self.beta * f_net.predict(feats)
            elif self.method == "soto":
                q = self._soto_scores(i, feats, priors[i])
            else:
                q = self._fen_scores(i, feats, priors[i])
            values.append(q)
        return values

    def _soto_scores(self, agent_idx, feats, priors):
        scores = []
        for action_idx, feat in enumerate(feats):
            z_i, z_mean = _metric_pair_from_features(feat)
            consumes = 1.0 if priors[action_idx] > 0.0 else 0.0
            welfare_pressure = max(0.0, z_mean - z_i) + 1.0 / (1.0 + max(0.0, z_i))
            scores.append((1.0 - self.beta) * priors[action_idx] + self.beta * consumes * welfare_pressure)
        return np.asarray(scores, dtype=float)

    def _fen_scores(self, agent_idx, feats, priors):
        scores = []
        for action_idx, feat in enumerate(feats):
            z_i, z_mean = _metric_pair_from_features(feat)
            consumes = 1.0 if priors[action_idx] > 0.0 else 0.0
            disadvantaged = z_i < z_mean
            gate = self.beta if disadvantaged else 0.25 * self.beta
            fair_priority = consumes * (max(0.0, z_mean - z_i) + 0.1)
            scores.append((1.0 - gate) * priors[action_idx] + gate * fair_priority)
        return np.asarray(scores, dtype=float)

    def select(self, candidate_lists, resources, min_resources=None, epsilon=0.0):
        features, consumptions, priors, actions = flatten_candidates(candidate_lists)
        if self.rng.random() < epsilon:
            q_values = [self.rng.random(len(candidates)) for candidates in candidate_lists]
        else:
            q_values = self.q_values(features, priors=priors)
        indices = solve_allocation(q_values, consumptions, resources, min_resources)
        chosen_actions = [actions[i][idx] for i, idx in enumerate(indices)]
        chosen_features = np.stack([features[i][idx] for i, idx in enumerate(indices)])
        return chosen_actions, chosen_features

    def best_next_features(self, candidate_lists, resources, min_resources=None):
        features, consumptions, priors, _ = flatten_candidates(candidate_lists)
        q_values = self.q_values(features, priors=priors)
        indices = solve_allocation(q_values, consumptions, resources, min_resources)
        return np.stack([features[i][idx] for i, idx in enumerate(indices)])

    def update_targets(self):
        if self.method == "jo":
            self.target_q.copy_from(self.q)
        elif self.method == "so":
            self.target_u.copy_from(self.u)
            self.target_f.copy_from(self.f)
        elif self.method == "fo":
            self.target_f.copy_from(self.f)

    def train_batch(self, batch):
        current_x = np.concatenate([item["chosen_features"] for item in batch], axis=0)
        utility_r = np.concatenate([item["utility"] for item in batch], axis=0)
        fair_r = np.concatenate([item["fair"] for item in batch], axis=0)
        next_x = np.concatenate(
            [
                self.best_next_features(
                    item["next_candidates"], item["next_resources"], item["next_min_resources"]
                )
                for item in batch
            ],
            axis=0,
        )

        if self.method == "jo":
            next_value = self.target_q.predict(next_x)
            target = (1.0 - self.beta) * utility_r + self.beta * fair_r + self.gamma * next_value
            return {"jo_loss": self.q.train_mse(current_x, target)}

        if self.method == "so":
            u_target = utility_r + self.gamma * self.target_u.predict(next_x)
            f_target = fair_r + self.gamma * self.target_f.predict(next_x)
            return {
                "u_loss": self.u.train_mse(current_x, u_target),
                "f_loss": self.f.train_mse(current_x, f_target),
            }

        if self.method == "fo":
            f_target = fair_r + self.gamma * self.target_f.predict(next_x)
            return {"f_loss": self.f.train_mse(current_x, f_target)}

        return {}
