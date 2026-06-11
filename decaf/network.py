import copy

import numpy as np


class MLP:
    """Tiny two-hidden-layer ReLU network with Adam, implemented in NumPy."""

    def __init__(self, input_dim, hidden_dim=20, lr=3e-4, seed=0):
        self.rng = np.random.default_rng(seed)
        scale = 0.15
        self.w1 = self.rng.normal(0, scale, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = self.rng.normal(0, scale, size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.w3 = self.rng.normal(0, scale, size=(hidden_dim, 1))
        self.b3 = np.zeros(1)
        self.lr = lr
        self.t = 0
        self.m = {name: np.zeros_like(value) for name, value in self.params().items()}
        self.v = {name: np.zeros_like(value) for name, value in self.params().items()}

    def params(self):
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
        }

    def clone(self):
        return copy.deepcopy(self)

    def copy_from(self, other):
        for name in self.params():
            getattr(self, name)[:] = getattr(other, name)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        y = h2 @ self.w3 + self.b3
        cache = (x, z1, h1, z2, h2)
        return y[:, 0], cache

    def predict(self, x):
        return self.forward(x)[0]

    def train_mse(self, x, target):
        pred, cache = self.forward(x)
        target = np.asarray(target, dtype=float)
        batch = max(1, x.shape[0])
        grad_y = (2.0 / batch) * (pred - target)[:, None]
        x, z1, h1, z2, h2 = cache

        grads = {}
        grads["w3"] = h2.T @ grad_y
        grads["b3"] = np.sum(grad_y, axis=0)
        grad_h2 = grad_y @ self.w3.T
        grad_z2 = grad_h2 * (z2 > 0.0)
        grads["w2"] = h1.T @ grad_z2
        grads["b2"] = np.sum(grad_z2, axis=0)
        grad_h1 = grad_z2 @ self.w2.T
        grad_z1 = grad_h1 * (z1 > 0.0)
        grads["w1"] = x.T @ grad_z1
        grads["b1"] = np.sum(grad_z1, axis=0)

        self._adam_step(grads)
        return float(np.mean((pred - target) ** 2))

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for name, grad in grads.items():
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - beta1**self.t)
            v_hat = self.v[name] / (1.0 - beta2**self.t)
            getattr(self, name)[:] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

