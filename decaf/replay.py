from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=250000):
        self.items = deque(maxlen=capacity)

    def add(self, item):
        self.items.append(item)

    def __len__(self):
        return len(self.items)

    def sample(self, rng, batch_size):
        batch_size = min(batch_size, len(self.items))
        indices = rng.choice(len(self.items), size=batch_size, replace=False)
        return [self.items[int(i)] for i in indices]

