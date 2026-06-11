from itertools import product

import numpy as np


def _action_resource_index(consumption):
    active = np.flatnonzero(consumption > 1e-9)
    if len(active) == 0:
        return -1
    if len(active) == 1 and abs(consumption[active[0]] - 1.0) <= 1e-9:
        return int(active[0])
    return None


def _solve_one_hot_dp(q_values, consumptions, resources, min_resources):
    k_resources = len(resources)
    full_limit_mask = 0
    for idx, available in enumerate(resources):
        if available >= 1.0:
            full_limit_mask |= 1 << idx

    min_mask = 0
    for idx, required in enumerate(min_resources):
        if required >= 1.0:
            min_mask |= 1 << idx

    states = {0: (0.0, [])}
    for agent_idx, agent_q in enumerate(q_values):
        next_states = {}
        for mask, (score, choice) in states.items():
            for action_idx, q_value in enumerate(agent_q):
                resource_idx = _action_resource_index(consumptions[agent_idx][action_idx])
                if resource_idx is None:
                    return None
                if resource_idx < 0:
                    next_mask = mask
                else:
                    resource_bit = 1 << resource_idx
                    if (full_limit_mask & resource_bit) == 0 or (mask & resource_bit):
                        continue
                    next_mask = mask | resource_bit
                next_score = score + float(q_value)
                if next_mask not in next_states or next_score > next_states[next_mask][0]:
                    next_states[next_mask] = (next_score, choice + [action_idx])
        states = next_states

    feasible = [
        (score, choice)
        for mask, (score, choice) in states.items()
        if (mask & min_mask) == min_mask
    ]
    if not feasible:
        return None
    return max(feasible, key=lambda item: item[0])[1]


def solve_allocation(q_values, consumptions, resources, min_resources=None):
    """Brute-force the small DECA allocation ILP.

    Each agent must receive one action. Resource use must not exceed resources.
    This is intentionally simple and dependency-free for the paper's toy domains.
    """
    best_score = -np.inf
    best_choice = None
    resource_array = np.asarray(resources, dtype=float)
    min_resource_array = (
        np.zeros_like(resource_array)
        if min_resources is None
        else np.asarray(min_resources, dtype=float)
    )

    if len(resource_array) <= 16 and np.all(resource_array <= 1.0 + 1e-9):
        dp_choice = _solve_one_hot_dp(
            q_values, consumptions, resource_array, min_resource_array
        )
        if dp_choice is not None:
            return dp_choice

    ranges = [range(len(values)) for values in q_values]
    for choice in product(*ranges):
        used = np.zeros_like(resource_array)
        score = 0.0
        for agent_idx, action_idx in enumerate(choice):
            used += consumptions[agent_idx][action_idx]
            score += q_values[agent_idx][action_idx]
        if np.any(used > resource_array + 1e-9):
            continue
        if np.any(used < min_resource_array - 1e-9):
            continue
        if score > best_score:
            best_score = score
            best_choice = choice

    if best_choice is None:
        raise RuntimeError("No feasible allocation found.")
    return list(best_choice)
