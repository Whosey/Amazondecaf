import csv
import json
from pathlib import Path

import numpy as np

from .agent import DecafAgent
from .envs import make_env
from .replay import ReplayBuffer


def linear_epsilon(episode, episodes, start=1.0, end=0.05):
    decay_episodes = max(1, episodes // 2)
    frac = min(1.0, episode / decay_episodes)
    return start + frac * (end - start)


def run_episode(env, agent, replay=None, epsilon=0.0, train_every=4, batch_size=64):
    candidates = env.reset()
    total_utility = np.zeros(env.n_agents, dtype=float)
    losses = []
    done = False
    step = 0
    last_info = {}

    while not done:
        chosen_actions, chosen_features = agent.select(
            candidates, env.resources, env.min_resources, epsilon=epsilon
        )
        utility, fair, next_candidates, done, last_info = env.step(chosen_actions)
        total_utility += utility
        if replay is not None:
            replay.add(
                {
                    "chosen_features": chosen_features,
                    "utility": utility,
                    "fair": fair,
                    "next_candidates": next_candidates,
                    "next_resources": env.resources.copy(),
                    "next_min_resources": env.min_resources.copy(),
                }
            )
            if len(replay) >= batch_size and step % train_every == 0:
                batch = replay.sample(agent.rng, batch_size)
                losses.append(agent.train_batch(batch))
        candidates = next_candidates
        step += 1

    return {
        "utility": float(np.sum(total_utility)),
        "agent_utility": total_utility.tolist(),
        "variance": float(last_info.get("variance", np.var(total_utility))),
        "fairness": float(last_info.get("fairness", -np.var(total_utility))),
        "losses": losses,
    }


def evaluate(
    env_name,
    agent,
    episodes=10,
    seed=10000,
    episode_steps=None,
    fairness_metric="variance",
):
    results = []
    for offset in range(episodes):
        env = make_env(
            env_name,
            seed=seed + offset,
            episode_steps=episode_steps,
            fairness_metric=fairness_metric,
        )
        results.append(run_episode(env, agent, epsilon=0.0))
    return {
        "utility_mean": float(np.mean([r["utility"] for r in results])),
        "utility_std": float(np.std([r["utility"] for r in results])),
        "fairness_mean": float(np.mean([r["fairness"] for r in results])),
        "fairness_std": float(np.std([r["fairness"] for r in results])),
        "variance_mean": float(np.mean([r["variance"] for r in results])),
        "variance_std": float(np.std([r["variance"] for r in results])),
        "episodes": episodes,
    }


def train_once(
    env_name,
    method,
    beta,
    episodes=200,
    seed=0,
    eval_episodes=10,
    episode_steps=None,
    target_every=20,
    train_every=4,
    batch_size=64,
    fairness_metric="variance",
):
    env = make_env(
        env_name, seed=seed, episode_steps=episode_steps, fairness_metric=fairness_metric
    )
    input_dim = len(env.reset()[0][0].features)
    agent = DecafAgent(method=method, input_dim=input_dim, beta=beta, seed=seed)
    replay = ReplayBuffer()

    history = []
    for episode in range(episodes):
        epsilon = linear_epsilon(episode, episodes)
        result = run_episode(
            env,
            agent,
            replay=replay,
            epsilon=epsilon,
            train_every=train_every,
            batch_size=batch_size,
        )
        if episode % target_every == 0:
            agent.update_targets()
        if (episode + 1) % max(1, episodes // 10) == 0:
            history.append(
                {
                    "episode": episode + 1,
                    "epsilon": epsilon,
                    "utility": result["utility"],
                    "fairness": result["fairness"],
                    "variance": result["variance"],
                }
            )

    metrics = evaluate(
        env_name,
        agent,
        episodes=eval_episodes,
        episode_steps=episode_steps,
        fairness_metric=fairness_metric,
    )
    return {
        "env": env_name,
        "fairness": fairness_metric,
        "method": method,
        "beta": beta,
        "episodes": episodes,
        "seed": seed,
        "metrics": metrics,
        "history": history,
    }


def _parse_seeds(seed_text, default_seed):
    if not seed_text:
        return [default_seed]
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def _aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = (row["env"], row["fairness"], row["method"], row["beta"], row["episodes"])
        groups.setdefault(key, []).append(row)

    aggregate = []
    for (env, fairness, method, beta, episodes), group in sorted(groups.items()):
        utility = np.array([row["utility_mean"] for row in group], dtype=float)
        fair = np.array([row["fairness_mean"] for row in group], dtype=float)
        variance = np.array([row["variance_mean"] for row in group], dtype=float)
        n = len(group)
        aggregate.append(
            {
                "env": env,
                "fairness": fairness,
                "method": method,
                "beta": beta,
                "episodes": episodes,
                "n_seeds": n,
                "utility_mean": float(np.mean(utility)),
                "utility_std": float(np.std(utility, ddof=1)) if n > 1 else 0.0,
                "utility_sem": float(np.std(utility, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "fairness_mean": float(np.mean(fair)),
                "fairness_std": float(np.std(fair, ddof=1)) if n > 1 else 0.0,
                "fairness_sem": float(np.std(fair, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "variance_mean": float(np.mean(variance)),
                "variance_std": float(np.std(variance, ddof=1)) if n > 1 else 0.0,
                "variance_sem": float(np.std(variance, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            }
        )
    return aggregate


def _write_csv(output, rows, fieldnames):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args):
    betas = [float(x) for x in args.betas.split(",")]
    methods = [x.strip().lower() for x in args.methods.split(",")]
    seeds = _parse_seeds(args.seeds, args.seed)
    results = []
    for method in methods:
        for beta in betas:
            for seed in seeds:
                result = train_once(
                    env_name=args.env,
                    method=method,
                    beta=beta,
                    episodes=args.episodes,
                    seed=seed,
                    eval_episodes=args.eval_episodes,
                    episode_steps=args.steps,
                    target_every=args.target_every,
                    train_every=args.train_every,
                    batch_size=args.batch_size,
                    fairness_metric=args.fairness,
                )
                print(
                    json.dumps(
                        result["metrics"]
                        | {"method": method, "beta": beta, "seed": seed},
                        indent=2,
                    )
                )
                results.append(result)

    rows = []
    for result in results:
        row = {
            "env": result["env"],
            "fairness": result["fairness"],
            "method": result["method"],
            "beta": result["beta"],
            "episodes": result["episodes"],
            "seed": result["seed"],
        }
        row.update(result["metrics"])
        row.pop("episodes", None)
        row["episodes"] = result["episodes"]
        rows.append(row)

    raw_fieldnames = [
        "env",
        "fairness",
        "method",
        "beta",
        "episodes",
        "seed",
        "utility_mean",
        "utility_std",
        "fairness_mean",
        "fairness_std",
        "variance_mean",
        "variance_std",
    ]
    _write_csv(args.output, rows, raw_fieldnames)

    summary_output = args.aggregate_output
    if summary_output is None and len(seeds) > 1:
        output_path = Path(args.output)
        summary_output = output_path.with_name(f"{output_path.stem}_summary{output_path.suffix}")
    if summary_output:
        summary_rows = _aggregate_rows(rows)
        _write_csv(
            summary_output,
            summary_rows,
            [
                "env",
                "fairness",
                "method",
                "beta",
                "episodes",
                "n_seeds",
                "utility_mean",
                "utility_std",
                "utility_sem",
                "fairness_mean",
                "fairness_std",
                "fairness_sem",
                "variance_mean",
                "variance_std",
                "variance_sem",
            ],
        )
        print(f"saved {summary_output}")
    return results
