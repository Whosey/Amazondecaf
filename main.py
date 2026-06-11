import argparse
import json

from decaf.trainer import run_sweep, train_once


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal DECAF reproduction runner.")
    parser.add_argument(
        "--env",
        choices=["biaseddm", "joballoc", "job", "matthew", "plant"],
        default="biaseddm",
    )
    parser.add_argument("--method", choices=["jo", "so", "fo", "fen", "soto"], default="so")
    parser.add_argument("--methods", default=None, help="Comma-separated methods for a sweep.")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--betas", default=None, help="Comma-separated beta values for a sweep.")
    parser.add_argument(
        "--fairness",
        choices=["variance", "alpha", "ggf", "maximin"],
        default="variance",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None, help="Override episode length.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds for a sweep.")
    parser.add_argument("--target-every", type=int, default=20)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="results/decaf_results.csv")
    parser.add_argument("--aggregate-output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.betas or args.methods:
        args.betas = args.betas or str(args.beta)
        args.methods = args.methods or args.method
        run_sweep(args)
        print(f"saved {args.output}")
        return

    result = train_once(
        env_name=args.env,
        method=args.method,
        beta=args.beta,
        episodes=args.episodes,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        episode_steps=args.steps,
        target_every=args.target_every,
        train_every=args.train_every,
        batch_size=args.batch_size,
        fairness_metric=args.fairness,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
