"""Reproduce the convergence study from the command line.

    python reproduce.py            # run with report defaults, print the table
    python reproduce.py --plot     # also save a log-scale loss plot
    python reproduce.py --seed 7   # different noise / batch draw

Matplotlib is imported lazily so the core study has no plotting dependency.
"""

from __future__ import annotations

import argparse

from experiment import run_study, summarise
from optimisers import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained logistic fitting study.")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--iters", type=int, default=120, help="iterations per method")
    parser.add_argument("--plot", action="store_true", help="save loss-trajectory plot")
    parser.add_argument("--out", default="figures/convergence.png", help="plot output path")
    return parser.parse_args()


def save_plot(results: dict[str, dict], out_path: str) -> None:
    import os

    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, record in results.items():
        ax.plot(range(1, len(record["history"]) + 1), record["history"], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss $L(\theta_k)$")
    ax.set_title("Convergence under shared constraints and iteration budget")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    args = parse_args()
    config = Config(seed=args.seed, iters=args.iters)
    results = run_study(config)
    print(summarise(results))
    if args.plot:
        save_plot(results, args.out)


if __name__ == "__main__":
    main()
