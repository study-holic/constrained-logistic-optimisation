"""Builds the synthetic problem and runs all three methods.

The only module that sees the comparison as a whole. Returns plain dicts, so
the caller can print, plot, or sweep without that logic living here.
"""

from __future__ import annotations

import numpy as np

from model import logistic
from optimisers import OPTIMISERS, Config


def make_data(config: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate inputs and noisy observations from the ground-truth curve."""
    x = np.linspace(config.x_range[0], config.x_range[1], config.n)
    clean = logistic(x, config.r_true, config.K_true, config.x0)
    y = clean + rng.normal(0.0, config.sigma, size=config.n)
    return x, y


def run_study(config: Config) -> dict[str, dict]:
    """Run every optimiser on the same dataset.

    Data and the SGD draw use separately seeded generators, so it's
    reproducible and every method sees identical data.
    """
    data_rng = np.random.default_rng(config.seed)
    x, y = make_data(config, data_rng)

    theta0 = np.array(config.theta0, dtype=float)
    results: dict[str, dict] = {}
    for offset, (name, optimise) in enumerate(OPTIMISERS.items(), start=1):
        method_rng = np.random.default_rng(config.seed + offset)
        theta, history = optimise(theta0, x, y, config, method_rng)
        results[name] = {
            "theta": theta,
            "history": history,
            "final_loss": history[-1],
        }
    return results


def summarise(results: dict[str, dict]) -> str:
    """Format a compact final-loss table (mirrors Table 2 of the report)."""
    width = max(len(name) for name in results)
    lines = [f"{'Method':<{width}}  Final loss   (r, K)"]
    for name, record in results.items():
        r, K = record["theta"]
        lines.append(
            f"{name:<{width}}  {record['final_loss']:.5f}     "
            f"({r:.3f}, {K:.3f})"
        )
    return "\n".join(lines)
