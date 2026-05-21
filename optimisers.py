"""The three constrained optimisers, plus the shared Config.

All share one signature so experiment.py can just loop over them:

    optimise(theta0, x, y, config, rng) -> (theta_final, loss_history)

and all project onto {r, K >= floor} with the same operator. Only the step
rule differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import analytic_gradient, mse_loss, numerical_hessian

# Loss value per iteration.
LossHistory = list[float]
OptimiserResult = tuple[np.ndarray, LossHistory]


@dataclass(frozen=True)
class Config:
    """Experiment configuration (defaults reproduce Table 1 of the report)."""

    n: int = 120                       # number of data points
    x_range: tuple[float, float] = (-2.0, 2.0)
    r_true: float = 1.5                # ground-truth steepness
    K_true: float = 10.0               # ground-truth saturation
    x0: float = 0.0                    # fixed midpoint
    sigma: float = 0.3                 # Gaussian noise std
    theta0: tuple[float, float] = (0.5, 5.0)   # initial (r, K)
    eta: float = 0.05                  # GD / SGD step size
    batch: int = 12                    # SGD mini-batch size
    newton_damping: float = 1e-2       # lambda in (H + lambda I)
    floor: float = 1e-6               # positivity floor epsilon
    iters: int = 120                   # iteration budget per method
    seed: int = 0


def project(theta: np.ndarray, floor: float) -> np.ndarray:
    """Project onto the feasible set {r >= floor, K >= floor}."""
    return np.maximum(theta, floor)


def projected_gradient_descent(
    theta0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: Config,
    rng: np.random.Generator | None = None,
) -> OptimiserResult:
    """Projected gradient descent. Deterministic baseline; rng is unused."""
    theta = np.asarray(theta0, dtype=float).copy()
    history: LossHistory = []
    for _ in range(config.iters):
        grad = analytic_gradient(theta, x, y, config.x0)
        theta = project(theta - config.eta * grad, config.floor)
        history.append(mse_loss(theta, x, y, config.x0))
    return theta, history


def projected_sgd(
    theta0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> OptimiserResult:
    """Projected stochastic gradient descent over uniform mini-batches."""
    theta = np.asarray(theta0, dtype=float).copy()
    history: LossHistory = []
    n = x.shape[0]
    for _ in range(config.iters):
        idx = rng.choice(n, config.batch, replace=False)
        grad = analytic_gradient(theta, x[idx], y[idx], config.x0)
        theta = project(theta - config.eta * grad, config.floor)
        history.append(mse_loss(theta, x, y, config.x0))   # full-data loss for comparability
    return theta, history


def damped_newton(
    theta0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: Config,
    rng: np.random.Generator | None = None,
    armijo_c1: float = 1e-4,
) -> OptimiserResult:
    """Damped Newton with backtracking line search.

    Solve (H + lambda I) d = grad, then halve the step until Armijo is happy.
    The damping and line search aren't polish; without them a raw Newton step
    goes haywire wherever the curvature estimate is badly conditioned. (rng
    unused.)
    """
    theta = np.asarray(theta0, dtype=float).copy()
    history: LossHistory = []
    identity = np.eye(2)
    for _ in range(config.iters):
        grad = analytic_gradient(theta, x, y, config.x0)
        H = numerical_hessian(theta, x, y, config.x0)
        step = np.linalg.solve(H + config.newton_damping * identity, grad)

        # backtracking line search (Armijo)
        t = 1.0
        f0 = mse_loss(theta, x, y, config.x0)
        directional = float(grad @ step)
        while t > 1e-4:
            candidate = project(theta - t * step, config.floor)
            if mse_loss(candidate, x, y, config.x0) <= f0 - armijo_c1 * t * directional:
                theta = candidate
                break
            t *= 0.5
        history.append(mse_loss(theta, x, y, config.x0))
    return theta, history


# name -> optimiser; experiment.py iterates over this.
OPTIMISERS = {
    "Projected GD": projected_gradient_descent,
    "Projected SGD": projected_sgd,
    "Damped Newton-type": damped_newton,
}
