"""The logistic model and its derivatives: curve, MSE loss, gradient, Hessian.

All the calculus lives here so the optimisers never have to care how the model
is differentiated; they just ask for a gradient or a Hessian. Derivations are
in the report (Sections 2 to 4).
"""

from __future__ import annotations

import numpy as np


def logistic(x: np.ndarray, r: float, K: float, x0: float = 0.0) -> np.ndarray:
    """Logistic curve K / (1 + e^{-r(x-x0)}). r = steepness, K = saturation, x0 = midpoint (fixed)."""
    return K / (1.0 + np.exp(-r * (x - x0)))


def mse_loss(theta: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float = 0.0) -> float:
    """Mean-squared-error objective L(theta) for theta = (r, K)."""
    r, K = theta
    residual = logistic(x, r, K, x0) - y
    return float(np.mean(residual ** 2))


def analytic_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float = 0.0) -> np.ndarray:
    """Closed-form gradient of the MSE loss w.r.t. (r, K).

    With sigmoid s_i = 1 / (1 + e^{-r(x_i-x0)}) and residual g_i = K*s_i - y_i:

        dL/dK = (2/n) sum_i  g_i s_i
        dL/dr = (2/n) sum_i  g_i K (x_i-x0) s_i (1-s_i)

    The s(1-s) term is what kills sensitivity to r in the tails; if the data
    don't cover the transition region, r is barely identifiable (report S4).
    """
    r, K = theta
    z = x - x0
    s = 1.0 / (1.0 + np.exp(-r * z))            # sigmoid factor s_i
    residual = K * s - y                        # g_i = f_i - y_i
    d_K = 2.0 * np.mean(residual * s)
    d_r = 2.0 * np.mean(residual * K * z * s * (1.0 - s))
    return np.array([d_r, d_K])


def numerical_hessian(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float = 0.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """Finite-difference 2x2 Hessian (four-point stencil per entry).

    Only 2D here, so this is cheap and accurate enough to drive the Newton step,
    and it keeps the second-order maths out of the optimiser. Analytic would
    work too; this was just the pragmatic call.
    """
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i = np.zeros(2)
            e_j = np.zeros(2)
            e_i[i] = eps
            e_j[j] = eps
            f_pp = mse_loss(theta + e_i + e_j, x, y, x0)
            f_pm = mse_loss(theta + e_i - e_j, x, y, x0)
            f_mp = mse_loss(theta - e_i + e_j, x, y, x0)
            f_mm = mse_loss(theta - e_i - e_j, x, y, x0)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
    return H
