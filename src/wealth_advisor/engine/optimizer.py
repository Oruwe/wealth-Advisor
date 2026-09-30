"""Execution layer: cvxpy-backed mean-variance optimizer bounded by fiduciary caps.

All portfolio math lives here. The strategic agent supplies two scalar parameters
(risk aversion, max equity exposure); this module turns those plus market inputs
into deterministic asset weights. No language model is invoked in this layer.

Formulation (long-only Markowitz with an equity-cap constraint):

    maximize   mu^T w - (lambda / 2) * w^T Sigma w
    subject to sum(w) == 1
               w >= 0
               sum(w[equity_indices]) <= max_equity_exposure
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DEFAULT_SOLVER = "OSQP"
ACCEPTED_STATUSES = {"optimal", "optimal_inaccurate"}


class OptimizerError(RuntimeError):
    """Raised when the optimizer cannot produce a valid weight vector."""


class FiduciaryOptimizer:
    """Long-only mean-variance optimizer with a hard cap on aggregate equity weight."""

    def __init__(self, solver: str = DEFAULT_SOLVER) -> None:
        self._solver = solver

    def optimize_allocation(
        self,
        *,
        expected_returns: Sequence[float],
        covariance_matrix: Sequence[Sequence[float]],
        equity_indices: list[int],
        target_risk_aversion: float,
        max_equity_exposure: float,
    ) -> list[float]:
        mu = np.asarray(expected_returns, dtype=float)
        sigma = np.asarray(covariance_matrix, dtype=float)
        self._validate(mu, sigma, equity_indices, target_risk_aversion, max_equity_exposure)

        n = int(mu.size)
        weights_var = cp.Variable(n, nonneg=True)

        objective = cp.Maximize(
            mu @ weights_var
            - (target_risk_aversion / 2.0) * cp.quad_form(weights_var, cp.psd_wrap(sigma))
        )
        constraints: list[Any] = [cp.sum(weights_var) == 1.0]
        if equity_indices:
            constraints.append(cp.sum(weights_var[equity_indices]) <= max_equity_exposure)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self._solver)
        except Exception as exc:
            raise OptimizerError(f"cvxpy solver {self._solver} failed: {exc}") from exc

        if problem.status not in ACCEPTED_STATUSES:
            raise OptimizerError(f"cvxpy returned status {problem.status!r}")

        raw = weights_var.value
        if raw is None:
            raise OptimizerError("cvxpy returned no weight vector")

        # Numerical slack: solvers can emit tiny negatives; clip and renormalize.
        clipped = np.clip(np.asarray(raw, dtype=float), 0.0, None)
        total = float(clipped.sum())
        if total <= 0.0:
            raise OptimizerError("optimizer produced an all-zero weight vector")
        normalized = clipped / total

        equity_weight = (
            float(normalized[equity_indices].sum()) if equity_indices else 0.0
        )
        logger.info(
            "optimization complete",
            extra={
                "status": problem.status,
                "target_risk_aversion": target_risk_aversion,
                "max_equity_exposure": max_equity_exposure,
                "equity_weight": equity_weight,
            },
        )
        return [float(x) for x in normalized]

    @staticmethod
    def _validate(
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        equity_indices: list[int],
        target_risk_aversion: float,
        max_equity_exposure: float,
    ) -> None:
        if mu.ndim != 1:
            raise OptimizerError(f"expected_returns must be 1-D, got shape {mu.shape}")
        n = int(mu.size)
        if n == 0:
            raise OptimizerError("expected_returns is empty")
        if sigma.shape != (n, n):
            raise OptimizerError(
                f"covariance_matrix shape {sigma.shape} does not match {n} assets"
            )
        if not np.allclose(sigma, sigma.T, atol=1e-8):
            raise OptimizerError("covariance_matrix is not symmetric")
        for index in equity_indices:
            if not 0 <= index < n:
                raise OptimizerError(
                    f"equity_indices contains {index}, out of range for {n} assets"
                )
        if not 1.0 <= target_risk_aversion <= 10.0:
            raise OptimizerError(
                f"target_risk_aversion {target_risk_aversion} outside [1.0, 10.0]"
            )
        if not 0.0 <= max_equity_exposure <= 1.0:
            raise OptimizerError(
                f"max_equity_exposure {max_equity_exposure} outside [0.0, 1.0]"
            )
