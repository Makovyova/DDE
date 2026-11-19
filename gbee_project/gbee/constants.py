"""Local continuous optimization inside the discrete graph-editing loop.

SciPy sees a fixed topology and independent real constant parameters.  This
module has no differentiable connection to the graph policy: REINFORCE learns
from the resulting scalar reward.  A local fit is never a global guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .dag import EvaluationError, ExpressionDAG, _matrix


def validate_regression_data(X: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Accept y of shape (N,) or (N,1), rejecting accidental broadcasting."""
    matrix = _matrix(X)
    if np.iscomplexobj(y):
        raise ValueError("y must contain real values")
    target = np.asarray(y, dtype=np.float64)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape != (len(matrix),) or not np.isfinite(target).all():
        raise ValueError("y must be finite and have shape (n_samples,) or (n_samples, 1)")
    return matrix, target


def _mse(
    graph: ExpressionDAG, X: NDArray[np.float64], y: NDArray[np.float64], constants: NDArray[np.float64] | None = None
) -> float:
    with np.errstate(over="raise", invalid="raise"):
        try:
            residual = graph.evaluate(X, constants) - y
            value = float(np.mean(np.square(residual)))
        except FloatingPointError as exc:
            raise EvaluationError("Residual MSE overflowed") from exc
    if not np.isfinite(value):
        raise EvaluationError("Residual MSE is nonfinite")
    return value


def residual_mse(graph: ExpressionDAG, X: ArrayLike, y: ArrayLike, penalty: float = 1e30) -> float:
    """MSE or a finite penalty for numerical equation failures, not bad data.

    Data-shape errors are deliberately propagated.  Valid finite losses are
    returned unchanged, including values greater than the configured penalty.
    """
    matrix, target = validate_regression_data(X, y)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be positive and finite")
    try:
        return _mse(graph, matrix, target)
    except EvaluationError:
        return float(penalty)


@dataclass(frozen=True, slots=True)
class ConstantFit:
    graph: ExpressionDAG
    initial_mse: float
    mse: float
    success: bool
    message: str
    nfev: int


class ConstantOptimizer:
    """Bounded-work local fitting that always retains the best finite iterate.

    ``maxiter`` is passed to SciPy and a separate objective-call budget limits
    line searches.  The initial point is always a candidate.  Restart seeds are
    deterministic per call; neither mutable graph state nor optimizer RNG state
    leaks between episodes.  ``success`` means SciPy reported convergence (or
    an already exact/no-constant finite fit); an improved non-converged result
    may still be returned, together with its diagnostic message.  When the
    initial equation is undefined, a candidate must improve its configured
    penalty before being accepted.  A huge finite residual must not look like
    a successful recovery of an invalid expression.
    """

    def __init__(
        self, method: str = "BFGS", maxiter: int = 60, restarts: int = 0, seed: int = 0, penalty: float = 1e30
    ) -> None:
        if method not in {"BFGS", "Nelder-Mead"}:
            raise ValueError("method must be 'BFGS' or 'Nelder-Mead'")
        if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if isinstance(restarts, bool) or not isinstance(restarts, Integral) or restarts < 0:
            raise ValueError("restarts must be a nonnegative integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("penalty must be positive and finite")
        self.method = method
        self.maxiter = int(maxiter)
        self.restarts = int(restarts)
        self.seed = int(seed)
        self.penalty = float(penalty)

    def fit(self, graph: ExpressionDAG, X: ArrayLike, y: ArrayLike) -> ConstantFit:
        matrix, target = validate_regression_data(X, y)
        initial = np.asarray([graph.nodes[i].value for i in graph.constant_ids], dtype=np.float64)
        try:
            initial_mse = _mse(graph, matrix, target)
            best_mse = initial_mse
            initial_valid = True
        except EvaluationError:
            initial_mse = self.penalty
            best_mse = self.penalty
            initial_valid = False
        if not len(initial):
            return ConstantFit(
                graph,
                initial_mse,
                initial_mse,
                initial_valid,
                "No constant parameters" if initial_valid else "Equation is numerically invalid",
                1,
            )
        if initial_valid and initial_mse <= 1e-24:
            return ConstantFit(graph, initial_mse, initial_mse, True, "Initial constants already fit exactly", 1)

        best_vector = initial.copy()
        best_valid = initial_valid
        calls = 1
        converged = False
        messages: list[str] = []
        rng = np.random.default_rng(self.seed)
        # Finite differences scale with the number of free parameters.
        budget_per_start = max(32, self.maxiter * (len(initial) + 1) * 4)

        class BudgetExceeded(RuntimeError):
            pass

        for restart in range(self.restarts + 1):
            start = initial.copy() if restart == 0 else initial + rng.normal(0, 1, len(initial))
            start_calls = calls

            def objective(vector: NDArray[np.float64]) -> float:
                nonlocal calls, best_vector, best_mse, best_valid, converged
                if calls - start_calls >= budget_per_start:
                    raise BudgetExceeded
                calls += 1
                if not np.isfinite(vector).all():
                    return self.penalty
                try:
                    value = _mse(graph, matrix, target, vector)
                except EvaluationError:
                    return self.penalty
                if value < best_mse:
                    best_mse = value
                    best_vector = np.asarray(vector, dtype=np.float64).copy()
                    best_valid = True
                    converged = False
                return value

            options: dict[str, float | int] = {"maxiter": self.maxiter}
            if self.method == "BFGS":
                options["gtol"] = 1e-8
            else:
                options.update({"xatol": 1e-9, "fatol": 1e-12, "maxfev": budget_per_start})
            try:
                result = minimize(objective, start, method=self.method, options=options)
                # Flat invalid-domain penalties can satisfy SciPy's stopping
                # criterion.  Count convergence only at a valid finite point
                # that agrees with the retained best objective value.
                if result.success and np.isfinite(result.x).all():
                    calls += 1
                    try:
                        result_mse = _mse(graph, matrix, target, result.x)
                    except EvaluationError:
                        pass
                    else:
                        if best_valid and np.isclose(result_mse, best_mse, rtol=1e-6, atol=1e-12):
                            converged = True
                messages.append(str(result.message))
            except BudgetExceeded:
                messages.append("Objective evaluation budget exhausted")
            if best_valid and best_mse <= 1e-24:
                converged = True
                break

        if not best_valid:
            return ConstantFit(
                graph,
                initial_mse,
                self.penalty,
                False,
                "; ".join(messages) + "; no finite iterate improved the invalid-domain penalty",
                calls,
            )
        fitted = graph.with_constants(best_vector)
        # Re-evaluate the chosen parameters, never trusting optimizer.fun alone.
        calls += 1
        final_mse = _mse(fitted, matrix, target)
        if initial_valid and final_mse > initial_mse:
            fitted, final_mse = graph, initial_mse
        return ConstantFit(fitted, initial_mse, final_mse, converged, "; ".join(messages), calls)
