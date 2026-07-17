"""Non-CHC dynamics baselines for Tracks A/B (the tree surrogate lives in ``chc.surrogate``).

``LinearFitDynamics`` is the DLM / linear-state-space competitor from the strategy matrix: a
least-squares linear next-state model ``x_{t+1} = A x_t + B u_t + c`` fit from transitions. For a
fully-observed state this is what a Kalman DLM reduces to. It captures the linear part without any
physics knowledge, so it is a fair "data-driven linear model" baseline -- competitive on one-step
error, but it misses nonlinearity and drifts over a rollout (unlike the hybrid model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearFitDynamics:
    """Least-squares linear next-state model ``[x, u, 1] @ coef`` (DLM / state-space baseline)."""

    _coef: np.ndarray | None = field(default=None, init=False, repr=False)

    def _features(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x, u = np.atleast_2d(np.asarray(x, float)), np.atleast_2d(np.asarray(u, float))
        return np.column_stack([x, u, np.ones((x.shape[0], 1))])

    def fit(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> LinearFitDynamics:
        coef, *_ = np.linalg.lstsq(self._features(x, u), np.asarray(x_next, float), rcond=None)
        self._coef = coef
        return self

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("call fit() before predict()")
        return self._features(x, u) @ self._coef

    def rollout(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, float)
        us = np.asarray(us, float)
        states = [x]
        for t in range(us.shape[0]):
            x = self.predict(x[None, :], us[t][None, :])[0]
            states.append(x)
        return np.stack(states)
