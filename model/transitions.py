from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

DEFAULT_STATES = ("CURRENT", "DPD_1_29", "DPD_30_59", "DPD_60_89", "DPD_90_PLUS", "DEFAULT")


def map_repayment_status(code: object) -> str | None:
    if code is None:
        return None
    try:
        value = float(code)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    if value <= 0:
        return "CURRENT"
    if value < 2:
        return "DPD_1_29"
    if value < 3:
        return "DPD_30_59"
    if value < 4:
        return "DPD_60_89"
    return "DPD_90_PLUS"


def transition_counts(sequences: Iterable[Sequence[str | None]], states: Sequence[str] = DEFAULT_STATES) -> np.ndarray:
    state_index = {state: index for index, state in enumerate(states)}
    if len(state_index) != len(states):
        raise ValueError("transition states must be unique")
    counts = np.zeros((len(states), len(states)), dtype=np.int64)
    for sequence in sequences:
        for current, following in zip(sequence[:-1], sequence[1:]):
            if current is None or following is None:
                continue
            if current not in state_index or following not in state_index:
                raise ValueError(f"unknown transition state: {current} -> {following}")
            counts[state_index[current], state_index[following]] += 1
    return counts


def transition_matrix(
    counts: np.ndarray,
    states: Sequence[str] = DEFAULT_STATES,
    *,
    absorbing: set[str] | None = None,
) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.shape != (len(states), len(states)) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("transition counts must be a finite non-negative square state matrix")
    absorbing = absorbing or set()
    matrix = np.zeros_like(values)
    for index, state in enumerate(states):
        if state in absorbing:
            matrix[index, index] = 1.0
            continue
        total = values[index].sum()
        if total <= 0:
            raise ValueError(f"transition state has no observed outgoing support: {state}")
        matrix[index] = values[index] / total
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("transition probabilities must sum to one")
    return matrix


def propagate_distribution(matrix: np.ndarray, initial: np.ndarray, months: int) -> np.ndarray:
    probabilities = np.asarray(matrix, dtype=float)
    distribution = np.asarray(initial, dtype=float).reshape(-1)
    if months < 0 or probabilities.shape != (len(distribution), len(distribution)):
        raise ValueError("invalid transition propagation dimensions/horizon")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("invalid stochastic transition matrix")
    if not np.isfinite(distribution).all() or (distribution < 0).any() or not np.isclose(distribution.sum(), 1.0, atol=1e-12):
        raise ValueError("initial distribution must be a probability vector")
    return distribution @ np.linalg.matrix_power(probabilities, months)
