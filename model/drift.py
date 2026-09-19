from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SHIFT_POLICY = {
    "version": "crix-distribution-shift-v1",
    "adversarialAuc": {"warning": 0.65, "material": 0.75},
    "psi": {"warning": 0.10, "material": 0.25},
    "jensenShannon": {"warning": 0.05, "material": 0.10},
    "supportBreachRate": {"warning": 0.05, "material": 0.10},
    "minimumRows": 500,
    "diagnosticOnly": True,
    "automaticRetraining": False,
    "automaticPromotion": False,
}

_STATUS_ORDER = {
    "pass": 0,
    "warning": 1,
    "material-shift": 2,
    "insufficient-data": -1,
}


def _status_from_value(value: float, thresholds: Mapping[str, float]) -> str:
    if value >= float(thresholds["material"]):
        return "material-shift"
    if value >= float(thresholds["warning"]):
        return "warning"
    return "pass"


def _combine_status(statuses: Sequence[str]) -> str:
    supported = [status for status in statuses if status != "insufficient-data"]
    if not supported:
        return "insufficient-data"
    return max(supported, key=lambda status: _STATUS_ORDER[status])


def _finite(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float).reshape(-1)
    return raw[np.isfinite(raw)]


def _histogram_edges(expected: np.ndarray, bins: int = 10) -> np.ndarray:
    finite = _finite(expected)
    if finite.size == 0:
        raise ValueError("shift metric baseline contains no finite values")
    points = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))
    if len(points) == 1:
        pivot = points[0]
        return np.array([-np.inf, pivot, np.nextafter(pivot, np.inf), np.inf])
    return np.concatenate(([-np.inf], points[1:-1], [np.inf]))


def population_stability_index_safe(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float | None:
    baseline = _finite(expected)
    current = _finite(actual)
    if baseline.size < 2 or current.size < 2:
        return None
    edges = _histogram_edges(baseline, bins=bins)
    e, _ = np.histogram(baseline, bins=edges)
    a, _ = np.histogram(current, bins=edges)
    epsilon = 1e-6
    e_share = np.maximum(e / max(1, e.sum()), epsilon)
    a_share = np.maximum(a / max(1, a.sum()), epsilon)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def jensen_shannon_divergence(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float | None:
    baseline = _finite(expected)
    current = _finite(actual)
    if baseline.size < 2 or current.size < 2:
        return None
    edges = _histogram_edges(baseline, bins=bins)
    e, _ = np.histogram(baseline, bins=edges)
    a, _ = np.histogram(current, bins=edges)
    epsilon = 1e-12
    p = (e.astype(float) + epsilon) / (e.sum() + epsilon * len(e))
    q = (a.astype(float) + epsilon) / (a.sum() + epsilon * len(a))
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _feature_shift(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    baseline_raw = np.asarray(expected, dtype=float).reshape(-1)
    actual_raw = np.asarray(actual, dtype=float).reshape(-1)
    baseline = _finite(baseline_raw)
    current = _finite(actual_raw)
    if baseline.size < 20 or current.size < 20:
        return {
            "status": "insufficient-data",
            "expectedCount": int(baseline.size),
            "actualCount": int(current.size),
        }

    p01, p10, p50, p90, p99 = np.quantile(baseline, [0.01, 0.10, 0.50, 0.90, 0.99])
    current_q10, current_q50, current_q90 = np.quantile(current, [0.10, 0.50, 0.90])
    psi = population_stability_index_safe(baseline, current)
    js = jensen_shannon_divergence(baseline, current)
    assert psi is not None and js is not None
    breach = float(np.mean((current < p01) | (current > p99)))
    missing_expected = 1.0 - baseline.size / len(baseline_raw)
    missing_actual = 1.0 - current.size / len(actual_raw)
    statuses = [
        _status_from_value(psi, SHIFT_POLICY["psi"]),
        _status_from_value(js, SHIFT_POLICY["jensenShannon"]),
        _status_from_value(breach, SHIFT_POLICY["supportBreachRate"]),
    ]
    return {
        "status": _combine_status(statuses),
        "psi": psi,
        "jensenShannon": js,
        "wasserstein": float(wasserstein_distance(baseline, current)),
        "support": {
            "trainingP01": float(p01),
            "trainingP99": float(p99),
            "actualBreachRate": breach,
        },
        "missingness": {
            "expectedRate": float(missing_expected),
            "actualRate": float(missing_actual),
            "absoluteShift": float(abs(missing_actual - missing_expected)),
        },
        "quantileMovement": {
            "p10": {"expected": float(p10), "actual": float(current_q10)},
            "p50": {"expected": float(p50), "actual": float(current_q50)},
            "p90": {"expected": float(p90), "actual": float(current_q90)},
        },
        "expectedCount": int(baseline.size),
        "actualCount": int(current.size),
    }


def _balanced_sample(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    seed: int,
    max_rows_per_population: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(len(expected), len(actual), max_rows_per_population)
    if n <= 0:
        return np.empty((0, expected.shape[1]), dtype=float), np.empty((0,), dtype=np.int8)
    left = rng.choice(len(expected), size=n, replace=False)
    right = rng.choice(len(actual), size=n, replace=False)
    X = np.vstack((expected[left], actual[right])).astype(float, copy=False)
    y = np.concatenate((np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)))
    return X, y


def adversarial_validation(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    seed: int = 42,
    folds: int = 5,
    max_rows_per_population: int = 20_000,
    minimum_rows_per_population: int = 500,
) -> dict[str, object]:
    baseline = np.asarray(expected, dtype=float)
    current = np.asarray(actual, dtype=float)
    if baseline.ndim != 2 or current.ndim != 2 or baseline.shape[1] != current.shape[1]:
        raise ValueError("adversarial validation requires aligned two-dimensional feature matrices")
    if baseline.shape[0] < minimum_rows_per_population or current.shape[0] < minimum_rows_per_population:
        return {
            "status": "insufficient-data",
            "auc": None,
            "rowsPerPopulation": min(len(baseline), len(current)),
            "minimumRowsPerPopulation": minimum_rows_per_population,
        }

    X, y = _balanced_sample(
        baseline,
        current,
        seed=seed,
        max_rows_per_population=max_rows_per_population,
    )
    if not np.isfinite(X).all():
        raise ValueError("adversarial validation feature matrices must be finite")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.20,
                    solver="lbfgs",
                    max_iter=300,
                    random_state=seed,
                ),
            ),
        ]
    )
    probability = cross_val_predict(
        estimator,
        X,
        y,
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    auc = float(roc_auc_score(y, probability))
    return {
        "status": _status_from_value(auc, SHIFT_POLICY["adversarialAuc"]),
        "auc": auc,
        "folds": folds,
        "seed": seed,
        "rowsPerPopulation": int(len(y) // 2),
        "classifier": "standardized-logistic-regression-c0.20",
        "interpretation": "Diagnostic population distinguishability only; not proof of model invalidity.",
    }


def _segment_evidence(
    expected: np.ndarray,
    actual: np.ndarray,
    feature_names: Sequence[str],
    expected_labels: np.ndarray,
    actual_labels: np.ndarray,
    *,
    min_rows: int,
) -> list[dict[str, object]]:
    left = np.asarray(expected_labels).astype(str)
    right = np.asarray(actual_labels).astype(str)
    if len(left) != len(expected) or len(right) != len(actual):
        raise ValueError("segment labels must align to feature matrices")
    rows: list[dict[str, object]] = []
    for label in sorted(set(left) | set(right)):
        left_mask = left == label
        right_mask = right == label
        left_count = int(left_mask.sum())
        right_count = int(right_mask.sum())
        if left_count < min_rows or right_count < min_rows:
            rows.append(
                {
                    "segment": label,
                    "status": "insufficient-data",
                    "expectedCount": left_count,
                    "actualCount": right_count,
                }
            )
            continue
        feature_rows = {
            name: _feature_shift(expected[left_mask, index], actual[right_mask, index])
            for index, name in enumerate(feature_names)
        }
        rows.append(
            {
                "segment": label,
                "status": _combine_status([str(item["status"]) for item in feature_rows.values()]),
                "expectedCount": left_count,
                "actualCount": right_count,
                "features": feature_rows,
            }
        )
    return rows


def build_distribution_shift_evidence(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    feature_names: Sequence[str],
    expected_dates: np.ndarray,
    actual_dates: np.ndarray,
    expected_segments: Mapping[str, np.ndarray] | None = None,
    actual_segments: Mapping[str, np.ndarray] | None = None,
    seed: int = 42,
) -> dict[str, object]:
    baseline = np.asarray(expected, dtype=float)
    current = np.asarray(actual, dtype=float)
    if baseline.ndim != 2 or current.ndim != 2:
        raise ValueError("distribution shift evidence requires two-dimensional feature matrices")
    if baseline.shape[1] != len(feature_names) or current.shape[1] != len(feature_names):
        raise ValueError("feature names do not match shift matrices")

    train_dates = np.asarray(expected_dates, dtype="datetime64[ns]").reshape(-1)
    oot_dates = np.asarray(actual_dates, dtype="datetime64[ns]").reshape(-1)
    if len(train_dates) != len(baseline) or len(oot_dates) != len(current):
        raise ValueError("date arrays must align to shift matrices")
    if np.isnat(train_dates).any() or np.isnat(oot_dates).any():
        raise ValueError("shift evidence dates must be complete")
    if np.max(train_dates) >= np.min(oot_dates):
        raise ValueError("chronological OOT boundary is mixed or overlapping")

    feature_rows = {
        name: _feature_shift(baseline[:, index], current[:, index])
        for index, name in enumerate(feature_names)
    }
    adversarial = adversarial_validation(baseline, current, seed=seed)
    statuses = [str(item["status"]) for item in feature_rows.values()] + [str(adversarial["status"])]

    segment_output: dict[str, object] = {}
    for name in sorted(set(expected_segments or {}) | set(actual_segments or {})):
        if name not in (expected_segments or {}) or name not in (actual_segments or {}):
            raise ValueError(f"segment definition missing on one population: {name}")
        segment_output[name] = _segment_evidence(
            baseline,
            current,
            feature_names,
            np.asarray((expected_segments or {})[name]),
            np.asarray((actual_segments or {})[name]),
            min_rows=int(SHIFT_POLICY["minimumRows"]),
        )

    return {
        "schemaVersion": 1,
        "policy": SHIFT_POLICY,
        "status": _combine_status(statuses),
        "adversarialValidation": adversarial,
        "features": feature_rows,
        "segments": segment_output,
        "chronology": {
            "expectedEnd": str(np.max(train_dates).astype("datetime64[D]")),
            "actualStart": str(np.min(oot_dates).astype("datetime64[D]")),
            "overlap": False,
        },
        "privacy": "aggregate-only; no borrower rows, IDs or raw feature vectors",
        "action": "review-only; no automatic retraining or champion promotion",
    }
