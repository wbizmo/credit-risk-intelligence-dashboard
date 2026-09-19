from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import genpareto

from evt import fit_pot_tail
from portfolio_risk import (
    _as_float_vector,
    _dependency_thresholds,
    _resolve_backend,
    _resolve_dependency,
    _simulate_loss_vector,
    _tail_contributions_reference,
    simulate_portfolio,
)


def _gpu_available() -> bool:
    try:
        import cupy as cp  # type: ignore
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


_GPU_AVAILABLE = _gpu_available()


def _reference_contributions(
    pd,
    lgd,
    ead,
    *,
    rho,
    scenarios,
    seed,
    chunk_size,
    quantiles,
    dependency_model=None,
):
    pd_v = _as_float_vector(pd, "pd")
    lgd_v = _as_float_vector(lgd, "lgd")
    ead_v = _as_float_vector(ead, "ead")
    spec = _resolve_dependency(pd_v.size, rho, dependency_model)
    thresholds = _dependency_thresholds(pd_v, spec)
    loss_given_default = lgd_v * ead_v
    losses = _simulate_loss_vector(
        loss_given_default=loss_given_default,
        thresholds=thresholds,
        spec=spec,
        scenarios=scenarios,
        chunk_size=chunk_size,
        seed=seed,
        backend=_resolve_backend("numpy"),
    )
    quantile_values = {
        float(q): float(np.quantile(losses, q, method="higher"))
        for q in quantiles
        if scenarios >= (100_000 if q >= 0.999 else max(1_000, int(np.ceil(100.0 / (1.0 - q)))))
    }
    reference = _tail_contributions_reference(
        losses=losses,
        quantile_values=quantile_values,
        loss_given_default=loss_given_default,
        thresholds=thresholds,
        spec=spec,
        scenarios=scenarios,
        chunk_size=chunk_size,
        seed=seed,
    )
    return losses, reference


class TailAttributionOptimizationTests(unittest.TestCase):
    def test_canonical_gaussian_digest_is_stable(self) -> None:
        pd = np.linspace(0.01, 0.20, 25)
        lgd = np.linspace(0.30, 0.75, 25)
        ead = np.linspace(1000, 5000, 25)
        result = simulate_portfolio(pd, lgd, ead, rho=0.18, scenarios=12000, seed=1234, chunk_size=127)
        self.assertEqual(
            result["lossDigest"],
            "553333f6f7f8c957af5d07939d68b11b86daa6cdf553ca914e6a7e84cd3c5849",
        )

    def test_shared_tail_replay_matches_reference_across_seeds_chunks_and_quantile_counts(self) -> None:
        pd = np.linspace(0.02, 0.18, 24)
        lgd = np.linspace(0.25, 0.70, 24)
        ead = np.linspace(500, 5000, 24)
        for seed in (3, 17):
            for chunk_size in (127, 2048):
                for quantiles in ((0.95,), (0.90, 0.95), (0.90, 0.95, 0.99)):
                    result = simulate_portfolio(
                        pd,
                        lgd,
                        ead,
                        rho=0.22,
                        scenarios=12000,
                        seed=seed,
                        chunk_size=chunk_size,
                        quantiles=quantiles,
                        tail_contributions=True,
                    )
                    _, reference = _reference_contributions(
                        pd,
                        lgd,
                        ead,
                        rho=0.22,
                        scenarios=12000,
                        seed=seed,
                        chunk_size=chunk_size,
                        quantiles=quantiles,
                    )
                    self.assertEqual(set(result["tailContributions"]), set(reference))
                    for key in reference:
                        np.testing.assert_allclose(
                            result["tailContributions"][key],
                            reference[key],
                            rtol=0,
                            atol=1e-10,
                        )

    def test_tail_contributions_remain_chunk_size_invariant(self) -> None:
        pd = np.linspace(0.015, 0.17, 30)
        lgd = np.linspace(0.25, 0.70, 30)
        ead = np.linspace(750, 6000, 30)
        kwargs = dict(
            rho=0.23,
            scenarios=12000,
            seed=404,
            quantiles=(0.90, 0.95, 0.99),
            tail_contributions=True,
        )
        small = simulate_portfolio(pd, lgd, ead, chunk_size=97, **kwargs)
        large = simulate_portfolio(pd, lgd, ead, chunk_size=4096, **kwargs)
        self.assertEqual(small["lossDigest"], large["lossDigest"])
        self.assertEqual(small["expectedLoss"], large["expectedLoss"])
        self.assertEqual(small["var"], large["var"])
        self.assertEqual(small["expectedShortfall"], large["expectedShortfall"])
        for key in small["tailContributions"]:
            np.testing.assert_allclose(
                small["tailContributions"][key],
                large["tailContributions"][key],
                rtol=0,
                atol=1e-10,
            )

    def test_tail_replay_handles_unsupported_zero_loss_and_concentrated_portfolios(self) -> None:
        zero_mix = simulate_portfolio(
            [0.03, 0.08, 0.15, 0.20],
            [0.0, 0.5, 0.0, 0.8],
            [1000, 0.0, 2500, 8000],
            rho=0.3,
            scenarios=12000,
            seed=9,
            quantiles=(0.95, 0.999),
            tail_contributions=True,
        )
        self.assertIn("0.95", zero_mix["tailContributions"])
        self.assertNotIn("0.999", zero_mix["tailContributions"])
        self.assertFalse(zero_mix["tailSupport"]["0.999"]["supported"])
        self.assertAlmostEqual(
            sum(zero_mix["tailContributions"]["0.95"]),
            zero_mix["expectedShortfall"]["0.95"],
            places=10,
        )

        concentrated = simulate_portfolio(
            [0.02, 0.03, 0.05, 0.08],
            [0.45] * 4,
            [100_000, 1000, 1000, 1000],
            rho=0.25,
            scenarios=12000,
            seed=31,
            quantiles=(0.95, 0.99),
            tail_contributions=True,
        )
        self.assertGreaterEqual(
            concentrated["tailContributions"]["0.99"][0],
            max(concentrated["tailContributions"]["0.99"][1:]),
        )
        self.assertFalse(concentrated["tailAttribution"]["fullScenarioObligorMatrixRetained"])


class BackendAndDependencyTests(unittest.TestCase):
    def test_numpy_is_default_and_cupy_is_lazy_optional(self) -> None:
        result = simulate_portfolio([0.05, 0.10], [0.4, 0.5], [1000, 2000], scenarios=2000)
        self.assertEqual(result["backend"]["name"], "numpy")
        self.assertIn("canonical", result["backend"]["reproducibilityClass"])

        with patch.dict(sys.modules, {"cupy": None}):
            with self.assertRaisesRegex(RuntimeError, "CuPy|CUDA"):
                simulate_portfolio([0.05], [0.5], [1000], scenarios=1000, backend="cupy")

    @unittest.skipUnless(_GPU_AVAILABLE, "optional CuPy/CUDA research test")
    def test_cupy_is_repeatable_and_statistically_agrees_with_numpy(self) -> None:
        pd = np.linspace(0.02, 0.15, 30)
        lgd = np.linspace(0.30, 0.65, 30)
        ead = np.linspace(500, 5000, 30)
        kwargs = dict(rho=0.2, scenarios=12000, seed=211, chunk_size=512, quantiles=(0.95, 0.99))
        first = simulate_portfolio(pd, lgd, ead, backend="cupy", **kwargs)
        second = simulate_portfolio(pd, lgd, ead, backend="cupy", **kwargs)
        cpu = simulate_portfolio(pd, lgd, ead, backend="numpy", **kwargs)

        self.assertEqual(first["lossDigest"], second["lossDigest"])
        self.assertEqual(first["expectedLoss"], second["expectedLoss"])
        self.assertEqual(first["backend"]["name"], "cupy")
        self.assertNotEqual(first["backend"]["device"], "cpu")
        self.assertLess(
            abs(first["expectedLoss"] - cpu["expectedLoss"]),
            max(1.0, 6 * cpu["monteCarlo"]["expectedLossStdError"]),
        )
        self.assertLess(
            abs(first["var"]["0.99"] - cpu["var"]["0.99"]),
            max(1.0, 0.15 * cpu["var"]["0.99"]),
        )

    def test_explicit_gaussian_dependency_is_backward_compatible(self) -> None:
        pd = np.linspace(0.02, 0.16, 20)
        lgd = np.linspace(0.3, 0.7, 20)
        ead = np.linspace(1000, 4000, 20)
        legacy = simulate_portfolio(pd, lgd, ead, rho=0.19, scenarios=10000, seed=44, chunk_size=333)
        explicit = simulate_portfolio(
            pd,
            lgd,
            ead,
            rho=0.01,
            dependency_model={"name": "gaussian", "rho": 0.19},
            scenarios=10000,
            seed=44,
            chunk_size=333,
        )
        self.assertEqual(legacy["lossDigest"], explicit["lossDigest"])
        self.assertEqual(legacy["expectedLoss"], explicit["expectedLoss"])
        self.assertEqual(legacy["var"], explicit["var"])

    def test_student_t_and_low_rank_models_are_seed_and_chunk_deterministic(self) -> None:
        n = 18
        pd = np.linspace(0.02, 0.18, n)
        lgd = np.linspace(0.3, 0.65, n)
        ead = np.linspace(500, 3500, n)
        loadings = np.column_stack((np.full(n, 0.25), np.linspace(0.05, 0.20, n)))
        config = {"name": "student-t", "df": 5, "loadings": loadings.tolist()}

        left = simulate_portfolio(pd, lgd, ead, scenarios=8000, seed=71, chunk_size=113, dependency_model=config)
        right = simulate_portfolio(pd, lgd, ead, scenarios=8000, seed=71, chunk_size=2048, dependency_model=config)
        self.assertEqual(left["lossDigest"], right["lossDigest"])
        self.assertEqual(left["expectedLoss"], right["expectedLoss"])
        self.assertEqual(left["dependencyModel"]["factorCount"], 2)
        self.assertEqual(left["dependencyModel"]["degreesOfFreedom"], 5.0)
        self.assertEqual(left["dependencyModel"]["version"], "student-t-low-rank-v1")

    def test_invalid_dependency_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "degrees_of_freedom"):
            simulate_portfolio([0.1], [0.5], [100], dependency_model={"name": "student-t", "df": 2})
        with self.assertRaisesRegex(ValueError, "shape"):
            simulate_portfolio(
                [0.1, 0.2],
                [0.5, 0.5],
                [100, 100],
                dependency_model={"name": "gaussian", "loadings": [[0.1, 0.2]]},
            )
        with self.assertRaisesRegex(ValueError, "either rho or loadings"):
            simulate_portfolio(
                [0.1],
                [0.5],
                [100],
                dependency_model={"name": "gaussian", "rho": 0.1, "loadings": [[0.2]]},
            )
        with self.assertRaisesRegex(ValueError, "squared factor loadings"):
            simulate_portfolio(
                [0.1],
                [0.5],
                [100],
                dependency_model={"name": "gaussian", "loadings": [[0.9, 0.9]]},
            )
        with self.assertRaisesRegex(ValueError, "unsupported dependency"):
            simulate_portfolio([0.1], [0.5], [100], dependency_model={"name": "dense-correlation"})
        with self.assertRaisesRegex(ValueError, "at most"):
            simulate_portfolio(
                [0.1],
                [0.5],
                [100],
                scenarios=1000,
                quantiles=tuple(np.linspace(0.01, 0.99, 129)),
            )

    def test_marginals_and_expected_loss_remain_anchored_across_dependency_choices(self) -> None:
        n = 80
        pd = np.full(n, 0.06)
        lgd = np.ones(n)
        ead = np.ones(n)
        gaussian = simulate_portfolio(pd, lgd, ead, rho=0.0, scenarios=20000, seed=91, quantiles=(0.95,))
        student = simulate_portfolio(
            pd,
            lgd,
            ead,
            scenarios=20000,
            seed=91,
            quantiles=(0.95,),
            dependency_model={"name": "student-t", "rho": 0.0, "df": 5},
        )
        target = float(np.sum(pd))
        self.assertLess(abs(gaussian["expectedLoss"] - target), 6 * gaussian["monteCarlo"]["expectedLossStdError"])
        self.assertLess(abs(student["expectedLoss"] - target), 6 * student["monteCarlo"]["expectedLossStdError"])

    def test_stronger_dependence_and_student_t_increase_controlled_tail_concentration(self) -> None:
        n = 100
        pd = np.full(n, 0.04)
        lgd = np.full(n, 0.5)
        ead = np.full(n, 1000.0)
        low = simulate_portfolio(pd, lgd, ead, rho=0.05, scenarios=20000, seed=101, quantiles=(0.99,))
        high = simulate_portfolio(pd, lgd, ead, rho=0.40, scenarios=20000, seed=101, quantiles=(0.99,))
        student = simulate_portfolio(
            pd,
            lgd,
            ead,
            scenarios=20000,
            seed=101,
            quantiles=(0.99,),
            dependency_model={"name": "student-t", "rho": 0.05, "df": 4},
        )
        self.assertGreater(high["expectedShortfall"]["0.99"], low["expectedShortfall"]["0.99"])
        self.assertGreater(student["expectedShortfall"]["0.99"], low["expectedShortfall"]["0.99"])


def _synthetic_pot_losses(shape: float, scale: float, tail_count: int = 999) -> np.ndarray:
    if tail_count >= 9_900:
        raise ValueError("tail_count must leave enough body observations")
    body = np.linspace(0.0, 10.0, 10_000 - tail_count)
    u = (np.arange(tail_count, dtype=float) + 0.5) / tail_count
    excess = genpareto.ppf(u, c=shape, loc=0.0, scale=scale)
    return np.concatenate((body, 10.0 + excess))


class EvtTests(unittest.TestCase):
    def test_recovers_known_gpd_tail_parameters(self) -> None:
        result = fit_pot_tail(
            _synthetic_pot_losses(0.20, 3.0),
            threshold_quantile=0.90,
            target_quantiles=(0.95, 0.99),
            threshold_candidates=(0.90, 0.925, 0.95),
        )
        self.assertIn(result["status"], {"ok", "unstable-fit"})
        self.assertAlmostEqual(result["shape"], 0.20, delta=0.05)
        self.assertAlmostEqual(result["scale"], 3.0, delta=0.25)
        self.assertEqual(result["method"], "peaks-over-threshold-gpd-v1")
        self.assertTrue(result["researchOnly"])

    def test_distinguishes_light_and_heavy_tails_and_reports_stability(self) -> None:
        light = fit_pot_tail(
            _synthetic_pot_losses(-0.10, 3.0),
            threshold_quantile=0.90,
            target_quantiles=(0.95, 0.99),
        )
        heavy = fit_pot_tail(
            _synthetic_pot_losses(0.35, 3.0),
            threshold_quantile=0.90,
            target_quantiles=(0.95, 0.99),
        )
        self.assertGreater(heavy["shape"], light["shape"])
        self.assertGreater(
            heavy["targets"]["0.99"]["var"],
            light["targets"]["0.99"]["var"],
        )
        self.assertGreaterEqual(len(heavy["thresholdStability"]["candidates"]), 3)

    def test_sparse_and_degenerate_tails_fail_explicitly(self) -> None:
        sparse = fit_pot_tail(
            np.arange(100.0),
            threshold_quantile=0.99,
            target_quantiles=(0.999,),
            minimum_exceedances=20,
        )
        self.assertEqual(sparse["status"], "insufficient-data")
        self.assertEqual(sparse["targets"], {})

        body = np.linspace(0.0, 10.0, 9001)
        degenerate = fit_pot_tail(
            np.concatenate((body, np.full(999, 15.0))),
            threshold_quantile=0.90,
            target_quantiles=(0.95, 0.99),
        )
        self.assertEqual(degenerate["status"], "unstable-fit")
        self.assertIn("degenerate", degenerate["statusDetail"])

    def test_expected_shortfall_is_withheld_when_gpd_mean_is_not_finite(self) -> None:
        result = fit_pot_tail(
            _synthetic_pot_losses(1.20, 1.0, tail_count=1999),
            threshold_quantile=0.80,
            target_quantiles=(0.95, 0.99),
            threshold_candidates=(0.80, 0.85, 0.90),
        )
        self.assertGreaterEqual(result["shape"], 1.0)
        self.assertIsNone(result["targets"]["0.99"]["expectedShortfall"])
        self.assertFalse(result["targets"]["0.99"]["finiteExpectedShortfall"])

    def test_bootstrap_is_bounded_and_emits_uncertainty_when_supported(self) -> None:
        result = fit_pot_tail(
            _synthetic_pot_losses(0.15, 2.5, tail_count=1499),
            threshold_quantile=0.85,
            target_quantiles=(0.95, 0.99),
            bootstrap_samples=20,
            seed=12,
        )
        self.assertEqual(result["bootstrap"]["samples"], 20)
        self.assertIsNotNone(result["targets"]["0.95"].get("varCi95"))
        with self.assertRaisesRegex(ValueError, "bootstrap_samples"):
            fit_pot_tail(
                _synthetic_pot_losses(0.1, 2.0),
                threshold_quantile=0.90,
                target_quantiles=(0.99,),
                bootstrap_samples=501,
            )


if __name__ == "__main__":
    unittest.main()
