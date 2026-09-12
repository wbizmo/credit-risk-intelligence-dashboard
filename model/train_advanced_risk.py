from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from capital import economic_capital, irb_corporate_capital
from decisioning import Candidate, optimize_exact
from ead import amortizing_balance
from ifrs9 import scenario_weighted_ecl
from lgd import economic_lgd
from macro_stress import fit_univariate_logit, stress_pd_path
from portfolio_risk import simulate_portfolio
from publish_champion_manifest import atomic_json
from registry import sha256_file
from research_data import download_lifecycle_lendingclub, load_lifecycle_lendingclub
from survival import predict_cumulative_pd_batch, predict_term_structure

SEED = 42
FRED_UNRATE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE&cosd=2006-01-01&coed=2014-12-31"
SURVIVAL_FEATURES = ["debtToIncome", "loanToIncome", "employmentYears"]


def _download_macro(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(FRED_UNRATE_URL, headers={"User-Agent": "CRIX-advanced-risk-research/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        temporary.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1 << 20):
                    output.write(chunk)
            frame = pd.read_csv(temporary)
            if "UNRATE" not in frame.columns or len(frame) < 90:
                raise RuntimeError(f"unexpected FRED UNRATE payload columns/rows: {frame.columns.tolist()}, {len(frame)}")
            temporary.replace(destination)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt * 3)
    raise RuntimeError(f"unable to download FRED UNRATE research snapshot: {last_error}")


def _load_macro(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "UNRATE" not in frame.columns:
        raise ValueError("macro snapshot must contain UNRATE")
    date_column = next((name for name in frame.columns if name != "UNRATE"), None)
    if date_column is None:
        raise ValueError("macro snapshot is missing its observation date")
    output = pd.DataFrame({
        "observationDate": pd.to_datetime(frame[date_column], errors="coerce"),
        "unemploymentRate": pd.to_numeric(frame["UNRATE"], errors="coerce"),
    }).dropna()
    output["macroPeriod"] = output["observationDate"].dt.to_period("M")
    output = output.drop_duplicates("macroPeriod", keep="last").sort_values("macroPeriod", kind="stable")
    if len(output) < 90:
        raise ValueError("macro snapshot does not cover the intended research window")
    return output


def _join_prior_month_unemployment(loans: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    frame = loans.copy()
    frame["macroPeriod"] = frame["issueDate"].dt.to_period("M") - 1
    joined = frame.merge(macro[["macroPeriod", "unemploymentRate"]], on="macroPeriod", how="left", validate="many_to_one")
    return joined.dropna(subset=["unemploymentRate"]).copy()


def _known_12m(frame: pd.DataFrame) -> pd.DataFrame:
    known_default = (frame["event"] == 1) & (frame["durationMonths"] <= 12)
    known_nondefault = ((frame["event"] == 0) & frame["resolved"]) | (frame["durationMonths"] >= 12)
    eligible = known_default | known_nondefault
    result = frame.loc[eligible].copy()
    result["default12m"] = known_default.loc[eligible].astype(np.int8)
    return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35, 35)))


def _macro_research(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    known = _known_12m(frame)
    train = known[known["issueDate"] <= "2011-12-31"].copy()
    oot = known[(known["issueDate"] >= "2012-01-01") & (known["issueDate"] <= "2012-12-31")].copy()
    if len(train) < 2_000 or len(oot) < 2_000 or train["default12m"].nunique() != 2 or oot["default12m"].nunique() != 2:
        raise RuntimeError(f"macro train/OOT cohorts are too small: {len(train)}, {len(oot)}")

    fit = fit_univariate_logit(train["unemploymentRate"], train["default12m"])
    macro_pd = _sigmoid(fit["intercept"] + fit["coefficient"] * oot["unemploymentRate"].to_numpy(dtype=float))
    intercept_only = float(train["default12m"].mean())
    baseline_pd = np.full(len(oot), intercept_only, dtype=float)
    y = oot["default12m"].to_numpy(dtype=np.int8)
    metrics = {
        "trainSamples": int(len(train)),
        "trainEvents": int(train["default12m"].sum()),
        "ootSamples": int(len(oot)),
        "ootEvents": int(y.sum()),
        "ootObservedRate": float(y.mean()),
        "interceptOnlyBrier": float(brier_score_loss(y, baseline_pd)),
        "macroBrier": float(brier_score_loss(y, macro_pd)),
        "macroAuc": float(roc_auc_score(y, macro_pd)),
    }
    return {
        "model": fit,
        "metrics": metrics,
        "driver": "UNRATE",
        "units": "percent, seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
        "sourceUrl": FRED_UNRATE_URL,
        "seriesSemantics": "revised historical series; not a real-time vintage dataset",
        "availabilityRule": "use only the previous calendar month's UNRATE at origination; current-month observations are never joined",
        "causalityClaim": False,
    }, oot


def _portfolio_rows(frame: pd.DataFrame, size: int = 250) -> pd.DataFrame:
    cohort = frame[(frame["issueDate"] >= "2012-01-01") & (frame["issueDate"] <= "2012-12-31")].copy()
    cohort = cohort.dropna(subset=SURVIVAL_FEATURES + ["fundedAmount", "annualRate", "termMonths", "unemploymentRate"])
    if len(cohort) < size:
        raise RuntimeError(f"portfolio cohort has only {len(cohort)} usable rows")
    return cohort.sample(n=size, random_state=SEED).sort_values(["issueDate", "loanId"], kind="stable").reset_index(drop=True)


def _empirical_lgd_baseline(frame: pd.DataFrame) -> dict:
    defaults = frame[
        (frame["event"] == 1)
        & (frame["observedEad"] > 0)
        & (frame["issueDate"] <= "2011-12-31")
        & (frame["outcomeAvailableAt"] <= "2014-12-31")
    ].copy()
    if len(defaults) < 500:
        raise RuntimeError("not enough historical defaults for the portfolio LGD baseline")
    raw = economic_lgd(
        defaults["observedEad"].to_numpy(dtype=float),
        defaults["recoveries"].to_numpy(dtype=float),
        defaults["recoveryCosts"].to_numpy(dtype=float),
    )
    exposure = defaults["observedEad"].to_numpy(dtype=float)
    weighted = float(np.sum(raw * exposure) / np.sum(exposure))
    return {
        "rawExposureWeightedLgD": weighted,
        "portfolioLgD": float(np.clip(weighted, 0.0, 1.0)),
        "defaults": int(len(defaults)),
        "convention": "historical exposure-weighted default-cohort LGD, bounded to [0,1] only for this portfolio-loss baseline",
    }


def _scheduled_ead12(portfolio: pd.DataFrame) -> np.ndarray:
    return np.asarray([
        amortizing_balance(float(row.fundedAmount), float(row.annualRate), int(row.termMonths), 12)
        for row in portfolio.itertuples(index=False)
    ], dtype=float)


def _decision_demo(portfolio: pd.DataFrame, pd12: np.ndarray, lgd: float, ead12: np.ndarray) -> dict:
    term36 = portfolio[portfolio["termMonths"] == 36].head(9)
    term60 = portfolio[portfolio["termMonths"] == 60].head(9)
    subset = pd.concat([term36, term60]).drop_duplicates("loanId").head(18).copy()
    if len(subset) < 12:
        subset = portfolio.head(18).copy()
    index = subset.index.to_numpy(dtype=int)
    candidates: list[Candidate] = []
    for local, row_index in enumerate(index):
        row = portfolio.loc[row_index]
        exposure = float(row["fundedAmount"])
        expected_loss = float(pd12[row_index] * lgd * ead12[row_index])
        gross_interest_proxy = exposure * float(row["annualRate"])
        candidates.append(Candidate(
            str(int(row["loanId"])), exposure, gross_interest_proxy - expected_loss, expected_loss,
            f"term-{int(row['termMonths'])}", True,
        ))
    total_exposure = sum(candidate.exposure for candidate in candidates)
    budget = 0.50 * total_exposure
    result = optimize_exact(
        candidates,
        budget=budget,
        max_expected_loss=0.08 * budget,
        max_segment_share=0.80,
        min_approval_count=1,
    )
    if result["status"] == "optimal":
        selected = [candidate for candidate in candidates if candidate.candidate_id in set(result["selectedIds"])]
        revalidated = {
            "exposure": sum(candidate.exposure for candidate in selected),
            "expectedLoss": sum(candidate.expected_loss for candidate in selected),
            "expectedReturn": sum(candidate.expected_return for candidate in selected),
        }
    else:
        revalidated = {"exposure": 0.0, "expectedLoss": 0.0, "expectedReturn": 0.0}
    return {
        "objective": "maximize one-year gross-interest proxy minus expected loss",
        "candidateCount": len(candidates),
        "budget": budget,
        "maxExpectedLoss": 0.08 * budget,
        "maxSegmentShare": 0.80,
        "result": result,
        "independentRevalidation": revalidated,
        "selectionBiasLimitation": "candidates are historically granted LendingClub loans; this is not rejected-applicant counterfactual inference",
    }


def _report(artifact: dict) -> str:
    macro = artifact["macro"]
    portfolio = artifact["portfolio"]
    capital = artifact["capital"]
    decision = artifact["decisioning"]
    return f"""# CRIX advanced risk research report

Generated from the final roadmap research stack. This report is **not** production IFRS 9 accounting advice, Basel compliance, regulatory approval, or bank model validation.

## Empirical macro overlay

- Driver: FRED/BLS UNRATE, monthly seasonally adjusted unemployment.
- Availability rule: previous calendar month only; current-month values are never joined.
- Series status: revised historical series, not a real-time vintage dataset.
- Train observations: {macro['metrics']['trainSamples']:,}; OOT observations: {macro['metrics']['ootSamples']:,}.
- Estimated unemployment coefficient: {macro['model']['coefficient']:.6f} (95% Wald CI {macro['model']['coefficientCi95'][0]:.6f} to {macro['model']['coefficientCi95'][1]:.6f}).
- OOT macro AUC: {macro['metrics']['macroAuc']:.4f}; macro Brier: {macro['metrics']['macroBrier']:.4f}; intercept-only Brier: {macro['metrics']['interceptOnlyBrier']:.4f}.

The coefficient is an empirical association in this granted-loan sample, not a causal unemployment effect.

## Correlated portfolio loss

The demonstration portfolio contains {portfolio['portfolioSize']} historically granted 2012 LendingClub loans, scored with CRIX-LifetimePD 1.0.0 at 12 months, a historical cohort LGD baseline, and contractual 12-month amortized EAD.

- Scenarios: {portfolio['baseline']['scenarios']:,}; one-factor rho: {portfolio['baseline']['rho']:.2f}.
- Expected loss: ${portfolio['baseline']['expectedLoss']:,.2f}.
- Unexpected loss: ${portfolio['baseline']['unexpectedLoss']:,.2f}.
- 99% VaR: ${portfolio['baseline']['var']['0.99']:,.2f}.
- 99% expected shortfall: ${portfolio['baseline']['expectedShortfall']['0.99']:,.2f}.
- 99.9% VaR: ${portfolio['baseline']['var']['0.999']:,.2f}.

## IFRS 9-style ECL research

The engine converts cumulative PD to marginal interval PD exactly once, applies scenario weights, EAD/LGD alignment and EIR discounting, and keeps Stage 1 12-month ECL separate from Stage 2 lifetime ECL. The worked example is illustrative and does not assert production IFRS 9 compliance.

## Capital research

- Economic capital at 99.9% VaR: ${capital['economic']['economicCapital']:,.2f} above expected loss.
- Aggregate IRB-style capital on the demonstration portfolio: ${capital['irbStyleAggregateCapital']:,.2f}.
- RWA-style amount at the explicit 8% conversion assumption: ${capital['irbStyleAggregateRwa']:,.2f}.

Accounting ECL and capital are deliberately reported as separate concepts.

## Decision optimisation

The bounded exact optimiser evaluated {decision['candidateCount']} observed candidates under budget, EL and term-concentration constraints. Status: **{decision['result']['status']}**. It independently rechecks the selected portfolio after optimisation and never relaxes hard eligibility constraints silently.

## Limitations

- LendingClub observations are granted-loan outcomes; selection-bias limitations remain.
- UNRATE is a revised historical series, so this run does not claim real-time macro-vintage backtesting.
- The macro model is intentionally univariate and interpretable; no causal claim is made.
- Portfolio correlation rho is a transparent research assumption, not a supervisory parameter estimate.
- LGD used by the portfolio demonstration is a historical cohort baseline; the separate CRIX-LGD research model remains conditional-at-default.
- `/api/v3` serving semantics and CRIX-MonoBoost 2.0.0 bytes are unchanged.
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate CRIX advanced risk research evidence for roadmap #20-#24")
    parser.add_argument("--lifecycle-data", type=Path, default=root / "model" / "data" / "loan_data_2007_2014.csv")
    parser.add_argument("--macro-data", type=Path, default=root / "model" / "public_data" / "UNRATE_2006_2014.csv")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if not args.lifecycle_data.exists():
        if args.no_download:
            raise FileNotFoundError(args.lifecycle_data)
        download_lifecycle_lendingclub(args.lifecycle_data)
    if not args.macro_data.exists():
        if args.no_download:
            raise FileNotFoundError(args.macro_data)
        _download_macro(args.macro_data)

    lifecycle = load_lifecycle_lendingclub(args.lifecycle_data)
    macro_source = _load_macro(args.macro_data)
    joined = _join_prior_month_unemployment(lifecycle.frame, macro_source)
    macro_artifact, _ = _macro_research(joined)
    macro_artifact["sourceSha256"] = sha256_file(args.macro_data)

    lifetime = json.loads((root / "model" / "artifacts" / "crix-lifetime-pd-v1.json").read_text())
    portfolio = _portfolio_rows(joined)
    features = portfolio[SURVIVAL_FEATURES].to_numpy(dtype=float)
    pd12 = predict_cumulative_pd_batch(lifetime["model"], features, 12)
    ead12 = _scheduled_ead12(portfolio)
    lgd_baseline = _empirical_lgd_baseline(joined)
    lgd = np.full(len(portfolio), lgd_baseline["portfolioLgD"], dtype=float)

    baseline_mc = simulate_portfolio(pd12, lgd, ead12, rho=0.15, scenarios=100_000, seed=SEED, chunk_size=1024)
    stress = stress_pd_path(
        pd12,
        portfolio["unemploymentRate"].to_numpy(dtype=float),
        portfolio["unemploymentRate"].to_numpy(dtype=float) + 2.0,
        coefficient=float(macro_artifact["model"]["coefficient"]),
        support=tuple(macro_artifact["model"]["support"]),
    )
    stressed_mc = simulate_portfolio(stress["pd"], lgd, ead12, rho=0.15, scenarios=100_000, seed=SEED, chunk_size=1024)

    row = portfolio.iloc[len(portfolio) // 2]
    curve = predict_term_structure(lifetime["model"], row[SURVIVAL_FEATURES].to_numpy(dtype=float), 36)["cumulativePd"]
    cumulative = [float(curve[11]), float(curve[23]), float(curve[35])]
    adverse_curve = stress_pd_path(
        cumulative, [float(row["unemploymentRate"])] * 3, [float(row["unemploymentRate"]) + 2.0] * 3,
        coefficient=float(macro_artifact["model"]["coefficient"]), support=tuple(macro_artifact["model"]["support"]),
    )["pd"].tolist()
    ead_curve = [
        amortizing_balance(float(row["fundedAmount"]), float(row["annualRate"]), int(row["termMonths"]), month)
        for month in (12, 24, 36)
    ]
    ecl_scenarios = {
        "baseline": {"weight": 0.75, "cumulativePd": cumulative, "lgd": [lgd_baseline["portfolioLgD"]] * 3, "ead": ead_curve},
        "unemploymentPlus2pp": {"weight": 0.25, "cumulativePd": adverse_curve, "lgd": [lgd_baseline["portfolioLgD"]] * 3, "ead": ead_curve},
    }
    ecl_stage1 = scenario_weighted_ecl(ecl_scenarios, stage=1, annual_eir=float(row["annualRate"]), interval_months=12)
    ecl_stage2 = scenario_weighted_ecl(ecl_scenarios, stage=2, annual_eir=float(row["annualRate"]), interval_months=12)

    tail_999 = baseline_mc["var"]["0.999"]
    if tail_999 is None:
        raise RuntimeError("100k simulation unexpectedly withheld 99.9% VaR")
    econ = economic_capital(expected_loss=baseline_mc["expectedLoss"], tail_loss=tail_999, confidence=0.999, tail_measure="VaR")
    irb = [
        irb_corporate_capital(float(pd12[i]), float(lgd[i]), float(ead12[i]), maturity_years=float(np.clip(portfolio.iloc[i]["termMonths"] / 12.0, 1.0, 5.0)))
        for i in range(len(portfolio))
    ]
    decision = _decision_demo(portfolio, pd12, lgd_baseline["portfolioLgD"], ead12)

    artifact = {
        "schemaVersion": 1,
        "name": "CRIX-AdvancedRisk-Research",
        "version": "1.0.0",
        "status": "research",
        "populationConditioning": "historically-granted-LendingClub-loans",
        "sourceDatasetSha256": lifecycle.source_sha256,
        "macro": macro_artifact,
        "portfolio": {
            "portfolioSize": len(portfolio),
            "horizonMonths": 12,
            "pdModelId": "CRIX-LifetimePD@1.0.0",
            "lgdBaseline": lgd_baseline,
            "eadConvention": "contractual amortized principal at 12 months",
            "baseline": baseline_mc,
            "unemploymentPlus2pp": {"pdOverlay": {"extrapolative": stress["extrapolative"]}, "simulation": stressed_mc},
        },
        "ifrs9Style": {
            "policyVersion": "ifrs9-research-v1",
            "representativeLoanId": str(int(row["loanId"])),
            "scenarioWeights": {"baseline": 0.75, "unemploymentPlus2pp": 0.25},
            "stage1": ecl_stage1,
            "stage2": ecl_stage2,
            "limitation": "the univariate 12m macro log-odds overlay is reused across the illustrative term structure; this is not a horizon-specific production macro model",
        },
        "capital": {
            "economic": econ,
            "irbStyleAggregateCapital": float(sum(item["capital"] for item in irb)),
            "irbStyleAggregateRwa": float(sum(item["rwaStyle"] for item in irb)),
            "formulaVersion": "basel-irb-corporate-research-v1",
            "status": "research only; not regulatory compliance",
        },
        "decisioning": decision,
        "runtimeContract": "/api/v3 and CRIX-MonoBoost 2.0.0 remain unchanged",
    }

    artifact_path = root / "model" / "artifacts" / "crix-advanced-risk-v1.json"
    atomic_json(artifact_path, artifact)
    (root / "model" / "ADVANCED_RISK_REPORT.md").write_text(_report(artifact), encoding="utf-8")
    print(json.dumps({
        "artifact": str(artifact_path.relative_to(root)),
        "macroSource": str(args.macro_data.relative_to(root)),
        "macroSha256": macro_artifact["sourceSha256"],
        "portfolioLossDigest": baseline_mc["lossDigest"],
        "decisionStatus": decision["result"]["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
