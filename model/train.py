from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

SEED = 42
FEATURES = [
    "debtToIncome", "creditUtilization", "delinquencies24m", "inquiries6m",
    "oldestTradeMonths", "openAccounts", "loanToIncome", "employmentYears",
    "cashBufferMonths", "onTimePaymentRate", "incomeStability", "recentCreditGrowth",
]
MONOTONE = [1, 1, 1, 1, -1, 0, 1, -1, -1, -1, -1, 1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=50_000):
    r = np.random.default_rng(SEED)
    income = np.exp(r.normal(np.log(65_000), 0.55, n)).clip(18_000, 350_000)
    dti = r.beta(2.0, 5.0, n) * 0.85
    util = r.beta(2.0, 3.2, n) * 1.15
    delin = r.poisson(0.42, n).clip(0, 8)
    inquiries = r.poisson(1.15, n).clip(0, 10)
    oldest = r.gamma(3.0, 48.0, n).clip(6, 360)
    openacc = (r.poisson(7, n) + 1).clip(1, 30)
    loan = np.exp(r.normal(np.log(18_000), 0.7, n)).clip(1_000, 120_000)
    lti = (loan / income).clip(0.01, 2.5)
    employment = r.gamma(2.2, 2.7, n).clip(0, 30)
    buffer = r.gamma(1.6, 1.6, n).clip(0, 12)
    ontime = (1 - r.beta(1.4, 10, n) * 0.45).clip(0.5, 1)
    stability = r.beta(6, 2.1, n)
    growth = (r.normal(0.1, 0.22, n) + 0.20 * (util > 0.8) + 0.08 * inquiries).clip(-0.5, 1.5)

    z = (
        -3.65 + 3.05 * dti + 2.55 * util + 0.38 * delin + 0.12 * inquiries
        - 0.0035 * oldest + 1.15 * lti - 0.050 * employment - 0.16 * buffer
        - 2.1 * (ontime - 0.82) - 0.85 * (stability - 0.55) + 0.85 * growth
        + 0.85 * ((dti > 0.52) & (util > 0.72)) + 0.55 * ((delin >= 2) & (inquiries >= 3))
        + 0.42 * ((buffer < 1.0) & (lti > 0.55))
        + r.normal(0, 0.72, n)
    )
    p = sigmoid(z)
    y = r.binomial(1, p)
    X = np.column_stack([
        dti, util, delin, inquiries, oldest, openacc, lti,
        employment, buffer, ontime, stability, growth,
    ])
    return X, y


X, y = generate()
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.40, random_state=SEED, stratify=y
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
)

model = xgb.XGBClassifier(
    n_estimators=18,
    max_depth=3,
    learning_rate=0.12,
    subsample=0.88,
    colsample_bytree=0.90,
    min_child_weight=20,
    reg_lambda=3.0,
    reg_alpha=0.05,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    max_bin=256,
    monotone_constraints=tuple(MONOTONE),
    random_state=SEED,
    n_jobs=4,
)
model.fit(X_train, y_train)

margin_cal = model.predict(X_cal, output_margin=True).reshape(-1, 1)
cal = LogisticRegression(C=1000, solver="lbfgs").fit(margin_cal, y_cal)
margin_test = model.predict(X_test, output_margin=True)
p_test = cal.predict_proba(margin_test.reshape(-1, 1))[:, 1]

auc = roc_auc_score(y_test, p_test)
brier = brier_score_loss(y_test, p_test)
ll = log_loss(y_test, p_test)
order = np.argsort(p_test)
ys = y_test[order]
pos = (ys == 1).astype(float)
neg = (ys == 0).astype(float)
ks = float(np.max(np.abs(np.cumsum(pos) / pos.sum() - np.cumsum(neg) / neg.sum())))

ROOT = Path(__file__).resolve().parents[1]
raw_path = ROOT / "model" / "raw.json"
artifact_dir = ROOT / "model" / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
artifact_path = artifact_dir / "crix-monoboost-v1.json"

model.get_booster().save_model(raw_path)
raw = json.loads(raw_path.read_text())
learner = raw["learner"]
booster = learner["gradient_booster"]["model"]
compact_trees = []
for tree in booster["trees"]:
    compact_trees.append({
        "left": tree["left_children"],
        "right": tree["right_children"],
        "feature": tree["split_indices"],
        "threshold": tree["split_conditions"],
        "defaultLeft": tree["default_left"],
    })

base = float(learner["learner_model_param"]["base_score"].strip("[]"))
q = np.quantile(p_test, np.linspace(0, 1, 11))
calibration = []
for i in range(10):
    mask = (p_test >= q[i]) & ((p_test <= q[i + 1]) if i == 9 else (p_test < q[i + 1]))
    calibration.append({
        "predicted": round(float(p_test[mask].mean()), 4),
        "observed": round(float(y_test[mask].mean()), 4),
        "count": int(mask.sum()),
    })

fpr, tpr, _ = roc_curve(y_test, p_test)
idx = np.linspace(0, len(fpr) - 1, 24, dtype=int)
roc = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx]

gain = model.get_booster().get_score(importance_type="gain")
importance = [{"feature": name, "gain": float(gain.get(f"f{i}", 0.0))} for i, name in enumerate(FEATURES)]
total_gain = sum(item["gain"] for item in importance) or 1
for item in importance:
    item["gain"] = round(item["gain"] / total_gain, 4)
importance = sorted(importance, key=lambda item: item["gain"], reverse=True)

artifact = {
    "name": "CRIX-MonoBoost",
    "version": "1.0.0",
    "trainedAt": "2026-08-29",
    "featureNames": FEATURES,
    "monotoneConstraints": MONOTONE,
    "baseScore": base,
    "calibration": {"slope": float(cal.coef_[0, 0]), "intercept": float(cal.intercept_[0])},
    "trees": compact_trees,
    "metrics": {
        "auc": round(float(auc), 4),
        "brier": round(float(brier), 4),
        "logLoss": round(float(ll), 4),
        "ks": round(ks, 4),
        "testSamples": int(len(y_test)),
        "defaultRate": round(float(y_test.mean()), 4),
    },
    "diagnostics": {
        "calibration": calibration,
        "roc": roc,
        "featureImportance": importance,
    },
    "reference": {
        "debtToIncome": 0.28,
        "creditUtilization": 0.30,
        "delinquencies24m": 0,
        "inquiries6m": 1,
        "oldestTradeMonths": 96,
        "openAccounts": 7,
        "loanToIncome": 0.28,
        "employmentYears": 5,
        "cashBufferMonths": 3,
        "onTimePaymentRate": 0.98,
        "incomeStability": 0.82,
        "recentCreditGrowth": 0.08,
    },
}

artifact_path.write_text(json.dumps(artifact, separators=(",", ":")))
print(artifact["metrics"])
print("artifact", artifact_path)
raw_path.unlink(missing_ok=True)
