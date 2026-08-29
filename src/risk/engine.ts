import rawArtifact from "../../model/artifacts/crix-monoboost-v1.json";
import type { ApplicationInput, Decision, ReasonCode, RiskResult, StressResult, StressSeverity } from "./types";

interface ModelTree {
  left: number[];
  right: number[];
  feature: number[];
  threshold: number[];
  defaultLeft: number[];
}

interface ModelArtifact {
  name: string;
  version: string;
  trainedAt: string;
  featureNames: string[];
  monotoneConstraints: number[];
  baseScore: number;
  calibration: { slope: number; intercept: number };
  trees: ModelTree[];
  metrics: Record<string, number>;
  diagnostics: {
    calibration: Array<{ predicted: number; observed: number; count: number }>;
    roc: Array<{ fpr: number; tpr: number }>;
    featureImportance: Array<{ feature: string; gain: number }>;
  };
  reference: Record<string, number>;
}

const artifact = rawArtifact as ModelArtifact;

const FEATURE_LABELS: Record<string, string> = {
  debtToIncome: "Debt-to-income ratio",
  creditUtilization: "Revolving utilization",
  delinquencies24m: "Recent delinquencies",
  inquiries6m: "Recent credit inquiries",
  oldestTradeMonths: "Credit history age",
  openAccounts: "Open account count",
  loanToIncome: "Requested loan / income",
  employmentYears: "Employment tenure",
  cashBufferMonths: "Liquidity buffer",
  onTimePaymentRate: "On-time payment rate",
  incomeStability: "Income stability",
  recentCreditGrowth: "Recent credit growth",
};

export const POLICY = Object.freeze({
  version: "CRIX-Policy 2.5",
  decline: { pd: 0.22, debtToIncome: 0.67, delinquencies24m: 5, loanToIncome: 1.4 },
  review: { pd: 0.105, debtToIncome: 0.48, confidence: 0.66, delinquencies24m: 2 },
});

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));
const logit = (value: number) => Math.log(value / (1 - value));

function numericInput(input: ApplicationInput): Record<string, number> {
  return {
    annualIncome: input.annualIncome,
    debtToIncome: input.debtToIncome,
    creditUtilization: input.creditUtilization,
    delinquencies24m: input.delinquencies24m,
    inquiries6m: input.inquiries6m,
    oldestTradeMonths: input.oldestTradeMonths,
    openAccounts: input.openAccounts,
    loanAmount: input.loanAmount,
    termMonths: input.termMonths,
    employmentYears: input.employmentYears,
    cashBufferMonths: input.cashBufferMonths,
    onTimePaymentRate: input.onTimePaymentRate,
    incomeStability: input.incomeStability,
    recentCreditGrowth: input.recentCreditGrowth,
  };
}

function assertFiniteInput(input: ApplicationInput): void {
  for (const [name, value] of Object.entries(numericInput(input))) {
    if (!Number.isFinite(value)) throw new RangeError(`Non-finite numeric input: ${name}`);
  }
  if (input.annualIncome <= 0) throw new RangeError("annualIncome must be greater than zero");
  if (input.loanAmount <= 0) throw new RangeError("loanAmount must be greater than zero");
}

function featureVector(input: ApplicationInput): number[] {
  assertFiniteInput(input);
  const loanToIncome = input.loanAmount / input.annualIncome;
  const values: Record<string, number> = {
    debtToIncome: input.debtToIncome,
    creditUtilization: input.creditUtilization,
    delinquencies24m: input.delinquencies24m,
    inquiries6m: input.inquiries6m,
    oldestTradeMonths: input.oldestTradeMonths,
    openAccounts: input.openAccounts,
    loanToIncome,
    employmentYears: input.employmentYears,
    cashBufferMonths: input.cashBufferMonths,
    onTimePaymentRate: input.onTimePaymentRate,
    incomeStability: input.incomeStability,
    recentCreditGrowth: input.recentCreditGrowth,
  };

  return artifact.featureNames.map((name) => {
    const value = values[name];
    if (value === undefined) throw new Error(`Model artifact requests unknown feature: ${name}`);
    return value;
  });
}

function rawChampionMargin(input: ApplicationInput): number {
  const x = featureVector(input);
  let margin = logit(artifact.baseScore);

  for (const tree of artifact.trees) {
    let node = 0;
    let hops = 0;

    while (true) {
      if (hops++ > 64) throw new Error("Model artifact traversal limit exceeded");
      const left = tree.left[node];
      const right = tree.right[node];
      const featureIndex = tree.feature[node];
      const threshold = tree.threshold[node];
      const defaultLeft = tree.defaultLeft[node];

      if (left === undefined || right === undefined || featureIndex === undefined || threshold === undefined || defaultLeft === undefined) {
        throw new Error("Model artifact contains an invalid tree node");
      }
      if (left === -1) {
        if (!Number.isFinite(threshold)) throw new Error("Model artifact contains a non-finite leaf");
        margin += threshold;
        break;
      }

      const value = x[featureIndex];
      if (value === undefined) throw new Error("Model artifact feature index is out of range");
      const goLeft = Number.isNaN(value) ? Boolean(defaultLeft) : value < threshold;
      node = goLeft ? left : right;
      if (node < 0) throw new Error("Model artifact points to an invalid child node");
    }
  }

  if (!Number.isFinite(margin)) throw new Error("Model produced a non-finite margin");
  return margin;
}

export function predictDefaultProbability(input: ApplicationInput): number {
  const margin = rawChampionMargin(input);
  const calibrated = sigmoid(artifact.calibration.slope * margin + artifact.calibration.intercept);
  return clamp(calibrated, 0.0001, 0.9999);
}

export function predictChallengerProbability(input: ApplicationInput): number {
  assertFiniteInput(input);
  const loanToIncome = input.loanAmount / input.annualIncome;
  const z =
    -4.05 +
    3.15 * input.debtToIncome +
    2.45 * input.creditUtilization +
    0.34 * input.delinquencies24m +
    0.11 * input.inquiries6m -
    0.0032 * input.oldestTradeMonths +
    1.02 * loanToIncome -
    0.047 * input.employmentYears -
    0.145 * input.cashBufferMonths -
    1.9 * (input.onTimePaymentRate - 0.82) -
    0.78 * (input.incomeStability - 0.55) +
    0.71 * input.recentCreditGrowth;
  return clamp(sigmoid(z), 0.0001, 0.9999);
}

function outOfDistributionSignals(input: ApplicationInput): string[] {
  const loanToIncome = input.loanAmount / input.annualIncome;
  const checks: Array<[boolean, string]> = [
    [input.annualIncome < 18_000 || input.annualIncome > 350_000, "annualIncome"],
    [input.debtToIncome > 0.85, "debtToIncome"],
    [input.creditUtilization > 1.15, "creditUtilization"],
    [input.delinquencies24m > 8, "delinquencies24m"],
    [input.inquiries6m > 10, "inquiries6m"],
    [input.oldestTradeMonths < 6 || input.oldestTradeMonths > 360, "oldestTradeMonths"],
    [input.openAccounts > 30, "openAccounts"],
    [loanToIncome > 2.5, "loanToIncome"],
    [input.employmentYears > 30, "employmentYears"],
    [input.cashBufferMonths > 12, "cashBufferMonths"],
    [input.onTimePaymentRate < 0.5, "onTimePaymentRate"],
    [input.recentCreditGrowth < -0.5 || input.recentCreditGrowth > 1.5, "recentCreditGrowth"],
  ];
  return checks.filter(([flag]) => flag).map(([, field]) => field);
}

function withReferenceFeature(input: ApplicationInput, feature: string): ApplicationInput {
  const reference = artifact.reference[feature];
  if (reference === undefined) return input;
  if (feature === "loanToIncome") return { ...input, loanAmount: reference * input.annualIncome };

  const candidate = { ...input };
  switch (feature) {
    case "debtToIncome": candidate.debtToIncome = reference; break;
    case "creditUtilization": candidate.creditUtilization = reference; break;
    case "delinquencies24m": candidate.delinquencies24m = reference; break;
    case "inquiries6m": candidate.inquiries6m = reference; break;
    case "oldestTradeMonths": candidate.oldestTradeMonths = reference; break;
    case "openAccounts": candidate.openAccounts = reference; break;
    case "employmentYears": candidate.employmentYears = reference; break;
    case "cashBufferMonths": candidate.cashBufferMonths = reference; break;
    case "onTimePaymentRate": candidate.onTimePaymentRate = reference; break;
    case "incomeStability": candidate.incomeStability = reference; break;
    case "recentCreditGrowth": candidate.recentCreditGrowth = reference; break;
  }
  return candidate;
}

function reasonCodes(input: ApplicationInput, pd: number): ReasonCode[] {
  return artifact.featureNames
    .map((feature) => {
      const counterfactualPd = predictDefaultProbability(withReferenceFeature(input, feature));
      const impact = pd - counterfactualPd;
      return {
        feature,
        label: FEATURE_LABELS[feature] ?? feature,
        impact,
        direction: impact >= 0 ? "risk-up" as const : "risk-down" as const,
      };
    })
    .filter((item) => Math.abs(item.impact) > 0.0005)
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, 5);
}

function scoreFromPd(pd: number): number {
  const goodBadOdds = (1 - pd) / Math.max(pd, 0.0001);
  return Math.round(clamp(600 + 20 * Math.log2(goodBadOdds / 20), 300, 850));
}

function gradeFromScore(score: number): string {
  if (score >= 760) return "A1";
  if (score >= 720) return "A2";
  if (score >= 680) return "B1";
  if (score >= 640) return "B2";
  if (score >= 600) return "C";
  if (score >= 540) return "D";
  return "E";
}

function policyDecision(pd: number, input: ApplicationInput, confidence: number): Decision {
  const loanToIncome = input.loanAmount / input.annualIncome;
  if (
    pd >= POLICY.decline.pd ||
    input.debtToIncome >= POLICY.decline.debtToIncome ||
    input.delinquencies24m >= POLICY.decline.delinquencies24m ||
    loanToIncome >= POLICY.decline.loanToIncome
  ) return "DECLINE";

  if (
    pd >= POLICY.review.pd ||
    input.debtToIncome >= POLICY.review.debtToIncome ||
    confidence < POLICY.review.confidence ||
    input.delinquencies24m >= POLICY.review.delinquencies24m
  ) return "REVIEW";

  return "APPROVE";
}

export function assessRisk(input: ApplicationInput): RiskResult {
  const pd = predictDefaultProbability(input);
  const challengerPd = predictChallengerProbability(input);
  const disagreement = Math.abs(pd - challengerPd);
  const outOfDistribution = outOfDistributionSignals(input);
  const confidence = clamp(0.96 - disagreement * 1.9 - outOfDistribution.length * 0.12, 0.35, 0.98);
  const loanToIncome = input.loanAmount / input.annualIncome;
  const lgd = clamp(0.34 + 0.17 * loanToIncome + 0.11 * (1 - input.incomeStability) + 0.08 * (input.cashBufferMonths < 1 ? 1 : 0), 0.25, 0.82);
  const ead = input.loanAmount;
  const expectedLoss = pd * lgd * ead;
  const score = scoreFromPd(pd);
  const decision = policyDecision(pd, input, confidence);
  const apr = clamp(8.5 + 22 * pd + 3.5 * loanToIncome + (decision === "REVIEW" ? 1.25 : 0), 8.5, 34.5);
  const flags: string[] = [];
  if (disagreement >= 0.08) flags.push("MODEL_DISAGREEMENT");
  if (outOfDistribution.length > 0) flags.push("OUT_OF_DISTRIBUTION");
  if (confidence < POLICY.review.confidence) flags.push("LOW_CONFIDENCE");

  return {
    pd,
    challengerPd,
    disagreement,
    lgd,
    ead,
    expectedLoss,
    expectedLossRate: ead > 0 ? expectedLoss / ead : 0,
    score,
    grade: gradeFromScore(score),
    decision,
    confidence,
    apr,
    reasons: reasonCodes(input, pd),
    modelVersion: `${artifact.name} ${artifact.version}`,
    policyVersion: POLICY.version,
    outOfDistribution,
    flags,
  };
}

export function stressApplication(input: ApplicationInput, severity: StressSeverity): StressResult {
  const baseline = assessRisk(input);
  const factor = severity === "severe" ? 1 : 0.55;
  const stressedInput: ApplicationInput = {
    ...input,
    annualIncome: input.annualIncome * (1 - 0.15 * factor),
    debtToIncome: clamp(input.debtToIncome + 0.10 * factor, 0, 1.5),
    creditUtilization: clamp(input.creditUtilization + 0.15 * factor, 0, 2),
    cashBufferMonths: Math.max(0, input.cashBufferMonths - 1.8 * factor),
    incomeStability: clamp(input.incomeStability - 0.16 * factor, 0, 1),
    recentCreditGrowth: clamp(input.recentCreditGrowth + 0.18 * factor, -1, 2),
  };
  const stressed = assessRisk(stressedInput);

  return {
    severity,
    input: stressedInput,
    baseline,
    stressed,
    delta: {
      pd: stressed.pd - baseline.pd,
      expectedLoss: stressed.expectedLoss - baseline.expectedLoss,
      score: stressed.score - baseline.score,
      decisionChanged: stressed.decision !== baseline.decision,
    },
  };
}

export function modelMetadata() {
  return {
    name: artifact.name,
    version: artifact.version,
    trainedAt: artifact.trainedAt,
    features: artifact.featureNames,
    monotoneConstraints: artifact.monotoneConstraints,
    calibration: artifact.calibration,
    metrics: artifact.metrics,
    diagnostics: artifact.diagnostics,
    treeCount: artifact.trees.length,
    policy: POLICY,
  };
}

export function verifyModelIntegrity(): boolean {
  if (!artifact.name || !artifact.version || artifact.trees.length === 0) return false;
  if (artifact.featureNames.length !== artifact.monotoneConstraints.length) return false;
  if (!Number.isFinite(artifact.calibration.slope) || !Number.isFinite(artifact.calibration.intercept)) return false;
  const sentinel: ApplicationInput = {
    annualIncome: 85_000,
    debtToIncome: 0.28,
    creditUtilization: 0.3,
    delinquencies24m: 0,
    inquiries6m: 1,
    oldestTradeMonths: 96,
    openAccounts: 7,
    loanAmount: 24_000,
    termMonths: 36,
    employmentYears: 5,
    cashBufferMonths: 3,
    onTimePaymentRate: 0.98,
    incomeStability: 0.82,
    recentCreditGrowth: 0.08,
  };
  const result = assessRisk(sentinel);
  return Number.isFinite(result.pd) && result.pd > 0 && result.pd < 1 && Number.isFinite(result.expectedLoss);
}
