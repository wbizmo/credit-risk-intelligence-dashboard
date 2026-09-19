import { runtimeManifestMetadata } from "./registry";
import {
  emptyComplexityCounters,
  evaluateBaselineWithCounters,
  evaluateCompiledBaseline,
  evaluateCompiledMargin,
  evaluateMarginWithFeatureOverride,
  evaluateOverrideWithCounters,
  evaluateReferenceMargin,
  getRuntimeModel,
  runtimeArtifactForBenchmark,
  verifyCompiledRuntimeModel,
  type BaselineEvaluation,
  type CompiledModel,
  type ComplexityCounters,
} from "./runtime";
import type {
  ApplicationInput,
  Counterfactual,
  Decision,
  PolicyReason,
  ReasonCode,
  RiskResult,
  StressResult,
  StressSeverity,
} from "./types";

interface ScoringContext {
  input: ApplicationInput;
  values: Record<string, number>;
  vector: readonly number[];
  loanToIncome: number;
}

interface ExplanationEvaluation {
  feature: string;
  label: string;
  current: number;
  reference: number;
  counterfactualPd: number;
  impact: number;
  lowerBound: number;
  upperBound: number;
}

const artifact = runtimeArtifactForBenchmark();

const FEATURE_LABELS: Record<string, string> = {
  debtToIncome: "Debt-to-income ratio",
  creditUtilization: "Revolving utilization",
  creditScore: "External bureau credit score",
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
  version: "CRIX-Policy 3.0",
  target: "origination default risk over final loan resolution",
  decline: { pd: 0.35, debtToIncome: 0.67, creditScore: 580, delinquencies24m: 5, loanToIncome: 1.4 },
  review: { pd: 0.20, debtToIncome: 0.48, creditScore: 660, confidence: 0.66, delinquencies24m: 2 },
});

const SENSITIVITY = Object.freeze({
  method: "deterministic-borrower-sensitivity" as const,
  version: "CRIX-Sensitivity 1.0",
  factors: Object.freeze({ mild: 0.55, severe: 1.0 }),
});

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));
const logit = (value: number) => Math.log(value / (1 - value));

function numericInput(input: ApplicationInput): Record<string, number> {
  return {
    annualIncome: input.annualIncome,
    debtToIncome: input.debtToIncome,
    creditScore: input.creditScore,
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

function createScoringContext(input: ApplicationInput, model: CompiledModel): ScoringContext {
  assertFiniteInput(input);
  const loanToIncome = input.loanAmount / input.annualIncome;
  const values: Record<string, number> = {
    debtToIncome: input.debtToIncome,
    creditUtilization: input.creditUtilization,
    creditScore: input.creditScore,
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
  const vector = model.featureNames.map((name) => {
    const value = values[name];
    if (value === undefined) throw new Error(`Model artifact requests unknown feature: ${name}`);
    return value;
  });
  return { input, values, vector, loanToIncome };
}

function calibratedProbability(model: CompiledModel, margin: number): number {
  const calibrated = sigmoid(model.calibrationSlope * margin + model.calibrationIntercept);
  return clamp(calibrated, 0.0001, 0.9999);
}

function championProbability(context: ScoringContext, model: CompiledModel): number {
  return calibratedProbability(model, evaluateCompiledMargin(model, context.vector));
}

export function predictDefaultProbability(input: ApplicationInput): number {
  const model = getRuntimeModel();
  return championProbability(createScoringContext(input, model), model);
}

function challengerProbability(
  context: ScoringContext,
  model: CompiledModel,
  counters?: ComplexityCounters,
): number {
  let margin = model.challengerIntercept;
  for (let index = 0; index < context.vector.length; index += 1) {
    if (counters) counters.challengerFeatureOps += 1;
    margin += model.challengerCoefficients[index]!
      * ((context.vector[index]! - model.challengerMeans[index]!) / model.challengerScales[index]!);
  }
  return clamp(sigmoid(margin), 0.0001, 0.9999);
}

export function predictChallengerProbability(input: ApplicationInput): number {
  const model = getRuntimeModel();
  return challengerProbability(createScoringContext(input, model), model);
}

function outOfDistributionSignals(context: ScoringContext, model: CompiledModel): string[] {
  const signals: string[] = [];
  for (let index = 0; index < model.featureNames.length; index += 1) {
    const feature = model.featureNames[index]!;
    const value = context.vector[index]!;
    if (value < model.lowerBounds[index]! || value > model.upperBounds[index]!) signals.push(feature);
  }
  return signals;
}

function withFeatureValue(input: ApplicationInput, feature: string, value: number): ApplicationInput {
  if (feature === "loanToIncome") return { ...input, loanAmount: value * input.annualIncome };
  const candidate = { ...input };
  switch (feature) {
    case "debtToIncome": candidate.debtToIncome = value; break;
    case "creditUtilization": candidate.creditUtilization = value; break;
    case "creditScore": candidate.creditScore = value; break;
    case "delinquencies24m": candidate.delinquencies24m = value; break;
    case "inquiries6m": candidate.inquiries6m = value; break;
    case "oldestTradeMonths": candidate.oldestTradeMonths = value; break;
    case "openAccounts": candidate.openAccounts = value; break;
    case "employmentYears": candidate.employmentYears = value; break;
    case "cashBufferMonths": candidate.cashBufferMonths = value; break;
    case "onTimePaymentRate": candidate.onTimePaymentRate = value; break;
    case "incomeStability": candidate.incomeStability = value; break;
    case "recentCreditGrowth": candidate.recentCreditGrowth = value; break;
    default: throw new Error(`Cannot perturb unknown feature: ${feature}`);
  }
  return candidate;
}

function referenceProbability(input: ApplicationInput, model: CompiledModel): number {
  const context = createScoringContext(input, model);
  return calibratedProbability(model, evaluateReferenceMargin(artifact, context.vector));
}

function explanationEvaluations(
  context: ScoringContext,
  pd: number,
  model: CompiledModel,
  baseline: BaselineEvaluation,
  mode: "sparse" | "reference",
  counters?: ComplexityCounters,
): ExplanationEvaluation[] {
  return model.featureNames.map((feature, featureIndex) => {
    const current = context.vector[featureIndex]!;
    const target = clamp(model.reference[featureIndex]!, model.lowerBounds[featureIndex]!, model.upperBounds[featureIndex]!);
    let counterfactualPd: number;

    if (mode === "reference") {
      counterfactualPd = referenceProbability(withFeatureValue(context.input, feature, target), model);
    } else {
      const margin = counters
        ? evaluateOverrideWithCounters(model, baseline, context.vector, featureIndex, target, counters)
        : evaluateMarginWithFeatureOverride(model, baseline, context.vector, featureIndex, target);
      counterfactualPd = calibratedProbability(model, margin);
    }

    return {
      feature,
      label: FEATURE_LABELS[feature] ?? feature,
      current,
      reference: target,
      counterfactualPd,
      impact: pd - counterfactualPd,
      lowerBound: model.lowerBounds[featureIndex]!,
      upperBound: model.upperBounds[featureIndex]!,
    };
  });
}

function reasonCodes(evaluations: ExplanationEvaluation[]): ReasonCode[] {
  return evaluations
    .map(({ feature, label, impact }) => ({
      feature,
      label,
      impact,
      direction: impact >= 0 ? "risk-up" as const : "risk-down" as const,
    }))
    .filter((item) => Math.abs(item.impact) > 0.0005)
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact) || a.feature.localeCompare(b.feature))
    .slice(0, 5);
}

function counterfactuals(evaluations: ExplanationEvaluation[], pd: number): Counterfactual[] {
  return evaluations
    .filter((item) => item.counterfactualPd < pd - 0.0005 && item.reference !== item.current)
    .map((item) => ({
      feature: item.feature,
      label: item.label,
      from: item.current,
      to: item.reference,
      pdBefore: pd,
      pdAfter: item.counterfactualPd,
      trainingLowerBound: item.lowerBound,
      trainingUpperBound: item.upperBound,
    }))
    .sort((a, b) => (b.pdBefore - b.pdAfter) - (a.pdBefore - a.pdAfter) || a.feature.localeCompare(b.feature))
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

function policyEvaluation(pd: number, context: ScoringContext, confidence: number): { decision: Decision; reasons: PolicyReason[] } {
  const decline: PolicyReason[] = [];
  if (pd >= POLICY.decline.pd) decline.push({ code: "PD_DECLINE_THRESHOLD", label: "Model PD meets the decline threshold", decision: "DECLINE" });
  if (context.input.debtToIncome >= POLICY.decline.debtToIncome) decline.push({ code: "DTI_DECLINE_THRESHOLD", label: "Debt-to-income meets the decline threshold", decision: "DECLINE" });
  if (context.input.creditScore <= POLICY.decline.creditScore) decline.push({ code: "CREDIT_SCORE_DECLINE_THRESHOLD", label: "Credit score meets the decline threshold", decision: "DECLINE" });
  if (context.input.delinquencies24m >= POLICY.decline.delinquencies24m) decline.push({ code: "DELINQUENCY_DECLINE_THRESHOLD", label: "Recent delinquencies meet the decline threshold", decision: "DECLINE" });
  if (context.loanToIncome >= POLICY.decline.loanToIncome) decline.push({ code: "LOAN_TO_INCOME_DECLINE_THRESHOLD", label: "Requested loan-to-income meets the decline threshold", decision: "DECLINE" });
  if (decline.length > 0) return { decision: "DECLINE", reasons: decline };

  const review: PolicyReason[] = [];
  if (pd >= POLICY.review.pd) review.push({ code: "PD_REVIEW_THRESHOLD", label: "Model PD meets the review threshold", decision: "REVIEW" });
  if (context.input.debtToIncome >= POLICY.review.debtToIncome) review.push({ code: "DTI_REVIEW_THRESHOLD", label: "Debt-to-income meets the review threshold", decision: "REVIEW" });
  if (context.input.creditScore <= POLICY.review.creditScore) review.push({ code: "CREDIT_SCORE_REVIEW_THRESHOLD", label: "Credit score meets the review threshold", decision: "REVIEW" });
  if (confidence < POLICY.review.confidence) review.push({ code: "LOW_CONFIDENCE_REVIEW_THRESHOLD", label: "Model confidence is below the review threshold", decision: "REVIEW" });
  if (context.input.delinquencies24m >= POLICY.review.delinquencies24m) review.push({ code: "DELINQUENCY_REVIEW_THRESHOLD", label: "Recent delinquencies meet the review threshold", decision: "REVIEW" });
  return review.length > 0 ? { decision: "REVIEW", reasons: review } : { decision: "APPROVE", reasons: [] };
}

function assessRiskInternal(
  input: ApplicationInput,
  mode: "sparse" | "reference",
  counters?: ComplexityCounters,
): RiskResult {
  const model = getRuntimeModel();
  const context = createScoringContext(input, model);
  const baseline = mode === "reference"
    ? { margin: 0, treeContributions: [] }
    : counters
      ? evaluateBaselineWithCounters(model, context.vector, counters)
      : evaluateCompiledBaseline(model, context.vector);
  const pd = mode === "reference"
    ? calibratedProbability(model, evaluateReferenceMargin(artifact, context.vector))
    : calibratedProbability(model, baseline.margin);
  const challengerPd = challengerProbability(context, model, counters);
  const disagreement = Math.abs(pd - challengerPd);
  const outOfDistribution = outOfDistributionSignals(context, model);
  const confidence = clamp(0.96 - disagreement * 1.9 - outOfDistribution.length * 0.12, 0.35, 0.98);
  const lgd = clamp(0.34 + 0.17 * context.loanToIncome + 0.11 * (1 - input.incomeStability) + 0.08 * (input.cashBufferMonths < 1 ? 1 : 0), 0.25, 0.82);
  const ead = input.loanAmount;
  const expectedLoss = pd * lgd * ead;
  const score = scoreFromPd(pd);
  const policy = policyEvaluation(pd, context, confidence);
  const apr = clamp(8.5 + 22 * pd + 3.5 * context.loanToIncome + (policy.decision === "REVIEW" ? 1.25 : 0), 8.5, 34.5);
  const flags: string[] = [];
  if (disagreement >= 0.08) flags.push("MODEL_DISAGREEMENT");
  if (outOfDistribution.length > 0) flags.push("OUT_OF_DISTRIBUTION");
  if (confidence < POLICY.review.confidence) flags.push("LOW_CONFIDENCE");
  const explanations = explanationEvaluations(context, pd, model, baseline, mode, counters);

  return {
    pd,
    pdHorizon: artifact.target.horizon,
    challengerPd,
    disagreement,
    lgd,
    ead,
    expectedLoss,
    expectedLossRate: ead > 0 ? expectedLoss / ead : 0,
    score,
    grade: gradeFromScore(score),
    decision: policy.decision,
    confidence,
    apr,
    reasons: reasonCodes(explanations),
    policyReasons: policy.reasons,
    counterfactuals: counterfactuals(explanations, pd),
    modelVersion: `${artifact.name} ${artifact.version}`,
    policyVersion: POLICY.version,
    outOfDistribution,
    flags,
  };
}

export function assessRisk(input: ApplicationInput): RiskResult {
  return assessRiskInternal(input, "sparse");
}

export function assessRiskReferenceForTest(input: ApplicationInput): RiskResult {
  return assessRiskInternal(input, "reference");
}

export function assessRiskWithComplexity(input: ApplicationInput): { result: RiskResult; complexity: ComplexityCounters } {
  const complexity = emptyComplexityCounters();
  return { result: assessRiskInternal(input, "sparse", complexity), complexity };
}

export function stressApplication(input: ApplicationInput, severity: StressSeverity): StressResult {
  const baseline = assessRisk(input);
  const factor = SENSITIVITY.factors[severity];
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
    method: SENSITIVITY.method,
    scenarioVersion: SENSITIVITY.version,
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
  const model = getRuntimeModel();
  return {
    name: artifact.name,
    version: artifact.version,
    trainedAt: artifact.trainedAt,
    target: artifact.target,
    features: artifact.featureNames,
    monotoneConstraints: artifact.monotoneConstraints,
    calibration: artifact.calibration,
    challenger: { name: artifact.challenger.name },
    metrics: artifact.metrics,
    diagnostics: artifact.diagnostics,
    trainingBounds: artifact.trainingBounds,
    training: { ...artifact.training, registry: runtimeManifestMetadata() },
    treeCount: model.trees.length,
    policy: POLICY,
    sensitivity: { method: SENSITIVITY.method, version: SENSITIVITY.version },
  };
}

export function verifyModelIntegrity(): boolean {
  if (!verifyCompiledRuntimeModel()) return false;
  try {
    const model = getRuntimeModel();
    if (model.featureNames.length !== artifact.monotoneConstraints.length) return false;
    const sentinel: ApplicationInput = {
      annualIncome: 85_000,
      debtToIncome: 0.28,
      creditScore: 720,
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
  } catch {
    return false;
  }
}

export function benchmarkChampionProbability(input: ApplicationInput): number {
  return predictDefaultProbability(input);
}

export function benchmarkChallengerProbability(input: ApplicationInput): number {
  return predictChallengerProbability(input);
}

export function benchmarkReferenceProbability(input: ApplicationInput): number {
  const model = getRuntimeModel();
  const context = createScoringContext(input, model);
  return clamp(sigmoid(artifact.calibration.slope * evaluateReferenceMargin(artifact, context.vector) + artifact.calibration.intercept), 0.0001, 0.9999);
}

export function benchmarkTreeDensity() {
  const model = getRuntimeModel();
  return model.featureNames.map((feature, index) => ({
    feature,
    trees: model.treesByFeature[index]!.length,
    totalTrees: model.trees.length,
  }));
}

export function benchmarkExpectedFullExplanationTreeVisits(): number {
  const model = getRuntimeModel();
  return model.featureNames.length * model.trees.length;
}

export function benchmarkBaseMargin(): number {
  return logit(artifact.baseScore);
}
