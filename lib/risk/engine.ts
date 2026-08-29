import artifact from "./model-artifact.json";
import type { ApplicationInput, Decision, PortfolioRecord, ReasonCode, RiskResult } from "./types";

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

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));
const logit = (value: number) => Math.log(value / (1 - value));

function featureVector(input: ApplicationInput): number[] {
  const loanToIncome = input.loanAmount / Math.max(input.annualIncome, 1);
  const values: Record<string, number> = { ...input, loanToIncome };
  return artifact.featureNames.map((name) => values[name]);
}

function rawChampionMargin(input: ApplicationInput): number {
  const x = featureVector(input);
  let margin = logit(artifact.baseScore);

  for (const tree of artifact.trees) {
    let node = 0;
    while (tree.left[node] !== -1) {
      const value = x[tree.feature[node]];
      const goLeft = Number.isNaN(value)
        ? Boolean(tree.defaultLeft[node])
        : value < tree.threshold[node];
      node = goLeft ? tree.left[node] : tree.right[node];
    }
    margin += tree.threshold[node];
  }

  return margin;
}

export function predictDefaultProbability(input: ApplicationInput): number {
  const margin = rawChampionMargin(input);
  return sigmoid(artifact.calibration.slope * margin + artifact.calibration.intercept);
}

export function predictChallengerProbability(input: ApplicationInput): number {
  const lti = input.loanAmount / Math.max(input.annualIncome, 1);
  const z =
    -4.05 +
    3.15 * input.debtToIncome +
    2.45 * input.creditUtilization +
    0.34 * input.delinquencies24m +
    0.11 * input.inquiries6m -
    0.0032 * input.oldestTradeMonths +
    1.02 * lti -
    0.047 * input.employmentYears -
    0.145 * input.cashBufferMonths -
    1.9 * (input.onTimePaymentRate - 0.82) -
    0.78 * (input.incomeStability - 0.55) +
    0.71 * input.recentCreditGrowth;
  return sigmoid(z);
}

function oodSignals(input: ApplicationInput): string[] {
  const checks: Array<[boolean, string]> = [
    [input.annualIncome < 12_000 || input.annualIncome > 500_000, "annual income"],
    [input.debtToIncome < 0 || input.debtToIncome > 0.9, "debt-to-income"],
    [input.creditUtilization < 0 || input.creditUtilization > 1.25, "credit utilization"],
    [input.oldestTradeMonths < 3 || input.oldestTradeMonths > 420, "credit history age"],
    [input.onTimePaymentRate < 0.45 || input.onTimePaymentRate > 1, "payment rate"],
    [input.incomeStability < 0 || input.incomeStability > 1, "income stability"],
  ];
  return checks.filter(([flag]) => flag).map(([, label]) => label);
}

function reasonCodes(input: ApplicationInput, pd: number): ReasonCode[] {
  const reference = artifact.reference as Record<string, number>;
  const candidates = artifact.featureNames.map((feature) => {
    const counterfactual = { ...input };
    if (feature === "loanToIncome") {
      counterfactual.loanAmount = reference.loanToIncome * input.annualIncome;
    } else {
      (counterfactual as unknown as Record<string, number>)[feature] = reference[feature];
    }
    const counterfactualPd = predictDefaultProbability(counterfactual);
    const impact = pd - counterfactualPd;
    return {
      feature,
      label: FEATURE_LABELS[feature] ?? feature,
      impact,
      direction: impact >= 0 ? "risk-up" : "risk-down",
    } as ReasonCode;
  });

  return candidates
    .filter((item) => Math.abs(item.impact) > 0.0005)
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, 5);
}

function scoreFromPd(pd: number): number {
  const goodBadOdds = (1 - pd) / Math.max(pd, 0.0001);
  const score = 600 + 20 * Math.log2(goodBadOdds / 20);
  return Math.round(clamp(score, 300, 850));
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
  const lti = input.loanAmount / Math.max(input.annualIncome, 1);
  if (pd >= 0.22 || input.debtToIncome >= 0.67 || input.delinquencies24m >= 5 || lti >= 1.4) return "DECLINE";
  if (pd >= 0.105 || input.debtToIncome >= 0.48 || confidence < 0.66 || input.delinquencies24m >= 2) return "REVIEW";
  return "APPROVE";
}

export function assessRisk(input: ApplicationInput): RiskResult {
  const pd = predictDefaultProbability(input);
  const challengerPd = predictChallengerProbability(input);
  const disagreement = Math.abs(pd - challengerPd);
  const ood = oodSignals(input);
  const confidence = clamp(0.96 - disagreement * 1.9 - ood.length * 0.12, 0.35, 0.98);
  const lti = input.loanAmount / Math.max(input.annualIncome, 1);
  const lgd = clamp(0.34 + 0.17 * lti + 0.11 * (1 - input.incomeStability) + 0.08 * (input.cashBufferMonths < 1 ? 1 : 0), 0.25, 0.82);
  const ead = input.loanAmount;
  const expectedLoss = pd * lgd * ead;
  const score = scoreFromPd(pd);
  const decision = policyDecision(pd, input, confidence);
  const apr = clamp(8.5 + 22 * pd + 3.5 * lti + (decision === "REVIEW" ? 1.25 : 0), 8.5, 34.5);

  return {
    pd,
    challengerPd,
    disagreement,
    lgd,
    ead,
    expectedLoss,
    score,
    grade: gradeFromScore(score),
    decision,
    confidence,
    apr,
    reasons: reasonCodes(input, pd),
    modelVersion: `${artifact.name} ${artifact.version}`,
    outOfDistribution: ood,
  };
}

export function stressApplication(input: ApplicationInput, severity: "mild" | "severe") {
  const factor = severity === "severe" ? 1 : 0.55;
  const stressed: ApplicationInput = {
    ...input,
    annualIncome: input.annualIncome * (1 - 0.15 * factor),
    debtToIncome: clamp(input.debtToIncome + 0.10 * factor, 0, 0.95),
    creditUtilization: clamp(input.creditUtilization + 0.15 * factor, 0, 1.25),
    cashBufferMonths: Math.max(0, input.cashBufferMonths - 1.8 * factor),
    incomeStability: clamp(input.incomeStability - 0.16 * factor, 0, 1),
    recentCreditGrowth: clamp(input.recentCreditGrowth + 0.18 * factor, -0.5, 1.5),
  };
  return { input: stressed, result: assessRisk(stressed) };
}

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generatePortfolio(count = 420, seed = 20260829): PortfolioRecord[] {
  const random = mulberry32(seed);
  const normal = () => {
    const u = Math.max(random(), 1e-7);
    const v = Math.max(random(), 1e-7);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  return Array.from({ length: count }, (_, index) => {
    const annualIncome = clamp(Math.exp(Math.log(62_000) + 0.55 * normal()), 18_000, 320_000);
    const debtToIncome = clamp(0.16 + random() * 0.48 + Math.max(0, normal()) * 0.045, 0.05, 0.82);
    const creditUtilization = clamp(0.08 + random() * 0.76 + Math.max(0, normal()) * 0.07, 0.02, 1.12);
    const delinquencies24m = Math.min(7, Math.floor(Math.pow(random(), 3.2) * 8));
    const inquiries6m = Math.min(9, Math.floor(Math.pow(random(), 2.2) * 10));
    const application: ApplicationInput = {
      borrowerName: `Applicant ${String(index + 1).padStart(4, "0")}`,
      annualIncome,
      debtToIncome,
      creditUtilization,
      delinquencies24m,
      inquiries6m,
      oldestTradeMonths: Math.round(clamp(24 + random() * 240 + normal() * 20, 6, 360)),
      openAccounts: Math.round(clamp(3 + random() * 14 + normal() * 2, 1, 28)),
      loanAmount: Math.round(clamp(2_500 + annualIncome * (0.08 + random() * 0.62), 1_000, 110_000) / 100) * 100,
      termMonths: [12, 24, 36, 48, 60][Math.floor(random() * 5)],
      employmentYears: clamp(random() * 14 + normal(), 0, 28),
      cashBufferMonths: clamp(random() * 7 + normal() * 0.8, 0, 12),
      onTimePaymentRate: clamp(0.86 + random() * 0.14 - delinquencies24m * 0.018, 0.55, 1),
      incomeStability: clamp(0.54 + random() * 0.43 - Math.max(0, normal()) * 0.04, 0.2, 0.99),
      recentCreditGrowth: clamp(-0.12 + random() * 0.52 + inquiries6m * 0.035, -0.4, 1.2),
    };
    return { id: `CR-${20260000 + index}`, ...application, ...assessRisk(application) };
  });
}

export const modelDiagnostics = artifact;
