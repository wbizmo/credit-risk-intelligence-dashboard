export type Decision = "APPROVE" | "REVIEW" | "DECLINE";
export type StressSeverity = "mild" | "severe";

export interface ApplicationInput {
  applicationId?: string;
  annualIncome: number;
  debtToIncome: number;
  creditScore: number;
  creditUtilization: number;
  delinquencies24m: number;
  inquiries6m: number;
  oldestTradeMonths: number;
  openAccounts: number;
  loanAmount: number;
  termMonths: number;
  employmentYears: number;
  cashBufferMonths: number;
  onTimePaymentRate: number;
  incomeStability: number;
  recentCreditGrowth: number;
}

export interface ReasonCode {
  feature: string;
  label: string;
  impact: number;
  direction: "risk-up" | "risk-down";
}

export interface PolicyReason {
  code: string;
  label: string;
  decision: Exclude<Decision, "APPROVE">;
}

export interface Counterfactual {
  feature: string;
  label: string;
  from: number;
  to: number;
  pdBefore: number;
  pdAfter: number;
  trainingLowerBound: number;
  trainingUpperBound: number;
}

export interface RiskResult {
  pd: number;
  pdHorizon: string;
  challengerPd: number;
  disagreement: number;
  lgd: number;
  ead: number;
  expectedLoss: number;
  expectedLossRate: number;
  score: number;
  grade: string;
  decision: Decision;
  confidence: number;
  apr: number;
  reasons: ReasonCode[];
  policyReasons: PolicyReason[];
  counterfactuals: Counterfactual[];
  modelVersion: string;
  policyVersion: string;
  outOfDistribution: string[];
  flags: string[];
}

export interface StressResult {
  method: "deterministic-borrower-sensitivity";
  scenarioVersion: string;
  severity: StressSeverity;
  input: ApplicationInput;
  baseline: RiskResult;
  stressed: RiskResult;
  delta: {
    pd: number;
    expectedLoss: number;
    score: number;
    decisionChanged: boolean;
  };
}
