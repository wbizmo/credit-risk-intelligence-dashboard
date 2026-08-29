export type Decision = "APPROVE" | "REVIEW" | "DECLINE";
export type StressSeverity = "mild" | "severe";

export interface ApplicationInput {
  applicationId?: string;
  annualIncome: number;
  debtToIncome: number;
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

export interface RiskResult {
  pd: number;
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
  modelVersion: string;
  policyVersion: string;
  outOfDistribution: string[];
  flags: string[];
}

export interface StressResult {
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
