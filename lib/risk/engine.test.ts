import { describe, expect, it } from "vitest";
import { assessRisk, predictDefaultProbability, stressApplication } from "./engine";
import type { ApplicationInput } from "./types";

const baseline: ApplicationInput = {
  borrowerName: "Test Applicant",
  annualIncome: 85000,
  debtToIncome: 0.28,
  creditUtilization: 0.3,
  delinquencies24m: 0,
  inquiries6m: 1,
  oldestTradeMonths: 96,
  openAccounts: 7,
  loanAmount: 24000,
  termMonths: 36,
  employmentYears: 5,
  cashBufferMonths: 3,
  onTimePaymentRate: 0.98,
  incomeStability: 0.82,
  recentCreditGrowth: 0.08,
};

describe("credit risk engine", () => {
  it("returns a bounded calibrated probability", () => {
    const pd = predictDefaultProbability(baseline);
    expect(pd).toBeGreaterThan(0);
    expect(pd).toBeLessThan(1);
  });

  it("raises risk when material credit stress is introduced", () => {
    const base = assessRisk(baseline);
    const stressed = stressApplication(baseline, "severe").result;
    expect(stressed.pd).toBeGreaterThan(base.pd);
    expect(stressed.expectedLoss).toBeGreaterThan(base.expectedLoss);
  });

  it("preserves monotonic behaviour for utilization in a representative case", () => {
    const low = predictDefaultProbability({ ...baseline, creditUtilization: 0.2 });
    const high = predictDefaultProbability({ ...baseline, creditUtilization: 0.9 });
    expect(high).toBeGreaterThanOrEqual(low);
  });

  it("emits a complete underwriting result", () => {
    const result = assessRisk(baseline);
    expect(result.score).toBeGreaterThanOrEqual(300);
    expect(result.score).toBeLessThanOrEqual(850);
    expect(["APPROVE", "REVIEW", "DECLINE"]).toContain(result.decision);
    expect(result.reasons.length).toBeGreaterThan(0);
  });
});
