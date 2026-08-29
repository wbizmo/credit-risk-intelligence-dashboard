import { describe, expect, it } from "vitest";
import { assessRisk, predictDefaultProbability, stressApplication, verifyModelIntegrity } from "./engine";
import type { ApplicationInput } from "./types";

const baseline: ApplicationInput = {
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

describe("CRIX risk engine", () => {
  it("passes model-integrity verification", () => {
    expect(verifyModelIntegrity()).toBe(true);
  });

  it("returns bounded calibrated probabilities and a complete result", () => {
    const result = assessRisk({ ...baseline, debtToIncome: 0.54, creditUtilization: 0.78 });
    expect(result.pd).toBeGreaterThan(0);
    expect(result.pd).toBeLessThan(1);
    expect(result.score).toBeGreaterThanOrEqual(300);
    expect(result.score).toBeLessThanOrEqual(850);
    expect(result.expectedLoss).toBeGreaterThan(0);
    expect(["APPROVE", "REVIEW", "DECLINE"]).toContain(result.decision);
    expect(result.reasons.length).toBeGreaterThan(0);
  });

  it("raises risk under severe stress", () => {
    const stressed = stressApplication(baseline, "severe");
    expect(stressed.stressed.pd).toBeGreaterThan(stressed.baseline.pd);
    expect(stressed.stressed.expectedLoss).toBeGreaterThan(stressed.baseline.expectedLoss);
    expect(stressed.delta.pd).toBeGreaterThan(0);
  });

  it("preserves monotonic utilization behaviour for a representative case", () => {
    const low = predictDefaultProbability({ ...baseline, creditUtilization: 0.2 });
    const high = predictDefaultProbability({ ...baseline, creditUtilization: 0.9 });
    expect(high).toBeGreaterThanOrEqual(low);
  });

  it("rejects non-finite engine inputs even when called outside the HTTP validator", () => {
    expect(() => assessRisk({ ...baseline, annualIncome: Number.NaN })).toThrow(/Non-finite/);
  });

  it("surfaces out-of-distribution inputs instead of silently trusting them", () => {
    const result = assessRisk({ ...baseline, annualIncome: 900_000 });
    expect(result.outOfDistribution).toContain("annualIncome");
    expect(result.flags).toContain("OUT_OF_DISTRIBUTION");
  });
});
