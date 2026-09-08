import { describe, expect, it } from "vitest";
import { assessRisk, predictDefaultProbability, stressApplication, verifyModelIntegrity } from "./engine";
import type { ApplicationInput } from "./types";

const baseline: ApplicationInput = {
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

describe("CRIX risk engine", () => {
  it("passes model-integrity verification", () => {
    expect(verifyModelIntegrity()).toBe(true);
  });

  it("returns bounded calibrated probabilities and a complete result", () => {
    const result = assessRisk({ ...baseline, debtToIncome: 0.54, creditScore: 650 });
    expect(result.pd).toBeGreaterThan(0);
    expect(result.pd).toBeLessThan(1);
    expect(result.pdHorizon).toContain("final-loan-resolution");
    expect(result.score).toBeGreaterThanOrEqual(300);
    expect(result.score).toBeLessThanOrEqual(850);
    expect(result.expectedLoss).toBeGreaterThan(0);
    expect(["APPROVE", "REVIEW", "DECLINE"]).toContain(result.decision);
    expect(result.reasons.length).toBeGreaterThan(0);
  });

  it("raises risk under severe stress", () => {
    const stressed = stressApplication(baseline, "severe");
    expect(stressed.stressed.pd).toBeGreaterThanOrEqual(stressed.baseline.pd);
    expect(stressed.stressed.expectedLoss).toBeGreaterThan(stressed.baseline.expectedLoss);
    expect(stressed.delta.pd).toBeGreaterThanOrEqual(0);
  });

  it("preserves monotonic debt-to-income behavior for a representative case", () => {
    const low = predictDefaultProbability({ ...baseline, debtToIncome: 0.15 });
    const high = predictDefaultProbability({ ...baseline, debtToIncome: 0.65 });
    expect(high).toBeGreaterThanOrEqual(low);
  });

  it("preserves monotonic bureau-score behavior for a representative case", () => {
    const weak = predictDefaultProbability({ ...baseline, creditScore: 620 });
    const strong = predictDefaultProbability({ ...baseline, creditScore: 780 });
    expect(strong).toBeLessThanOrEqual(weak);
  });

  it("rejects non-finite engine inputs even when called outside the HTTP validator", () => {
    expect(() => assessRisk({ ...baseline, annualIncome: Number.NaN })).toThrow(/Non-finite/);
  });

  it("surfaces feature values outside the real training support", () => {
    const result = assessRisk({ ...baseline, creditScore: 840 });
    expect(result.outOfDistribution).toContain("creditScore");
    expect(result.flags).toContain("OUT_OF_DISTRIBUTION");
  });
});
