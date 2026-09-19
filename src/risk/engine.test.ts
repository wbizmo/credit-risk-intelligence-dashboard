import { describe, expect, it } from "vitest";
import {
  assessRisk,
  assessRiskReferenceForTest,
  assessRiskWithComplexity,
  benchmarkExpectedFullExplanationTreeVisits,
  benchmarkTreeDensity,
  predictDefaultProbability,
  stressApplication,
  verifyModelIntegrity,
} from "./engine";
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
    expect(Array.isArray(result.policyReasons)).toBe(true);
    expect(Array.isArray(result.counterfactuals)).toBe(true);
  });

  it("keeps policy reasons separate from champion-model reasons", () => {
    const result = assessRisk({
      ...baseline,
      creditScore: 570,
      delinquencies24m: 6,
    });

    expect(result.decision).toBe("DECLINE");
    expect(result.policyReasons.map((reason) => reason.code)).toContain("CREDIT_SCORE_DECLINE_THRESHOLD");
    expect(result.policyReasons.map((reason) => reason.code)).toContain("DELINQUENCY_DECLINE_THRESHOLD");
    expect(result.reasons.every((reason) => ["debtToIncome", "loanToIncome", "creditScore", "employmentYears"].includes(reason.feature))).toBe(true);
  });

  it("returns only bounded lower-risk model counterfactuals", () => {
    const result = assessRisk({
      ...baseline,
      debtToIncome: 0.58,
      loanAmount: 50_000,
      creditScore: 630,
      employmentYears: 1,
    });

    for (const item of result.counterfactuals) {
      expect(["debtToIncome", "loanToIncome", "creditScore", "employmentYears"]).toContain(item.feature);
      expect(item.pdAfter).toBeLessThan(item.pdBefore);
      expect(Number.isFinite(item.to)).toBe(true);
      expect(item.to).toBeGreaterThanOrEqual(item.trainingLowerBound);
      expect(item.to).toBeLessThanOrEqual(item.trainingUpperBound);
    }
  });

  it("is deterministic for identical inputs", () => {
    const first = assessRisk({ ...baseline, debtToIncome: 0.41 });
    const second = assessRisk({ ...baseline, debtToIncome: 0.41 });
    expect(second).toEqual(first);
  });

  it("raises risk under severe deterministic borrower sensitivity without mutating the input", () => {
    const original = { ...baseline };
    const stressed = stressApplication(baseline, "severe");
    expect(stressed.method).toBe("deterministic-borrower-sensitivity");
    expect(stressed.scenarioVersion).toBe("CRIX-Sensitivity 1.0");
    expect(stressed.stressed.pd).toBeGreaterThanOrEqual(stressed.baseline.pd);
    expect(stressed.stressed.expectedLoss).toBeGreaterThan(stressed.baseline.expectedLoss);
    expect(stressed.delta.pd).toBeGreaterThanOrEqual(0);
    expect(baseline).toEqual(original);
  });

  it("keeps severe sensitivity at least as adverse as mild", () => {
    const mild = stressApplication(baseline, "mild");
    const severe = stressApplication(baseline, "severe");
    expect(severe.input.annualIncome).toBeLessThanOrEqual(mild.input.annualIncome);
    expect(severe.input.debtToIncome).toBeGreaterThanOrEqual(mild.input.debtToIncome);
    expect(severe.input.creditUtilization).toBeGreaterThanOrEqual(mild.input.creditUtilization);
    expect(severe.input.cashBufferMonths).toBeLessThanOrEqual(mild.input.cashBufferMonths);
    expect(severe.input.incomeStability).toBeLessThanOrEqual(mild.input.incomeStability);
    expect(severe.input.recentCreditGrowth).toBeGreaterThanOrEqual(mild.input.recentCreditGrowth);
  });

  it("preserves monotonic debt-to-income behavior for a representative case", () => {
    const low = predictDefaultProbability({ ...baseline, debtToIncome: 0.15 });
    const high = predictDefaultProbability({ ...baseline, debtToIncome: 0.65 });
    expect(high).toBeGreaterThanOrEqual(low);
  });

  it("preserves monotonic requested-loan-to-income behavior for a representative case", () => {
    const low = predictDefaultProbability({ ...baseline, loanAmount: 8_500 });
    const high = predictDefaultProbability({ ...baseline, loanAmount: 55_000 });
    expect(high).toBeGreaterThanOrEqual(low);
  });

  it("preserves monotonic bureau-score behavior for a representative case", () => {
    const weak = predictDefaultProbability({ ...baseline, creditScore: 620 });
    const strong = predictDefaultProbability({ ...baseline, creditScore: 780 });
    expect(strong).toBeLessThanOrEqual(weak);
  });

  it("preserves monotonic employment-tenure behavior for a representative case", () => {
    const short = predictDefaultProbability({ ...baseline, employmentYears: 0.5 });
    const long = predictDefaultProbability({ ...baseline, employmentYears: 10 });
    expect(long).toBeLessThanOrEqual(short);
  });

  it("rejects non-finite engine inputs even when called outside the HTTP validator", () => {
    expect(() => assessRisk({ ...baseline, annualIncome: Number.NaN })).toThrow(/Non-finite/);
  });

  it("surfaces feature values outside the real training support", () => {
    const result = assessRisk({ ...baseline, creditScore: 840 });
    expect(result.outOfDistribution).toContain("creditScore");
    expect(result.flags).toContain("OUT_OF_DISTRIBUTION");
  });

  it("preserves the complete precompiled scoring contract across a deterministic corpus", () => {
    for (let index = 0; index < 120; index += 1) {
      const input: ApplicationInput = {
        ...baseline,
        debtToIncome: 0.08 + ((index * 17) % 60) / 100,
        creditScore: 575 + ((index * 19) % 230),
        loanAmount: 8_000 + ((index * 3_700) % 48_000),
        employmentYears: 0.5 + ((index * 7) % 20) / 2,
      };
      const compiled = assessRisk(input);
      const reference = assessRiskReferenceForTest(input);

      expect(compiled.pd).toBeCloseTo(reference.pd, 14);
      expect(compiled.challengerPd).toBeCloseTo(reference.challengerPd, 14);
      expect(compiled.confidence).toBeCloseTo(reference.confidence, 14);
      expect(compiled.grade).toBe(reference.grade);
      expect(compiled.score).toBe(reference.score);
      expect(compiled.decision).toBe(reference.decision);
      expect(compiled.policyReasons).toEqual(reference.policyReasons);
      expect(compiled.flags).toEqual(reference.flags);
      expect(compiled.outOfDistribution).toEqual(reference.outOfDistribution);
      expect(compiled.reasons.map((item) => item.feature)).toEqual(reference.reasons.map((item) => item.feature));
      expect(compiled.counterfactuals.map((item) => item.feature)).toEqual(reference.counterfactuals.map((item) => item.feature));

      for (let reasonIndex = 0; reasonIndex < compiled.reasons.length; reasonIndex += 1) {
        expect(compiled.reasons[reasonIndex]!.impact).toBeCloseTo(reference.reasons[reasonIndex]!.impact, 12);
      }
      for (let counterfactualIndex = 0; counterfactualIndex < compiled.counterfactuals.length; counterfactualIndex += 1) {
        expect(compiled.counterfactuals[counterfactualIndex]!.pdAfter)
          .toBeCloseTo(reference.counterfactuals[counterfactualIndex]!.pdAfter, 12);
      }
    }
  });

  it("preserves loan-to-income counterfactual semantics through loanAmount perturbation", () => {
    const input: ApplicationInput = { ...baseline, loanAmount: 50_000 };
    const originalLoanAmount = input.loanAmount;
    const compiled = assessRisk(input);
    const reference = assessRiskReferenceForTest(input);
    const compiledLoanToIncome = compiled.counterfactuals.find((item) => item.feature === "loanToIncome");
    const referenceLoanToIncome = reference.counterfactuals.find((item) => item.feature === "loanToIncome");

    expect(compiledLoanToIncome).toBeDefined();
    expect(referenceLoanToIncome).toBeDefined();
    expect(compiledLoanToIncome?.to).toBeCloseTo(referenceLoanToIncome!.to, 14);
    expect(compiledLoanToIncome?.pdAfter).toBeCloseTo(referenceLoanToIncome!.pdAfter, 12);
    expect(input.loanAmount).toBe(originalLoanAmount);
  });

  it("uses tree-sparse explanation traversal for the deployed model", () => {
    const { complexity } = assessRiskWithComplexity(baseline);
    const density = benchmarkTreeDensity();
    const expectedSparseVisits = density.reduce((sum, item) => sum + item.trees, 0);

    expect(complexity.explanationTreeVisits).toBe(expectedSparseVisits);
    expect(complexity.explanationTreeVisits).toBeLessThan(benchmarkExpectedFullExplanationTreeVisits());
    expect(complexity.championTreeVisits).toBe(density[0]!.totalTrees);
  });

  it("creates fresh complexity counters without changing scoring outputs", () => {
    const ordinary = assessRisk(baseline);
    const first = assessRiskWithComplexity(baseline);
    const second = assessRiskWithComplexity(baseline);

    expect(first.result).toEqual(ordinary);
    expect(second.result).toEqual(ordinary);
    expect(second.complexity).toEqual(first.complexity);
  });

  it("keeps diagnostic work linear across 1/10/25/50-item batch equivalents", () => {
    const counts = [1, 10, 25, 50];
    const one = assessRiskWithComplexity(baseline).complexity;

    for (const count of counts) {
      let championTreeVisits = 0;
      let explanationTreeVisits = 0;
      let challengerFeatureOps = 0;

      for (let index = 0; index < count; index += 1) {
        const { complexity } = assessRiskWithComplexity({
          ...baseline,
          debtToIncome: baseline.debtToIncome + index / 10_000,
        });
        championTreeVisits += complexity.championTreeVisits;
        explanationTreeVisits += complexity.explanationTreeVisits;
        challengerFeatureOps += complexity.challengerFeatureOps;
      }

      expect(championTreeVisits).toBe(one.championTreeVisits * count);
      expect(explanationTreeVisits).toBe(one.explanationTreeVisits * count);
      expect(challengerFeatureOps).toBe(one.challengerFeatureOps * count);
    }
  });

});
