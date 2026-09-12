import { performance } from "node:perf_hooks";
import { assessRisk, stressApplication } from "./engine";
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

interface BenchmarkResult {
  name: string;
  iterations: number;
  elapsedMs: number;
  meanMs: number;
  heapDeltaBytes: number;
}

function run(name: string, iterations: number, operation: (index: number) => unknown): BenchmarkResult {
  for (let index = 0; index < Math.min(100, iterations); index += 1) operation(index);
  const heapBefore = process.memoryUsage().heapUsed;
  const started = performance.now();
  for (let index = 0; index < iterations; index += 1) operation(index);
  const elapsedMs = performance.now() - started;
  const heapDeltaBytes = process.memoryUsage().heapUsed - heapBefore;
  return { name, iterations, elapsedMs, meanMs: elapsedMs / iterations, heapDeltaBytes };
}

const results = [
  run("single-assessment", 2_000, (index) => assessRisk({
    ...baseline,
    debtToIncome: 0.20 + (index % 50) / 100,
    creditScore: 610 + (index % 150),
  })),
  run("severe-sensitivity", 500, (index) => stressApplication({
    ...baseline,
    loanAmount: 12_000 + (index % 20) * 1_000,
  }, "severe")),
  run("50-item-batch-equivalent", 100, (batchIndex) => {
    for (let index = 0; index < 50; index += 1) {
      assessRisk({
        ...baseline,
        debtToIncome: 0.18 + ((batchIndex + index) % 60) / 100,
        creditScore: 600 + ((batchIndex * 7 + index) % 180),
      });
    }
  }),
];

console.log(JSON.stringify({
  node: process.version,
  model: "CRIX-MonoBoost 2.0",
  note: "Evidence-only benchmark: compare like-for-like CI runs; absolute wall-clock thresholds are intentionally not treated as model correctness.",
  results,
}, null, 2));
