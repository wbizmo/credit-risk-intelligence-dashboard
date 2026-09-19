import os from "node:os";
import { performance } from "node:perf_hooks";
import { summarizeDurations } from "./benchmark-utils";
import {
  assessRisk,
  assessRiskWithComplexity,
  benchmarkBaseMargin,
  benchmarkChallengerProbability,
  benchmarkChampionProbability,
  benchmarkExpectedFullExplanationTreeVisits,
  benchmarkReferenceProbability,
  benchmarkTreeDensity,
  stressApplication,
  verifyModelIntegrity,
} from "./engine";
import {
  compileModelArtifact,
  runtimeArtifactForBenchmark,
  type ComplexityCounters,
  type ModelTree,
} from "./runtime";
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
  warmupIterations: number;
  elapsedMs: number;
  meanMs: number;
  p50Ms: number;
  p90Ms: number;
  p95Ms: number;
  p99Ms: number;
  opsPerSecond: number;
  heapDeltaBytes: number;
  rssDeltaBytes: number;
  complexity?: ComplexityCounters;
}

function run(
  name: string,
  iterations: number,
  warmupIterations: number,
  operation: (index: number) => unknown,
  complexity?: ComplexityCounters,
): BenchmarkResult {
  for (let index = 0; index < warmupIterations; index += 1) operation(index);

  const memoryBefore = process.memoryUsage();
  const samples = new Array<number>(iterations);
  const started = performance.now();

  for (let index = 0; index < iterations; index += 1) {
    const operationStarted = performance.now();
    operation(index);
    samples[index] = performance.now() - operationStarted;
  }

  const elapsedMs = performance.now() - started;
  const memoryAfter = process.memoryUsage();
  return {
    name,
    iterations,
    warmupIterations,
    elapsedMs,
    ...summarizeDurations(samples, elapsedMs),
    heapDeltaBytes: memoryAfter.heapUsed - memoryBefore.heapUsed,
    rssDeltaBytes: memoryAfter.rss - memoryBefore.rss,
    ...(complexity ? { complexity } : {}),
  };
}

type NumericStorage = ArrayLike<number>;
interface StorageTree {
  left: NumericStorage;
  right: NumericStorage;
  feature: NumericStorage;
  threshold: NumericStorage;
  defaultLeft: NumericStorage;
}

function toPackedTree(tree: ModelTree): StorageTree {
  return {
    left: [...tree.left],
    right: [...tree.right],
    feature: [...tree.feature],
    threshold: [...tree.threshold],
    defaultLeft: [...tree.defaultLeft],
  };
}

function toTypedTree(tree: ModelTree): StorageTree {
  return {
    left: Int32Array.from(tree.left),
    right: Int32Array.from(tree.right),
    feature: Int16Array.from(tree.feature),
    threshold: Float64Array.from(tree.threshold),
    defaultLeft: Uint8Array.from(tree.defaultLeft),
  };
}

function evaluateStorageTrees(trees: readonly StorageTree[], vector: readonly number[]): number {
  let margin = benchmarkBaseMargin();
  for (const tree of trees) {
    let node = 0;
    while (tree.left[node] !== -1) {
      const featureIndex = tree.feature[node]!;
      const value = vector[featureIndex]!;
      node = (Number.isNaN(value) ? Boolean(tree.defaultLeft[node]) : value < tree.threshold[node]!)
        ? tree.left[node]!
        : tree.right[node]!;
    }
    margin += tree.threshold[node]!;
  }
  return margin;
}

function benchmarkStorageRepresentation() {
  const artifact = runtimeArtifactForBenchmark();
  const packed = artifact.trees.map(toPackedTree);
  const typed = artifact.trees.map(toTypedTree);
  const vector = [baseline.debtToIncome, baseline.loanAmount / baseline.annualIncome, baseline.creditScore, baseline.employmentYears];
  const iterations = 25_000;
  const warmupIterations = 5_000;

  const measure = (trees: readonly StorageTree[]) => {
    for (let index = 0; index < warmupIterations; index += 1) evaluateStorageTrees(trees, vector);
    const started = performance.now();
    for (let index = 0; index < iterations; index += 1) evaluateStorageTrees(trees, vector);
    const elapsedMs = performance.now() - started;
    return {
      iterations,
      warmupIterations,
      elapsedMs,
      opsPerSecond: iterations / (elapsedMs / 1_000),
    };
  };

  const packedResult = measure(packed);
  const typedResult = measure(typed);
  return {
    actualModelTrees: artifact.trees.length,
    actualModelNodes: artifact.trees.reduce((sum, tree) => sum + tree.left.length, 0),
    selectedRuntimeRepresentation: "packed-js-arrays",
    packed: packedResult,
    typed: typedResult,
    note: "The production compiler keeps packed frozen JS arrays. This benchmark records the actual deployed artifact under the current Node runtime so representation choice remains evidence-backed.",
  };
}

const complexityProbe = assessRiskWithComplexity(baseline);
const treeDensity = benchmarkTreeDensity();
const fullExplanationTreeVisits = benchmarkExpectedFullExplanationTreeVisits();
const sparseExplanationTreeVisits = complexityProbe.complexity.explanationTreeVisits;

const results: BenchmarkResult[] = [
  run("single-assessment", 2_000, 250, (index) => assessRisk({
    ...baseline,
    debtToIncome: 0.20 + (index % 50) / 100,
    creditScore: 610 + (index % 150),
  }), complexityProbe.complexity),
  run("champion-probability-only", 8_000, 1_000, (index) => benchmarkChampionProbability({
    ...baseline,
    debtToIncome: 0.20 + (index % 50) / 100,
    creditScore: 610 + (index % 150),
  })),
  run("reference-champion-probability", 2_000, 250, (index) => benchmarkReferenceProbability({
    ...baseline,
    debtToIncome: 0.20 + (index % 50) / 100,
    creditScore: 610 + (index % 150),
  })),
  run("challenger-probability", 8_000, 1_000, (index) => benchmarkChallengerProbability({
    ...baseline,
    loanAmount: 12_000 + (index % 20) * 1_000,
  })),
  run("full-explanation-counterfactual-path", 1_500, 200, (index) => assessRiskWithComplexity({
    ...baseline,
    debtToIncome: 0.18 + (index % 60) / 100,
    loanAmount: 10_000 + (index % 30) * 1_000,
  })),
  run("mild-sensitivity", 400, 50, (index) => stressApplication({
    ...baseline,
    loanAmount: 12_000 + (index % 20) * 1_000,
  }, "mild")),
  run("severe-sensitivity", 400, 50, (index) => stressApplication({
    ...baseline,
    loanAmount: 12_000 + (index % 20) * 1_000,
  }, "severe")),
  run("50-item-batch-equivalent", 100, 10, (batchIndex) => {
    for (let index = 0; index < 50; index += 1) {
      assessRisk({
        ...baseline,
        debtToIncome: 0.18 + ((batchIndex + index) % 60) / 100,
        creditScore: 600 + ((batchIndex * 7 + index) % 180),
      });
    }
  }),
  run("model-artifact-compilation", 250, 25, () => compileModelArtifact(runtimeArtifactForBenchmark())),
  run("model-integrity-verification-cached", 1_000, 100, () => {
    if (!verifyModelIntegrity()) throw new Error("Model integrity verification failed during benchmark");
  }),
];

if (complexityProbe.complexity.championTreeVisits !== treeDensity[0]!.totalTrees) {
  throw new Error("Champion tree-visit budget changed unexpectedly");
}
if (sparseExplanationTreeVisits >= fullExplanationTreeVisits) {
  throw new Error("Sparse explanation traversal no longer reduces deployed-model tree visits");
}
if (complexityProbe.complexity.challengerFeatureOps !== treeDensity.length) {
  throw new Error("Challenger complexity is no longer linear in the champion feature count");
}

console.log(JSON.stringify({
  node: process.version,
  platform: process.platform,
  arch: process.arch,
  cpu: os.cpus()[0]?.model ?? "unknown",
  cpuCount: os.cpus().length,
  model: "CRIX-MonoBoost 2.0",
  benchmarkPolicy: {
    wallClock: "evidence-only; no brittle CI timing threshold",
    deterministicComplexity: "hard-gated",
    batchComplexity: "O(B * score_cost)",
  },
  explanationDensity: {
    byFeature: treeDensity,
    fullRescoreTreeVisitsPerAssessment: fullExplanationTreeVisits,
    sparseTreeVisitsPerAssessment: sparseExplanationTreeVisits,
    reductionFraction: 1 - sparseExplanationTreeVisits / fullExplanationTreeVisits,
  },
  storageRepresentation: benchmarkStorageRepresentation(),
  results,
}, null, 2));
