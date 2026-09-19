export interface DistributionSummary {
  meanMs: number;
  p50Ms: number;
  p90Ms: number;
  p95Ms: number;
  p99Ms: number;
  opsPerSecond: number;
}

export function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) throw new Error("empty benchmark sample");
  if (!Number.isFinite(p) || p <= 0 || p > 1) throw new RangeError("percentile must be in (0, 1]");
  const index = Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1);
  return sorted[index]!;
}

export function summarizeDurations(samples: readonly number[], elapsedMs: number): DistributionSummary {
  if (samples.length === 0) throw new Error("empty benchmark sample");
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) throw new RangeError("elapsedMs must be positive");
  const sorted = [...samples].sort((a, b) => a - b);
  const total = samples.reduce((sum, sample) => sum + sample, 0);
  return {
    meanMs: total / samples.length,
    p50Ms: percentile(sorted, 0.50),
    p90Ms: percentile(sorted, 0.90),
    p95Ms: percentile(sorted, 0.95),
    p99Ms: percentile(sorted, 0.99),
    opsPerSecond: samples.length / (elapsedMs / 1_000),
  };
}
