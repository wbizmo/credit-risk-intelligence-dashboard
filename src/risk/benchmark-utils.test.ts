import { describe, expect, it } from "vitest";
import { percentile, summarizeDurations } from "./benchmark-utils";

describe("benchmark utilities", () => {
  it("calculates nearest-rank percentiles deterministically", () => {
    const sorted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(sorted, 0.50)).toBe(5);
    expect(percentile(sorted, 0.90)).toBe(9);
    expect(percentile(sorted, 0.95)).toBe(10);
    expect(percentile(sorted, 0.99)).toBe(10);
  });

  it("rejects empty samples and invalid percentile requests", () => {
    expect(() => percentile([], 0.5)).toThrow(/empty/);
    expect(() => percentile([1], 0)).toThrow(/percentile/);
    expect(() => percentile([1], 1.01)).toThrow(/percentile/);
  });

  it("summarizes latency and throughput from explicit samples", () => {
    const summary = summarizeDurations([1, 2, 3, 4], 20);
    expect(summary.meanMs).toBe(2.5);
    expect(summary.p50Ms).toBe(2);
    expect(summary.p95Ms).toBe(4);
    expect(summary.opsPerSecond).toBe(200);
  });
});
