import { monitorEventLoopDelay, performance } from "node:perf_hooks";
import { AggregationTemporality, InMemoryMetricExporter, MeterProvider, PeriodicExportingMetricReader, type PushMetricExporter } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-http";
import type { RiskResult } from "./risk/types";

export interface TelemetrySettings {
  enabled: boolean;
  endpoint?: string;
  exportIntervalMs: number;
}

export interface Telemetry {
  readonly enabled: boolean;
  recordHttp(method: string, route: string, statusCode: number, durationMs: number): void;
  recordAuthRejection(route: string): void;
  recordRateLimitRejection(route: string): void;
  recordScore(result: RiskResult, durationMs: number): void;
  recordStress(result: RiskResult, durationMs: number): void;
  recordBatch(results: readonly RiskResult[], batchSize: number, durationMs: number): void;
  setReadiness(ready: boolean, modelVersion?: string): void;
  forceFlush(): Promise<boolean>;
  shutdown(): Promise<void>;
}

export interface TelemetryFactoryOptions {
  exporter?: PushMetricExporter;
}

const disabledTelemetry: Telemetry = {
  enabled: false,
  recordHttp: () => undefined,
  recordAuthRejection: () => undefined,
  recordRateLimitRejection: () => undefined,
  recordScore: () => undefined,
  recordStress: () => undefined,
  recordBatch: () => undefined,
  setReadiness: () => undefined,
  forceFlush: async () => true,
  shutdown: async () => undefined,
};

const routeLabel = (route: string): string =>
  route.startsWith("/") && !route.includes("?") && route.length <= 128 ? route : "unmatched";

const statusClass = (statusCode: number): string => {
  const group = Math.floor(statusCode / 100);
  return group >= 1 && group <= 5 ? `${group}xx` : "other";
};

const disagreementBucket = (value: number): string => {
  if (value < 0.02) return "lt_0_02";
  if (value < 0.05) return "0_02_to_0_05";
  if (value < 0.08) return "0_05_to_0_08";
  return "gte_0_08";
};

export function createTelemetry(settings: TelemetrySettings, options: TelemetryFactoryOptions = {}): Telemetry {
  if (!settings.enabled) return disabledTelemetry;

  const exporter = options.exporter ?? (settings.endpoint
    ? new OTLPMetricExporter({ url: settings.endpoint, concurrencyLimit: 1, timeoutMillis: 5_000 })
    : undefined);
  const readers = exporter
    ? [new PeriodicExportingMetricReader({
      exporter,
      exportIntervalMillis: settings.exportIntervalMs,
      exportTimeoutMillis: Math.min(5_000, settings.exportIntervalMs - 1),
    })]
    : [];
  const provider = new MeterProvider({ readers });
  const meter = provider.getMeter("crix-credit-risk-intelligence", "3.2");

  const httpRequests = meter.createCounter("crix.http.requests", { description: "Completed HTTP requests" });
  const httpDuration = meter.createHistogram("crix.http.duration_ms", { unit: "ms" });
  const httpErrors = meter.createCounter("crix.http.errors", { description: "Completed 4xx/5xx requests" });
  const authRejections = meter.createCounter("crix.auth.rejections");
  const rateLimitRejections = meter.createCounter("crix.rate_limit.rejections");

  const scoreDuration = meter.createHistogram("crix.risk.score.duration_ms", { unit: "ms" });
  const stressDuration = meter.createHistogram("crix.risk.stress.duration_ms", { unit: "ms" });
  const batchDuration = meter.createHistogram("crix.risk.batch.duration_ms", { unit: "ms" });
  const batchSize = meter.createHistogram("crix.risk.batch.size");
  const decisions = meter.createCounter("crix.risk.decisions");
  const oodSignals = meter.createCounter("crix.risk.ood_signals");
  const lowConfidence = meter.createCounter("crix.risk.low_confidence");
  const disagreements = meter.createCounter("crix.risk.disagreement_buckets");

  let readiness = 0;
  let readinessModel = "unknown";
  const eventLoopDelay = monitorEventLoopDelay({ resolution: 20 });
  eventLoopDelay.enable();

  const rssGauge = meter.createObservableGauge("crix.runtime.rss_bytes", { unit: "By" });
  rssGauge.addCallback((observer) => observer.observe(process.memoryUsage().rss));
  const heapGauge = meter.createObservableGauge("crix.runtime.heap_used_bytes", { unit: "By" });
  heapGauge.addCallback((observer) => observer.observe(process.memoryUsage().heapUsed));
  const uptimeGauge = meter.createObservableGauge("crix.runtime.uptime_seconds", { unit: "s" });
  uptimeGauge.addCallback((observer) => observer.observe(process.uptime()));
  const eventLoopDelayGauge = meter.createObservableGauge("crix.runtime.event_loop_delay_p95_ms", { unit: "ms" });
  eventLoopDelayGauge.addCallback((observer) => observer.observe(Number(eventLoopDelay.percentile(95)) / 1_000_000));
  const eventLoopUtilizationGauge = meter.createObservableGauge("crix.runtime.event_loop_utilization");
  eventLoopUtilizationGauge.addCallback((observer) => observer.observe(performance.eventLoopUtilization().utilization));
  const readinessGauge = meter.createObservableGauge("crix.runtime.readiness");
  readinessGauge.addCallback((observer) => observer.observe(readiness, { model_version: readinessModel }));

  const recordAssessment = (result: RiskResult, operation: "score" | "stress" | "batch"): void => {
    const shared = {
      operation,
      model_version: result.modelVersion,
      policy_version: result.policyVersion,
    };
    decisions.add(1, { ...shared, decision: result.decision });
    disagreements.add(1, { ...shared, bucket: disagreementBucket(result.disagreement) });
    if (result.flags.includes("LOW_CONFIDENCE")) lowConfidence.add(1, shared);
    for (const feature of result.outOfDistribution) oodSignals.add(1, { ...shared, feature });
  };

  return {
    enabled: true,
    recordHttp(method, route, statusCode, durationMs) {
      const attributes = { route: routeLabel(route), method, status_class: statusClass(statusCode) };
      httpRequests.add(1, attributes);
      httpDuration.record(durationMs, attributes);
      if (statusCode >= 400) httpErrors.add(1, attributes);
    },
    recordAuthRejection(route) {
      authRejections.add(1, { route: routeLabel(route) });
    },
    recordRateLimitRejection(route) {
      rateLimitRejections.add(1, { route: routeLabel(route) });
    },
    recordScore(result, durationMs) {
      scoreDuration.record(durationMs, { model_version: result.modelVersion, policy_version: result.policyVersion });
      recordAssessment(result, "score");
    },
    recordStress(result, durationMs) {
      stressDuration.record(durationMs, { model_version: result.modelVersion, policy_version: result.policyVersion });
      recordAssessment(result, "stress");
    },
    recordBatch(results, size, durationMs) {
      batchSize.record(size);
      batchDuration.record(durationMs);
      for (const result of results) recordAssessment(result, "batch");
    },
    setReadiness(ready, modelVersion = "unknown") {
      readiness = ready ? 1 : 0;
      readinessModel = modelVersion;
    },
    async forceFlush() {
      try {
        await provider.forceFlush();
        return true;
      } catch {
        return false;
      }
    },
    async shutdown() {
      eventLoopDelay.disable();
      try {
        await provider.shutdown();
      } catch {
        // Telemetry shutdown/export failure must not alter service shutdown semantics.
      }
    },
  };
}

export function createInMemoryTelemetry(exportIntervalMs = 60_000): {
  telemetry: Telemetry;
  exporter: InMemoryMetricExporter;
} {
  const exporter = new InMemoryMetricExporter(AggregationTemporality.CUMULATIVE);
  return {
    exporter,
    telemetry: createTelemetry({ enabled: true, exportIntervalMs }, { exporter }),
  };
}
