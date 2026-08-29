import { randomUUID, timingSafeEqual } from "node:crypto";
import cors from "@fastify/cors";
import helmet from "@fastify/helmet";
import rateLimit from "@fastify/rate-limit";
import swagger from "@fastify/swagger";
import swaggerUi from "@fastify/swagger-ui";
import Fastify, { type FastifyRequest } from "fastify";
import { loadConfig, type AppConfig } from "./config";
import { assessRisk, modelMetadata, stressApplication, verifyModelIntegrity } from "./risk/engine";
import type { ApplicationInput, StressSeverity } from "./risk/types";
import {
  API_MAJOR_PATH,
  API_VERSION,
  applicationSchema,
  batchRequestSchema,
  batchResponseSchema,
  errorSchema,
  healthResponseSchema,
  modelResponseSchema,
  readyResponseSchema,
  scoreResponseSchema,
  stressRequestSchema,
  stressResponseSchema,
} from "./schemas";

const secureEqual = (actual: string, expected: string): boolean => {
  const a = Buffer.from(actual);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
};

const protectedRoute = (request: FastifyRequest) => request.url === API_MAJOR_PATH || request.url.startsWith(`${API_MAJOR_PATH}/`);

interface ValidationIssue {
  instancePath?: string;
  keyword: string;
  message?: string;
}

interface RequestErrorShape {
  validation?: ValidationIssue[];
  statusCode?: number;
  message?: string;
}

export async function buildApp(config: AppConfig = loadConfig()) {
  const app = Fastify({
    logger: config.environment === "test" ? false : {
      level: config.logLevel,
      redact: ["req.headers.authorization", "req.headers.x-api-key"],
    },
    genReqId: () => randomUUID(),
    bodyLimit: 64 * 1024,
    requestTimeout: 10_000,
    connectionTimeout: 10_000,
    keepAliveTimeout: 72_000,
    maxParamLength: 256,
  });

  await app.register(helmet, { contentSecurityPolicy: false });
  await app.register(rateLimit, {
    max: config.rateLimitMax,
    timeWindow: "1 minute",
    keyGenerator: (request) => request.ip,
    errorResponseBuilder: (request) => ({
      error: "RATE_LIMITED",
      message: "Too many requests. Retry after the current rate-limit window.",
      requestId: request.id,
    }),
  });

  if (config.corsOrigins.length > 0) {
    await app.register(cors, {
      origin: config.corsOrigins,
      methods: ["GET", "POST", "OPTIONS"],
      allowedHeaders: ["content-type", "x-api-key", "x-request-id"],
    });
  }

  await app.register(swagger, {
    openapi: {
      info: {
        title: "CRIX Credit Risk Intelligence API",
        version: API_VERSION,
        description: "Stateless credit-risk scoring, model diagnostics, policy decisioning and stress testing. The bundled model is trained on synthetic data and is for engineering/model-risk demonstration only.",
      },
      tags: [
        { name: "System", description: "Service health and discoverability" },
        { name: "Risk", description: "Credit-risk scoring and stress testing" },
        { name: "Model", description: "Model and policy metadata" },
      ],
      components: {
        securitySchemes: {
          ApiKeyAuth: { type: "apiKey", in: "header", name: "x-api-key", description: "Required only when CRIX_API_KEY is configured on the server." },
        },
      },
    },
  });

  await app.register(swaggerUi, {
    routePrefix: "/docs",
    staticCSP: true,
    uiConfig: { docExpansion: "list", deepLinking: true },
  });

  const modelReady = verifyModelIntegrity();

  app.addHook("onRequest", async (request, reply) => {
    reply.header("x-request-id", request.id);
    reply.header("cache-control", "no-store");

    if (!config.apiKey || !protectedRoute(request)) return;
    const supplied = request.headers["x-api-key"];
    if (typeof supplied !== "string" || !secureEqual(supplied, config.apiKey)) {
      return reply.code(401).send({
        error: "UNAUTHORIZED",
        message: "A valid x-api-key header is required.",
        requestId: request.id,
      });
    }
  });

  app.get("/", {
    schema: { tags: ["System"], summary: "API index" },
  }, async () => ({
    service: "CRIX Credit Risk Intelligence API",
    apiVersion: API_VERSION,
    status: "ok",
    authentication: config.apiKey ? "api-key" : "public-demo",
    endpoints: {
      health: "/health",
      readiness: "/ready",
      docs: "/docs",
      openapi: "/openapi.json",
      score: `${API_MAJOR_PATH}/risk/score`,
    },
  }));

  app.get("/health", {
    config: { rateLimit: { max: 600, timeWindow: "1 minute" } },
    schema: { tags: ["System"], summary: "Liveness probe", response: { 200: healthResponseSchema } },
  }, async () => ({
    status: "ok",
    service: "crix-credit-risk-intelligence-api",
    version: API_VERSION,
    uptimeSeconds: Math.floor(process.uptime()),
    timestamp: new Date().toISOString(),
  }));

  app.get("/ready", {
    config: { rateLimit: { max: 600, timeWindow: "1 minute" } },
    schema: { tags: ["System"], summary: "Readiness probe", response: { 200: readyResponseSchema, 503: readyResponseSchema } },
  }, async (_request, reply) => {
    if (!modelReady) return reply.code(503).send({ status: "not-ready", modelLoaded: false, version: API_VERSION });
    return { status: "ready", modelLoaded: true, version: API_VERSION, model: modelMetadata().name };
  });

  app.get("/openapi.json", {
    schema: { tags: ["System"], summary: "OpenAPI document" },
  }, async () => app.swagger());

  app.get(`${API_MAJOR_PATH}/model`, {
    schema: {
      tags: ["Model"],
      summary: "Model and policy metadata",
      description: "Returns model version, calibration, held-out validation metrics, feature metadata, diagnostics and current policy thresholds. Tree internals are intentionally not returned by the API.",
      response: { 200: modelResponseSchema, 401: errorSchema, 429: errorSchema },
    },
  }, async (request) => ({ requestId: request.id, apiVersion: API_VERSION, ...modelMetadata() }));

  app.post<{ Body: ApplicationInput }>(`${API_MAJOR_PATH}/risk/score`, {
    config: { rateLimit: { max: 60, timeWindow: "1 minute" } },
    schema: {
      tags: ["Risk"],
      summary: "Score a single credit application",
      description: "Runs the calibrated monotonic champion, transparent challenger, confidence/OOD checks, LGD/EAD/expected-loss estimation, reason codes, and independent policy decision.",
      body: applicationSchema,
      response: { 200: scoreResponseSchema, 400: errorSchema, 401: errorSchema, 429: errorSchema },
    },
  }, async (request) => ({ requestId: request.id, apiVersion: API_VERSION, result: assessRisk(request.body) }));

  app.post<{ Body: { application: ApplicationInput; severity: StressSeverity } }>(`${API_MAJOR_PATH}/risk/stress`, {
    config: { rateLimit: { max: 30, timeWindow: "1 minute" } },
    schema: {
      tags: ["Risk"],
      summary: "Stress-test an application",
      description: "Applies a deterministic mild or severe borrower shock and re-runs the complete model + policy lifecycle.",
      body: stressRequestSchema,
      response: { 200: stressResponseSchema, 400: errorSchema, 401: errorSchema, 429: errorSchema },
    },
  }, async (request) => ({ requestId: request.id, apiVersion: API_VERSION, ...stressApplication(request.body.application, request.body.severity) }));

  app.post<{ Body: { applications: ApplicationInput[] } }>(`${API_MAJOR_PATH}/risk/batch`, {
    config: { rateLimit: { max: 10, timeWindow: "1 minute" } },
    schema: {
      tags: ["Risk"],
      summary: "Score a bounded batch",
      description: "Scores up to 50 applications synchronously. The hard batch bound protects CPU/memory on the public demo deployment.",
      body: batchRequestSchema,
      response: { 200: batchResponseSchema, 400: errorSchema, 401: errorSchema, 429: errorSchema },
    },
  }, async (request) => {
    const results = request.body.applications.map((application, index) => ({
      index,
      ...(application.applicationId ? { applicationId: application.applicationId } : {}),
      result: assessRisk(application),
    }));
    const counts = results.reduce((acc, item) => {
      acc[item.result.decision] += 1;
      return acc;
    }, { APPROVE: 0, REVIEW: 0, DECLINE: 0 });
    const totalExpectedLoss = results.reduce((sum, item) => sum + item.result.expectedLoss, 0);
    const averagePd = results.reduce((sum, item) => sum + item.result.pd, 0) / results.length;

    return {
      requestId: request.id,
      apiVersion: API_VERSION,
      results,
      summary: { count: results.length, decisions: counts, averagePd, totalExpectedLoss },
    };
  });

  app.setNotFoundHandler(async (request, reply) => reply.code(404).send({
    error: "NOT_FOUND",
    message: "Route not found. See /docs for the API contract.",
    requestId: request.id,
  }));

  app.setErrorHandler(async (error, request, reply) => {
    const requestError = error as RequestErrorShape;

    if (requestError.validation) {
      return reply.code(400).send({
        error: "VALIDATION_ERROR",
        message: "Request validation failed.",
        requestId: request.id,
        details: requestError.validation.map((item) => ({ instancePath: item.instancePath, keyword: item.keyword, message: item.message })),
      });
    }

    if (requestError.statusCode === 429) {
      return reply.code(429).send({ error: "RATE_LIMITED", message: "Too many requests.", requestId: request.id });
    }

    const statusCode = requestError.statusCode && requestError.statusCode < 500 ? requestError.statusCode : 500;
    request.log.error({ err: error }, "Unhandled request error");
    return reply.code(statusCode).send({
      error: statusCode < 500 ? "REQUEST_ERROR" : "INTERNAL_ERROR",
      message: statusCode < 500 ? (requestError.message ?? "Request failed.") : "The server could not process this request.",
      requestId: request.id,
    });
  });

  return app;
}
