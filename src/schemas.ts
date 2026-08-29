export const API_VERSION = "2.5.0";
export const API_MAJOR_PATH = "/api/v2";
export const MAX_BATCH_SIZE = 50;

export const applicationSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "annualIncome", "debtToIncome", "creditUtilization", "delinquencies24m",
    "inquiries6m", "oldestTradeMonths", "openAccounts", "loanAmount", "termMonths",
    "employmentYears", "cashBufferMonths", "onTimePaymentRate", "incomeStability", "recentCreditGrowth"
  ],
  properties: {
    applicationId: { type: "string", minLength: 1, maxLength: 128, pattern: "^[A-Za-z0-9._:-]+$" },
    annualIncome: { type: "number", exclusiveMinimum: 0, maximum: 10_000_000 },
    debtToIncome: { type: "number", minimum: 0, maximum: 2 },
    creditUtilization: { type: "number", minimum: 0, maximum: 2 },
    delinquencies24m: { type: "integer", minimum: 0, maximum: 50 },
    inquiries6m: { type: "integer", minimum: 0, maximum: 50 },
    oldestTradeMonths: { type: "integer", minimum: 0, maximum: 1_000 },
    openAccounts: { type: "integer", minimum: 1, maximum: 100 },
    loanAmount: { type: "number", exclusiveMinimum: 0, maximum: 10_000_000 },
    termMonths: { type: "integer", minimum: 1, maximum: 360 },
    employmentYears: { type: "number", minimum: 0, maximum: 80 },
    cashBufferMonths: { type: "number", minimum: 0, maximum: 120 },
    onTimePaymentRate: { type: "number", minimum: 0, maximum: 1 },
    incomeStability: { type: "number", minimum: 0, maximum: 1 },
    recentCreditGrowth: { type: "number", minimum: -1, maximum: 3 }
  }
} as const;

export const reasonCodeSchema = {
  type: "object",
  additionalProperties: false,
  required: ["feature", "label", "impact", "direction"],
  properties: {
    feature: { type: "string" },
    label: { type: "string" },
    impact: { type: "number" },
    direction: { type: "string", enum: ["risk-up", "risk-down"] }
  }
} as const;

export const riskResultSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "pd", "challengerPd", "disagreement", "lgd", "ead", "expectedLoss", "expectedLossRate",
    "score", "grade", "decision", "confidence", "apr", "reasons", "modelVersion", "policyVersion",
    "outOfDistribution", "flags"
  ],
  properties: {
    pd: { type: "number", minimum: 0, maximum: 1 },
    challengerPd: { type: "number", minimum: 0, maximum: 1 },
    disagreement: { type: "number", minimum: 0, maximum: 1 },
    lgd: { type: "number", minimum: 0, maximum: 1 },
    ead: { type: "number", minimum: 0 },
    expectedLoss: { type: "number", minimum: 0 },
    expectedLossRate: { type: "number", minimum: 0, maximum: 1 },
    score: { type: "integer", minimum: 300, maximum: 850 },
    grade: { type: "string" },
    decision: { type: "string", enum: ["APPROVE", "REVIEW", "DECLINE"] },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    apr: { type: "number", minimum: 0 },
    reasons: { type: "array", maxItems: 5, items: reasonCodeSchema },
    modelVersion: { type: "string" },
    policyVersion: { type: "string" },
    outOfDistribution: { type: "array", items: { type: "string" } },
    flags: { type: "array", items: { type: "string" } }
  }
} as const;

export const errorSchema = {
  type: "object",
  additionalProperties: false,
  required: ["error", "message", "requestId"],
  properties: {
    error: { type: "string" },
    message: { type: "string" },
    requestId: { type: "string" },
    details: { type: "array", items: { type: "object", additionalProperties: true } }
  }
} as const;

export const healthResponseSchema = {
  type: "object",
  additionalProperties: false,
  required: ["status", "service", "version", "uptimeSeconds", "timestamp"],
  properties: {
    status: { type: "string", enum: ["ok"] },
    service: { type: "string" },
    version: { type: "string" },
    uptimeSeconds: { type: "integer", minimum: 0 },
    timestamp: { type: "string", format: "date-time" }
  }
} as const;

export const readyResponseSchema = {
  type: "object",
  additionalProperties: false,
  required: ["status", "modelLoaded", "version"],
  properties: {
    status: { type: "string", enum: ["ready", "not-ready"] },
    modelLoaded: { type: "boolean" },
    version: { type: "string" },
    model: { type: "string" }
  }
} as const;

export const scoreResponseSchema = {
  type: "object",
  additionalProperties: false,
  required: ["requestId", "apiVersion", "result"],
  properties: {
    requestId: { type: "string" },
    apiVersion: { type: "string" },
    result: riskResultSchema
  }
} as const;

export const stressRequestSchema = {
  type: "object",
  additionalProperties: false,
  required: ["application", "severity"],
  properties: {
    application: applicationSchema,
    severity: { type: "string", enum: ["mild", "severe"] }
  }
} as const;

export const stressResponseSchema = {
  type: "object",
  additionalProperties: false,
  required: ["requestId", "apiVersion", "severity", "input", "baseline", "stressed", "delta"],
  properties: {
    requestId: { type: "string" },
    apiVersion: { type: "string" },
    severity: { type: "string", enum: ["mild", "severe"] },
    input: applicationSchema,
    baseline: riskResultSchema,
    stressed: riskResultSchema,
    delta: {
      type: "object",
      additionalProperties: false,
      required: ["pd", "expectedLoss", "score", "decisionChanged"],
      properties: {
        pd: { type: "number" },
        expectedLoss: { type: "number" },
        score: { type: "integer" },
        decisionChanged: { type: "boolean" }
      }
    }
  }
} as const;

export const batchRequestSchema = {
  type: "object",
  additionalProperties: false,
  required: ["applications"],
  properties: {
    applications: { type: "array", minItems: 1, maxItems: MAX_BATCH_SIZE, items: applicationSchema }
  }
} as const;

export const batchResponseSchema = {
  type: "object",
  additionalProperties: false,
  required: ["requestId", "apiVersion", "results", "summary"],
  properties: {
    requestId: { type: "string" },
    apiVersion: { type: "string" },
    results: {
      type: "array",
      maxItems: MAX_BATCH_SIZE,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["index", "result"],
        properties: {
          index: { type: "integer", minimum: 0 },
          applicationId: { type: "string" },
          result: riskResultSchema
        }
      }
    },
    summary: {
      type: "object",
      additionalProperties: false,
      required: ["count", "decisions", "averagePd", "totalExpectedLoss"],
      properties: {
        count: { type: "integer", minimum: 1, maximum: MAX_BATCH_SIZE },
        decisions: {
          type: "object",
          additionalProperties: false,
          required: ["APPROVE", "REVIEW", "DECLINE"],
          properties: {
            APPROVE: { type: "integer", minimum: 0 },
            REVIEW: { type: "integer", minimum: 0 },
            DECLINE: { type: "integer", minimum: 0 }
          }
        },
        averagePd: { type: "number", minimum: 0, maximum: 1 },
        totalExpectedLoss: { type: "number", minimum: 0 }
      }
    }
  }
} as const;

export const modelResponseSchema = {
  type: "object",
  additionalProperties: false,
  required: ["requestId", "apiVersion", "name", "version", "trainedAt", "features", "monotoneConstraints", "calibration", "metrics", "diagnostics", "treeCount", "policy"],
  properties: {
    requestId: { type: "string" },
    apiVersion: { type: "string" },
    name: { type: "string" },
    version: { type: "string" },
    trainedAt: { type: "string" },
    features: { type: "array", items: { type: "string" } },
    monotoneConstraints: { type: "array", items: { type: "integer" } },
    calibration: {
      type: "object",
      additionalProperties: false,
      required: ["slope", "intercept"],
      properties: { slope: { type: "number" }, intercept: { type: "number" } }
    },
    metrics: {
      type: "object",
      additionalProperties: { type: "number" }
    },
    diagnostics: {
      type: "object",
      additionalProperties: false,
      required: ["calibration", "roc", "featureImportance"],
      properties: {
        calibration: {
          type: "array",
          items: {
            type: "object",
            additionalProperties: false,
            required: ["predicted", "observed", "count"],
            properties: { predicted: { type: "number" }, observed: { type: "number" }, count: { type: "integer" } }
          }
        },
        roc: {
          type: "array",
          items: {
            type: "object",
            additionalProperties: false,
            required: ["fpr", "tpr"],
            properties: { fpr: { type: "number" }, tpr: { type: "number" } }
          }
        },
        featureImportance: {
          type: "array",
          items: {
            type: "object",
            additionalProperties: false,
            required: ["feature", "gain"],
            properties: { feature: { type: "string" }, gain: { type: "number" } }
          }
        }
      }
    },
    treeCount: { type: "integer", minimum: 1 },
    policy: { type: "object", additionalProperties: true }
  }
} as const;
