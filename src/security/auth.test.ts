import { describe, expect, it } from "vitest";
import { apiKeyFromHeader, matchApiKey, rateLimitIdentity } from "./auth";

const keys = [
  "crix-current-key-0123456789abcdef",
  "crix-next-key-0123456789abcdefghi",
] as const;

describe("API-key matching", () => {
  it("accepts both bounded rotation slots and rejects unknown credentials", () => {
    expect(matchApiKey(keys[0], keys)).toEqual({ matched: true, keySlot: 0 });
    expect(matchApiKey(keys[1], keys)).toEqual({ matched: true, keySlot: 1 });
    expect(matchApiKey("definitely-not-an-active-key", keys)).toEqual({ matched: false });
  });

  it("uses only a bounded slot identity for authenticated rate limiting", () => {
    expect(rateLimitIdentity(keys[0], keys, "127.0.0.1")).toBe("credential:1");
    expect(rateLimitIdentity(keys[1], keys, "127.0.0.1")).toBe("credential:2");
    expect(rateLimitIdentity("unknown", keys, "203.0.113.9")).toBe("ip:203.0.113.9");
  });

  it("accepts only a single string API-key header", () => {
    expect(apiKeyFromHeader(keys[0])).toBe(keys[0]);
    expect(apiKeyFromHeader([keys[0]])).toBeUndefined();
    expect(apiKeyFromHeader(undefined)).toBeUndefined();
  });
});
