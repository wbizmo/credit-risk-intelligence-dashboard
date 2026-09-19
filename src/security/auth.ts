import { createHash, timingSafeEqual } from "node:crypto";

const digest = (value: string): Buffer => createHash("sha256").update(value, "utf8").digest();

export interface ApiKeyMatch {
  matched: boolean;
  keySlot?: number;
}

export function matchApiKey(supplied: string, expectedKeys: readonly string[]): ApiKeyMatch {
  const suppliedDigest = digest(supplied);
  let matchedSlot = -1;

  for (let index = 0; index < expectedKeys.length; index += 1) {
    const expectedDigest = digest(expectedKeys[index]!);
    const equal = timingSafeEqual(suppliedDigest, expectedDigest);
    if (equal && matchedSlot < 0) matchedSlot = index;
  }

  return matchedSlot >= 0 ? { matched: true, keySlot: matchedSlot } : { matched: false };
}

export function apiKeyFromHeader(value: string | string[] | undefined): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

export function rateLimitIdentity(
  supplied: string | undefined,
  expectedKeys: readonly string[],
  clientIp: string,
): string {
  if (supplied) {
    const match = matchApiKey(supplied, expectedKeys);
    if (match.matched) return `credential:${match.keySlot! + 1}`;
  }
  return `ip:${clientIp}`;
}
