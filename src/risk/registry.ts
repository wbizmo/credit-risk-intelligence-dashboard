import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

interface RuntimeManifest {
  modelId: string;
  modelName: string;
  version: string;
  artifactSha256: string;
  featureContractVersion: string;
  status: "research" | "challenger" | "approved-demo-champion" | "retired";
}

const ARTIFACT_PATH = resolve(process.cwd(), "model", "artifacts", "crix-monoboost-v2.json");
const MANIFEST_PATH = resolve(process.cwd(), "model", "artifacts", "crix-monoboost-v2.manifest.json");

function loadManifest(): RuntimeManifest {
  const parsed = JSON.parse(readFileSync(MANIFEST_PATH, "utf8")) as Partial<RuntimeManifest>;
  if (
    typeof parsed.modelId !== "string" ||
    typeof parsed.modelName !== "string" ||
    typeof parsed.version !== "string" ||
    typeof parsed.artifactSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(parsed.artifactSha256) ||
    typeof parsed.featureContractVersion !== "string" ||
    !["research", "challenger", "approved-demo-champion", "retired"].includes(parsed.status ?? "")
  ) {
    throw new Error("Invalid CRIX runtime model manifest");
  }
  return parsed as RuntimeManifest;
}

function artifactSha256(): string {
  return createHash("sha256").update(readFileSync(ARTIFACT_PATH)).digest("hex");
}

export function verifyRuntimeArtifactManifest(): boolean {
  try {
    const manifest = loadManifest();
    return manifest.modelId === "CRIX-MonoBoost@2.0.0"
      && manifest.modelName === "CRIX-MonoBoost"
      && manifest.version === "2.0.0"
      && manifest.status === "approved-demo-champion"
      && artifactSha256() === manifest.artifactSha256;
  } catch {
    return false;
  }
}

export function runtimeManifestMetadata() {
  const manifest = loadManifest();
  if (!verifyRuntimeArtifactManifest()) throw new Error("CRIX runtime model manifest verification failed");
  return Object.freeze({
    modelId: manifest.modelId,
    artifactSha256: manifest.artifactSha256,
    status: manifest.status,
    featureContractVersion: manifest.featureContractVersion,
  });
}
