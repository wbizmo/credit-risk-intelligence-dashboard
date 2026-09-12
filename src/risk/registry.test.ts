import { describe, expect, it } from "vitest";
import { runtimeManifestMetadata, verifyRuntimeArtifactManifest } from "./registry";

describe("CRIX runtime model registry", () => {
  it("verifies the deployed champion artifact against its committed manifest", () => {
    expect(verifyRuntimeArtifactManifest()).toBe(true);
  });

  it("exposes only safe registry metadata", () => {
    const metadata = runtimeManifestMetadata();
    expect(metadata.modelId).toBe("CRIX-MonoBoost@2.0.0");
    expect(metadata.status).toBe("approved-demo-champion");
    expect(metadata.artifactSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(metadata.featureContractVersion).toBeTruthy();
    expect(metadata).not.toHaveProperty("trees");
    expect(metadata).not.toHaveProperty("datasetPath");
  });
});
